# BASIWAN v2 — From-Scratch Q4_K + Q6_K CUDA Kernel for Wan2.2

## Mission

Replace `tools/gguf_vendor/q4_marlin/q4k_q6k_marlin.cu` with a ground-up rewrite
that targets the structural ceilings the current ("v1") kernel hits. Aim to
ship a kernel that is **measurably faster** than v1 on Wan production M values
(M ∈ {7808, 18000, 32400, 75600}) at bit-identical output quality.

## Why a v2 — the structural ceilings v1 hits

### Hard ceiling #1: register exhaustion at BLOCK_M=128

ptxas (-Xptxas=-v) on v1 Q4_K BLOCK_M=128 reports:
- 80 bytes stack frame, 80 bytes spill stores, 84 bytes spill loads
- 128 registers/thread (at the 2 CTAs/SM cap)

Any attempt to add registers (epilogue coalescing, warp-spec, larger acc tiles)
either (a) cascades into MORE spilling on the constrained path, or (b) drops
occupancy below 2 CTAs/SM. v1 has zero slack.

### Hard ceiling #2: warp-spec is architecturally net-negative at 4+4

Measured across 4 warp-spec variants:
4-warp consumer × 2× MMA work serially vs 8-warp baseline parallel MMA
fundamentally halves throughput. The 4+4 split shape doesn't fit this kernel.

The 6+2 split (consumer covers BLOCK_M=96 or 192) would preserve more MMA
throughput, but requires retuning tile dispatch. v1 is locked to BLOCK_M ∈ {64, 128}.

### Hard ceiling #3: epilogue is scalar-store-only

The scalar 4-store-per-(mf,nf) epilogue is the only viable path under v1's
register budget. Packed `cvt.f16x2.f32` or `pack_pair` add temporaries that
cascade into MORE spilling (measured: 80 → 192 byte stack frame, +12% wall).

### Hard ceiling #4: per-token modulation kernel attempts hit diffusion's
bf16-rounding sensitivity

A Triton fused per-token LN+modulation kernel is numerically
more precise than the eager 3-bf16-quantization path but produces -0.0115
composite Q regression. The diffusion model has adapted to the eager
rounding pattern. Anything outside that rounding sequence drifts noise
predictions. This is downstream of Marlin but constrains the broader pipeline.

## What v2 changes

### 1. Real warp specialization — 6+2 split with BLOCK_M=192

- 8 warps per CTA: 6 consumer + 2 producer
- BLOCK_M=192 (6 × 32 mfrags via mma.m16n8k16, 2 mfrags per consumer warp)
- BLOCK_N=64 (no change — keep L2-optimal dispatch)
- BLOCK_K=32 (no change — fits Q4_K superblock granularity)
- WARPS_N=1 (full N tile per warp, 8 nfrag × 8 cols = 64 cols per warp)
- WARPS_M=6 (consumer warps split BLOCK_M=192 along M axis)

Per-consumer-warp MMA work per kt:
- mfrag × nfrag × kfrag = 2 × 8 × 2 = 32 mma calls
- 6 consumer warps × 32 mma = 192 mma per kt per CTA
- Per CTA, BLOCK_M × BLOCK_N = 192 × 64 = 12288 output cells per kt

Producer warps (2 of 8): dedicated to dequant. 32-thread × 2 warps = 64 producer
threads, doing 4× more work per thread than the baseline's 256 threads to fully
keep the consumer fed. Should still match the consumer's compute throughput
because dequant is bandwidth-bound, not thread-count-bound.

### 2. Fused dequant directly into ldmatrix-format SMEM

v1 dequants Q4_K → fp16 SMEM, then ldmatrix reads it back. v2 has the producer
warps write the dequanted values DIRECTLY in the ldmatrix.x4 layout (interleaved
8-row tiles, swizzled to avoid bank conflicts). Eliminates one SMEM round-trip.

### 3. Multi-stage cp.async pipeline (4 stages, with mbarrier on Hopper-style
SMEM tagging if possible)

v1 uses STAGES=3 with `cp.async.cg`. v2:
- STAGES=4 (deeper pipeline, more HBM latency hiding)
- Use `cp.async.ca` for B (the larger object) for better L2 hit
- Use `cp.async.cg` for A and scales (small objects, cache-bypass)
- Async barrier scheme that doesn't require full-CTA __syncthreads between stages

### 4. Native mma instruction with bf16 accumulator option

v1 always accumulates in fp32. For some pathways (e.g., when the downstream
ops will quantize to bf16 anyway), accumulating in bf16 is acceptable and
faster. v2 templates on the accumulator dtype.

### 5. M-tile padding handled inside the kernel (no host-side padding pool)

v1's `BareGGMLLinear` does host-side M-padding to a multiple of 64. v2's kernel
handles non-aligned M natively (boundary CTAs check `tile_m_base + lane_m < M`).
Eliminates the host-side padding-pool allocator pressure.

### 6. L2 hint instructions on cp.async

v1 uses default cache mode. v2 uses `cp.async.cg` (cache-global, no L1) for
small object loads and `cp.async.ca` (cache-all, L1+L2) for B (large, reused
across M-tiles). Should improve L2 hit on B.

### 7. SMEM swizzle for bank-conflict-free ldmatrix.x4

v1 has bank conflicts on the A read path (8 threads of a warp read the same
SMEM bank simultaneously, causing replays). v2 swizzles SMEM layout so that
consecutive lanes of an mma fragment hit DIFFERENT banks.

## Status

Shipped, gated behind `BASIWAN_V2=1` (opt-in default-on for sm_89) in
`tools/gguf_vendor/q4_marlin/__init__.py`. The v2 kernel is a drop-in API match
for `marlin_q4k_q6k_gemm`, verified **bit-identical to v1** against the golden
reference (dequant + `F.linear` at fp32). Measured end-to-end across 3 prompts ×
2 shapes: **-5.3% mean wall at bit-identical CLIP-T** (microbench Q4_K 0.76×,
Q6_K 0.92× of v1).

## Reference materials

- v1 kernel: `tools/gguf_vendor/q4_marlin/q4k_q6k_marlin.cu`
- vllm MMQ kernel (reference for tensor-core Q4_K patterns; the installed vllm
  package's `_C` ggml MMQ ops)
- Original CUTLASS warp specialization examples:
  flash-attention/csrc/cutlass/examples/48_hopper_warp_specialized_gemm/
- Cutlass arch barrier docs:
  flash-attention/csrc/cutlass/arch/barrier.h:157-332
- Marlin paper (the IST-Austria one, not Modal's):
  https://arxiv.org/abs/2408.11743 (specifically the Q4 dequant+mma fusion patterns)
- nvcuda::wmma docs (for legacy reference on tile shapes):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma

## Constraints

- **Bit-identical to dequant + F.linear at fp32 reference**: any departure must be
  measured against composite Q on a 5-prompt e2e (not just unit precision).
- **Ada sm_89 target** (RTX 4090). Forward-port to Hopper sm_90+ later.
- **bf16 + fp16 inputs**. Output dtype matches input.
- **Q4_K group_size=32, Q6_K group_size=16** (per Wan2.2 GGUF).
- **No host-side M padding required** (host can still pad if it wants).
