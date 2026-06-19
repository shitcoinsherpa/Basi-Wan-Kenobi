// Independent reimplementation informed by the Marlin kernel design
// (IST-DASLab/marlin and the vLLM port, both Apache-2.0). Permutation
// tables and the LOP3 fp16 dequant idiom follow Marlin. Citation per the
// authors' request: Frantar et al., arXiv:2408.11743. See CREDITS.md.
/*
 * BASIWAN v2 — from-scratch Q4_K + Q6_K CUDA kernel for Wan2.2.
 *
 * Build state: Phase B.1 — dequant helpers + QuantTag types ported from v1.
 *
 * Architecture target (see docs/DESIGN.md):
 *   - 6+2 warp-spec (6 consumer + 2 producer warps per CTA)
 *   - BLOCK_M=192 (consumer covers 2× 16-row mfrags = 32 rows per warp)
 *   - BLOCK_N=64, BLOCK_K=32, STAGES=4
 *   - Fused dequant directly into ldmatrix.x4 SMEM layout
 *
 * The dequant routines below are ported (clean copy) from v1's
 * tools/gguf_vendor/q4_marlin/q4k_q6k_marlin.cu. They are battle-tested
 * against gguf.quants's reference and bit-identical to dequant + F.linear.
 */

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "perm_inverse.cuh"  // brings in basiwan_v2::c_PERM_INV[1024]
// perm_forward.cuh (c_PERM[1024]) was for the D.2-FIX-v2 forward scatter that
// was rejected by measurement (atomicOr serializes SMEM ~50K cycles/K-tile;
// see marlin_v2_phase_d2_fix_partial_win_2026-06-06.md). Header stays in repo
// for future bf16-rounding-matched re-eval; not included in current build.

namespace basiwan_v2 {

// ---- Constants ----
constexpr int TILE = 16;          // m16n8k16 fragment edge
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_CTA = 8;
constexpr int THREADS_PER_CTA = WARPS_PER_CTA * WARP_SIZE;  // 256

// Warp specialization roles — 6+2 split (vs v1's 4+4 net-negative split).
constexpr int PRODUCER_WARPS = 2;
constexpr int CONSUMER_WARPS = WARPS_PER_CTA - PRODUCER_WARPS;  // 6
constexpr int PRODUCER_WARP_END = PRODUCER_WARPS;
constexpr int CONSUMER_WARP_BASE = PRODUCER_WARPS;

// Target tile shape.
constexpr int BLOCK_M_TARGET = 192;  // CONSUMER_WARPS × 32 = 192
constexpr int BLOCK_N_TARGET = 64;
constexpr int BLOCK_K_TARGET = 32;
constexpr int STAGES_TARGET = 4;

// ---- mma.sync m16n8k16 wrappers ----
//
// PTX: mma.sync.aligned.m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32
//   - 16 rows × 8 cols × 16 K accumulation per warp per call.
//   - A: row-major; per-lane fragment = 4 uint32 = 8 fp16 (or bf16).
//   - B: col-major; per-lane fragment = 2 uint32 = 4 fp16 (or bf16).
//   - C/D: fp32 accumulator; per-lane = 4 fp32.
__device__ __forceinline__ void mma_m16n8k16_f32_fp16(
    float& c0, float& c1, float& c2, float& c3,
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1));
}
__device__ __forceinline__ void mma_m16n8k16_f32_bf16(
    float& c0, float& c1, float& c2, float& c3,
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        :  "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),
           "r"(b0),  "r"(b1));
}
template<typename DataT>
__device__ __forceinline__ void mma_m16n8k16_f32(
    float& c0, float& c1, float& c2, float& c3,
    const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
    const uint32_t b0, const uint32_t b1
) {
    if constexpr (std::is_same<DataT, half>::value) {
        mma_m16n8k16_f32_fp16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    } else {
        mma_m16n8k16_f32_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1);
    }
}

// ---- LOP3 helper for nibble extraction ----
//
// lop3.b32 computes (op_a, op_b, op_c) → out per the immediate lut.
// We use it to extract a nibble field and place it in a half-precision exponent slot.
template<int lut>
__device__ __forceinline__ uint32_t lop3(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t res;
    asm("lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// ---- Dtype conversions ----
template<typename DataT>
__device__ __forceinline__ DataT float_to_data(float x);
template<>
__device__ __forceinline__ half float_to_data<half>(float x) {
    return __float2half(x);
}
template<>
__device__ __forceinline__ __nv_bfloat16 float_to_data<__nv_bfloat16>(float x) {
    return __float2bfloat16_rn(x);
}
template<typename DataT>
__device__ __forceinline__ float data_to_float(DataT x);
template<>
__device__ __forceinline__ float data_to_float<half>(half x) {
    return __half2float(x);
}
template<>
__device__ __forceinline__ float data_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template<typename DataT>
__device__ __forceinline__ DataT int_to_data(int x);
template<>
__device__ __forceinline__ half int_to_data<half>(int x) {
    return __int2half_rn(x);
}
template<>
__device__ __forceinline__ __nv_bfloat16 int_to_data<__nv_bfloat16>(int x) {
    return __float2bfloat16_rn(static_cast<float>(x));
}

// ---- Pack a DataT pair into a uint32 (for ldmatrix-friendly sh_B_dequant layout) ----
template<typename DataT>
__device__ __forceinline__ uint32_t pack_pair(DataT lo, DataT hi);
template<>
__device__ __forceinline__ uint32_t pack_pair<half>(half lo, half hi) {
    half2 h2 = __halves2half2(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h2);
}
template<>
__device__ __forceinline__ uint32_t pack_pair<__nv_bfloat16>(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    __nv_bfloat162 b2 = __halves2bfloat162(lo, hi);
    return *reinterpret_cast<uint32_t*>(&b2);
}

// ---- Q4_K inline dequant (ported from v1) ----
//
// Input: q_packed = 8 4-bit nibbles in a uint32 (lo bits hold nibble 0).
//        eff_scale: half2 holding (sub_scale, sub_scale) for this sub-block.
//        eff_min:   half2 holding (sub_min,   sub_min)   for this sub-block.
// Output: 4 half2 values = 8 fp16 nibbles dequantized as (n × scale - min).
//
// Marlin's LOP3 trick: place the 4-bit nibble in the fp16 exponent
// field of `0x6400` (= 1024.0). Then `fp16(0x6400 | nibble) - fp16(1024.0)` gives
// the value as fp16 nibble in [0..15]. Multiply by eff_scale, subtract eff_min.
//
// Note: this is asymmetric (no -8 fold). Our scales already encode the asymmetry.
__device__ __forceinline__ void dequant_q4k_8_to_half2(
    uint32_t q_packed,
    half2 eff_scale,
    half2 eff_min,
    half2 out[4]
) {
    constexpr int LO = 0x000f000f;
    constexpr int HI = 0x00f000f0;
    constexpr int EX = 0x64006400;

    uint32_t lo0 = lop3<0xca>(q_packed,        LO, EX);
    uint32_t hi0 = lop3<0xca>(q_packed,        HI, EX);
    uint32_t lo1 = lop3<0xca>(q_packed >> 8,   LO, EX);
    uint32_t hi1 = lop3<0xca>(q_packed >> 8,   HI, EX);

    constexpr int SUB = 0x64006400;
    half2 hsub = *reinterpret_cast<const half2*>(&SUB);

    half2 v0 = __hsub2(*reinterpret_cast<half2*>(&lo0), hsub);
    half2 v1 = __hsub2(*reinterpret_cast<half2*>(&hi0), hsub);
    half2 v2 = __hsub2(*reinterpret_cast<half2*>(&lo1), hsub);
    half2 v3 = __hsub2(*reinterpret_cast<half2*>(&hi1), hsub);

    constexpr int INV16 = 0x2c002c00;  // fp16(1/16) packed
    half2 hinv16 = *reinterpret_cast<const half2*>(&INV16);
    v1 = __hmul2(v1, hinv16);
    v3 = __hmul2(v3, hinv16);

    half2 neg_min = __hneg2(eff_min);
    out[0] = __hfma2(v0, eff_scale, neg_min);
    out[1] = __hfma2(v1, eff_scale, neg_min);
    out[2] = __hfma2(v2, eff_scale, neg_min);
    out[3] = __hfma2(v3, eff_scale, neg_min);
}

// ---- Q6_K inline dequant (ported from v1) ----
__device__ __forceinline__ void dequant_q6k_8_to_half2(
    uint32_t ql_packed,
    uint32_t qh_packed,
    half2 eff_scale,
    half2 out[4]
) {
    constexpr int LO = 0x000f000f;
    constexpr int HI = 0x00f000f0;
    constexpr int EX = 0x64006400;

    uint32_t lo0 = lop3<0xca>(ql_packed,      LO, EX);
    uint32_t hi0 = lop3<0xca>(ql_packed,      HI, EX);
    uint32_t lo1 = lop3<0xca>(ql_packed >> 8, LO, EX);
    uint32_t hi1 = lop3<0xca>(ql_packed >> 8, HI, EX);

    constexpr int SUB_BASE = 0x64006400;
    half2 hsub_base = *reinterpret_cast<const half2*>(&SUB_BASE);
    half2 v_ql0 = __hsub2(*reinterpret_cast<half2*>(&lo0), hsub_base);
    half2 v_qh0 = __hsub2(*reinterpret_cast<half2*>(&hi0), hsub_base);
    half2 v_ql1 = __hsub2(*reinterpret_cast<half2*>(&lo1), hsub_base);
    half2 v_qh1 = __hsub2(*reinterpret_cast<half2*>(&hi1), hsub_base);

    // hi nibbles came in ×16, divide.
    constexpr int INV16 = 0x2c002c00;
    half2 hinv16 = *reinterpret_cast<const half2*>(&INV16);
    v_qh0 = __hmul2(v_qh0, hinv16);
    v_qh1 = __hmul2(v_qh1, hinv16);

    // qh contribution: 2 bits per value.
    uint32_t qh_bits01 = qh_packed & 0xFu;
    uint32_t qh_bits23 = (qh_packed >> 4) & 0xFu;
    uint32_t qh_bits45 = (qh_packed >> 8) & 0xFu;
    uint32_t qh_bits67 = (qh_packed >> 12) & 0xFu;
    auto build_qh16 = [](uint32_t pair) -> half2 {
        int va = pair & 0x3;
        int vb = (pair >> 2) & 0x3;
        half ha = __float2half(static_cast<float>(va * 16));
        half hb = __float2half(static_cast<float>(vb * 16));
        return __halves2half2(ha, hb);
    };
    half2 qh16_01 = build_qh16(qh_bits01);
    half2 qh16_23 = build_qh16(qh_bits23);
    half2 qh16_45 = build_qh16(qh_bits45);
    half2 qh16_67 = build_qh16(qh_bits67);

    constexpr int SUB_32 = 0xd000d000;  // fp16(-32) packed
    half2 hsub32 = *reinterpret_cast<const half2*>(&SUB_32);

    half2 v0 = __hadd2(__hadd2(v_ql0, qh16_01), hsub32);
    half2 v1 = __hadd2(__hadd2(v_qh0, qh16_23), hsub32);
    half2 v2 = __hadd2(__hadd2(v_ql1, qh16_45), hsub32);
    half2 v3 = __hadd2(__hadd2(v_qh1, qh16_67), hsub32);

    out[0] = __hmul2(v0, eff_scale);
    out[1] = __hmul2(v1, eff_scale);
    out[2] = __hmul2(v2, eff_scale);
    out[3] = __hmul2(v3, eff_scale);
}

// ---- Quant-type tag ----
struct Q4K {};
struct Q6K {};

// ---- cp.async wrappers (sm_80+, used in Phase B.2+) ----
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 16;\n"  // .ca = cache all (L1+L2) — better for B reuse
        :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_16_cg(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"  // .cg = cache global (L2 only) — for A/scales
        :: "r"(smem_addr), "l"(gmem_ptr));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ uint32_t cvt_to_smem_ptr(void* p) {
    // Use CUDA's built-in intrinsic which handles the 64→32 narrowing correctly.
    // Inline-PTX `cvta.to.shared.u64` returns u64 and needs `=l` constraint;
    // the intrinsic returns the 32-bit SMEM byte offset directly.
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ============================================================================
// Phase B.2: marlin_kernel_v2 template skeleton
// ============================================================================
//
// Architecture: 6+2 warp specialization.
//   - 8 warps per CTA. Warps 0-1: producer (dequant). Warps 2-7: consumer (MMA + epilogue).
//   - BLOCK_M=192 (consumer split: WARPS_M=6 × WARP_TILE_M=32; each warp owns 2 mfrags).
//   - BLOCK_N=64 (full N per warp; WARPS_N=1; 8 nfrags per consumer warp).
//   - BLOCK_K=32 (Q4_K group_size aligned).
//   - STAGES=4 cp.async stages.
//
// Phase B.2 is the SKELETON only. Static_asserts hold; SMEM allocation correct;
// kernel launches without crashing. K-loop body / mma / ldmatrix come in Phase B.3+.
// To confirm the kernel runs end-to-end, Phase B.2 writes zeros to C in the
// expected per-lane fragment layout — letting downstream correctness tests
// verify the indexing math BEFORE the MMA logic lands.
template<typename QuantTag, typename DataT,
         int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES, int GROUP_SIZE>
__global__ void marlin_kernel_v2(
    const DataT* __restrict__ A,
    const int*   __restrict__ B_ql,
    const int*   __restrict__ B_qh,
    const DataT* __restrict__ eff_scales,
    const DataT* __restrict__ eff_mins,
    DataT*       __restrict__ C,
    int M, int N, int K
) {
    constexpr int WARPS_M = CONSUMER_WARPS;          // 6
    constexpr int WARPS_N = 1;                       // full N per consumer warp
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;
    constexpr int MMA_M_PER_WARP = WARP_TILE_M / MMA_M;
    constexpr int MMA_N_PER_WARP = WARP_TILE_N / MMA_N;
    constexpr int K_FRAGS_PER_KT = BLOCK_K / MMA_K;
    constexpr int SCALES_PER_KT = (BLOCK_K + GROUP_SIZE - 1) / GROUP_SIZE;

    static_assert(BLOCK_M % WARPS_M == 0, "BLOCK_M must divide WARPS_M=6 for 6+2 layout");
    static_assert(WARP_TILE_M % MMA_M == 0, "WARP_TILE_M must be multiple of MMA_M=16");
    static_assert(BLOCK_N % WARPS_N == 0, "BLOCK_N must divide WARPS_N");
    static_assert(WARP_TILE_N % MMA_N == 0, "WARP_TILE_N must be multiple of MMA_N=8");
    static_assert(BLOCK_K % MMA_K == 0, "BLOCK_K must be multiple of MMA_K=16");
    static_assert(SCALES_PER_KT * GROUP_SIZE == BLOCK_K, "BLOCK_K must align with GROUP_SIZE");

    // === Tile coords ===
    // Tested D.3 L2 supergroup swizzle (GROUP_M=8) — measured +7-10% Q4_K wall
    // regression and only modest Q6_K improvement (1.21x → 1.16x). Default
    // (pid_m=blockIdx.y, pid_n=blockIdx.x) is already L2-friendly enough.
    const int pid_m = blockIdx.y;
    const int pid_n = blockIdx.x;
    const int tile_m_base = pid_m * BLOCK_M;
    const int tile_n_base = pid_n * BLOCK_N;
    if (tile_m_base >= M || tile_n_base >= N) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const bool is_producer = warp_id < PRODUCER_WARPS;
    const bool is_consumer = !is_producer;
    const int cons_warp_local = warp_id - CONSUMER_WARP_BASE;
    const int warp_m = cons_warp_local;  // each consumer warp owns one M strip
    const int warp_n = 0;                // WARPS_N=1 → all consumer warps own full N

    // === SMEM allocation ===
    //
    // SMEM layout (offsets are cumulative bytes):
    //   sh_A           STAGES × BLOCK_M × (BLOCK_K+pad) DataT — cp.async staged
    //   sh_B_ql        STAGES × (BLOCK_K/16) × (BLOCK_N*2) int32 — packed weights, cp.async staged
    //   sh_B_qh        STAGES × (BLOCK_K/16) × BLOCK_N    int32 — Q6_K high-2-bit, staged (Q6_K only)
    //   sh_scales      STAGES × SCALES_PER_KT × BLOCK_N DataT — per-kt scales, staged
    //   sh_mins        STAGES × SCALES_PER_KT × BLOCK_N DataT — per-kt mins, staged (Q4_K only)
    //   sh_B_dequant   2 × BLOCK_K × BLOCK_N DataT — producer-written, consumer-read (parity)
    //
    // The staged regions (sh_A, sh_B_ql, sh_B_qh, sh_scales, sh_mins) participate
    // in the cp.async pipeline. The sh_B_dequant region is double-buffered for
    // producer/consumer parity exchange (it's computed by the producer warps from
    // the corresponding staged sh_B_ql + sh_scales + sh_mins, not loaded from gmem).
    extern __shared__ uint8_t smem[];
    constexpr int A_STRIDE = BLOCK_K + 8;  // +8 for swizzle padding to avoid bank conflicts
    constexpr int A_STAGE_BYTES = BLOCK_M * A_STRIDE * sizeof(DataT);
    constexpr int A_BYTES = STAGES * A_STAGE_BYTES;

    constexpr int B_QL_ROWS_PER_KT = BLOCK_K / 16;     // 32/16 = 2 marlin-K groups per kt
    constexpr int B_QL_COLS_PER_KT = BLOCK_N * 2;      // marlin Q4_K layout: N*2 int32 per row
    constexpr int B_QL_STAGE_BYTES = B_QL_ROWS_PER_KT * B_QL_COLS_PER_KT * sizeof(int);
    constexpr int B_QL_BYTES = STAGES * B_QL_STAGE_BYTES;

    constexpr int B_QH_ROWS_PER_KT = BLOCK_K / 16;
    constexpr int B_QH_COLS_PER_KT = BLOCK_N;          // marlin Q6_K layout: N int32 per row
    constexpr int B_QH_STAGE_BYTES = std::is_same<QuantTag, Q6K>::value
                                         ? (B_QH_ROWS_PER_KT * B_QH_COLS_PER_KT * sizeof(int))
                                         : 0;
    constexpr int B_QH_BYTES = STAGES * B_QH_STAGE_BYTES;

    constexpr int SCALES_STAGE_BYTES = SCALES_PER_KT * BLOCK_N * sizeof(DataT);
    constexpr int SCALES_BYTES = STAGES * SCALES_STAGE_BYTES;
    constexpr int MINS_BYTES = std::is_same<QuantTag, Q4K>::value ? SCALES_BYTES : 0;

    // sh_B_dequant uses BASIWAN v1's pair layout: K_PAIRS_PER_TILE rows of
    // (BLOCK_N + 1) uint32 each. The "+1" column pad is critical for bank
    // conflict avoidance during consumer ldmatrix reads; without it, lanes
    // within a warp hit the same SMEM bank and the reads serialize. v1 has
    // B_PAIR_STRIDE_U32 = BLOCK_N + 1 and our first attempt with stride=BLOCK_N
    // produced wildly wrong output (max_abs ~125 vs 5e-4 expected).
    constexpr int B_PAIR_STRIDE_U32_LOC = BLOCK_N + 1;
    constexpr int BPAIR_SINGLE_U32_LOC = (BLOCK_K / 2) * B_PAIR_STRIDE_U32_LOC;
    constexpr int B_DEQUANT_BYTES = 2 * BPAIR_SINGLE_U32_LOC * sizeof(uint32_t);

    // Phase D.2-FIX: pre-unpacked qh layout (Q6_K only).
    // sh_B_qh_unpacked[stage][k_pair][n_inner] = uint32 with bits 0-1 = nib_qh_lo,
    // bits 2-3 = nib_qh_hi. The cooperative_unpack_qh_kt pass populates this from
    // sh_B_qh (which has the conflict-prone access pattern).
    // The dequant pass then reads ONE uint32 per pair — 32 threads × 4 bytes =
    // 32 banks accessed in parallel = ZERO conflict.
    // Size: STAGES × (BLOCK_K/2) × BLOCK_N × 4 bytes = 4 × 16 × 64 × 4 = 16 KB.
    constexpr int BQH_UNPACK_INTS_PER_KT_LOC = (BLOCK_K / 2) * BLOCK_N;
    constexpr int BQH_UNPACK_BYTES = std::is_same<QuantTag, Q6K>::value
                                         ? (STAGES * BQH_UNPACK_INTS_PER_KT_LOC * (int)sizeof(uint32_t))
                                         : 0;

    DataT* sh_A = reinterpret_cast<DataT*>(smem);
    int*   sh_B_ql = reinterpret_cast<int*>(smem + A_BYTES);
    int*   sh_B_qh = std::is_same<QuantTag, Q6K>::value
                         ? reinterpret_cast<int*>(smem + A_BYTES + B_QL_BYTES)
                         : nullptr;
    DataT* sh_scales = reinterpret_cast<DataT*>(smem + A_BYTES + B_QL_BYTES + B_QH_BYTES);
    DataT* sh_mins = std::is_same<QuantTag, Q4K>::value
                         ? reinterpret_cast<DataT*>(smem + A_BYTES + B_QL_BYTES + B_QH_BYTES + SCALES_BYTES)
                         : nullptr;
    DataT* sh_B_dequant = reinterpret_cast<DataT*>(
        smem + A_BYTES + B_QL_BYTES + B_QH_BYTES + SCALES_BYTES + MINS_BYTES);
    uint32_t* sh_B_qh_unpacked = std::is_same<QuantTag, Q6K>::value
                                     ? reinterpret_cast<uint32_t*>(
                                         smem + A_BYTES + B_QL_BYTES + B_QH_BYTES
                                         + SCALES_BYTES + MINS_BYTES + B_DEQUANT_BYTES)
                                     : nullptr;
    (void)sh_B_qh_unpacked;
    (void)sh_A; (void)sh_B_ql; (void)sh_B_qh; (void)sh_scales; (void)sh_mins; (void)sh_B_dequant;

    // === Accumulator (consumer only) — declared but not yet used ===
    float acc[MMA_M_PER_WARP][MMA_N_PER_WARP][4] = {};
    (void)acc;

    // === Phase B.3.1: cp.async A loader ===
    //
    // A loader pattern (full-CTA participation for maximum issuance throughput):
    //   - Per stage, load BLOCK_M=192 × BLOCK_K=32 fp16/bf16 = 6144 elements = 12 KB
    //   - cp.async.16 transfers 16 bytes = 8 fp16 per call
    //   - 256 threads × 8 fp16 = 2048 fp16 per pass = 64 rows per pass
    //   - 3 passes per stage covers 192 rows
    //   - Each row contains 32 fp16 = 4 cp.async chunks (cols 0,8,16,24)
    //   - Thread layout: t=(0..255), pass=(0..2), linear=t+pass*256
    //     row_local = linear / 4, col_local_chunks = linear % 4, col_local = chunks * 8
    //
    // A is shared across all CTAs that own the same M tile (N tiles in same M row),
    // so we use cp.async.cg (cache global = L2 only); A tile fits in L2 (~1.3 MB).
    constexpr int A_ELEMS_PER_CALL = 8;
    constexpr int A_CHUNKS_PER_ROW = BLOCK_K / A_ELEMS_PER_CALL;  // 32/8 = 4
    constexpr int A_THREADS_PER_PASS = THREADS_PER_CTA;
    constexpr int A_ROWS_PER_PASS = (A_THREADS_PER_PASS) / A_CHUNKS_PER_ROW;  // 256/4 = 64
    constexpr int A_PASSES_PER_STAGE = BLOCK_M / A_ROWS_PER_PASS;  // 192/64 = 3
    static_assert(A_PASSES_PER_STAGE * A_ROWS_PER_PASS == BLOCK_M,
                  "A loader: passes × rows_per_pass must equal BLOCK_M");
    static_assert(A_CHUNKS_PER_ROW * A_ELEMS_PER_CALL == BLOCK_K,
                  "A loader: chunks × elems_per_call must equal BLOCK_K");

    // === Phase B.3.2: cp.async B_ql loader ===
    //
    // B_ql gmem layout: (K/16, N*2) int32. Each row covers 16 K positions; each
    // int32 packs 8 nibbles (for Q4_K) or 8 ql-low-4-bit values (for Q6_K).
    // Per kt: BLOCK_K/16 = 2 rows × BLOCK_N*2 = 128 int32 = 256 int32 per stage.
    // With 256 threads × 4 bytes per int32, that's 1 pass per stage.
    //
    // B is per-CTA-unique (each CTA owns a unique N-tile band; B_ql cols are tile_n_base*2..).
    // Use cp.async.cg (L2 only — no L1 pollution for this per-CTA stream).
    constexpr int B_QL_INTS_PER_KT = B_QL_ROWS_PER_KT * B_QL_COLS_PER_KT;  // 2 × 128 = 256
    constexpr int B_QL_PASSES_PER_STAGE = (B_QL_INTS_PER_KT + THREADS_PER_CTA - 1) / THREADS_PER_CTA;
    static_assert(B_QL_INTS_PER_KT <= 4 * THREADS_PER_CTA,
                  "B_ql loader: too many int32 per stage to load in <=4 passes");

    auto issue_B_ql_kt = [&](int kt_target, int dst_stage) {
        // For Q4_K, B_ql global addressing:
        //   B_ql[(kt_target * B_QL_ROWS_PER_KT + row_local) * (N * 2)
        //        + (tile_n_base * 2 + col_local)]
        // Each thread loads ONE int32 per pass.
        #pragma unroll
        for (int pass = 0; pass < B_QL_PASSES_PER_STAGE; ++pass) {
            const int linear = tid + pass * THREADS_PER_CTA;
            if (linear >= B_QL_INTS_PER_KT) break;
            const int row_local = linear / B_QL_COLS_PER_KT;
            const int col_local = linear % B_QL_COLS_PER_KT;
            const int g_row = kt_target * B_QL_ROWS_PER_KT + row_local;
            const int g_col = tile_n_base * 2 + col_local;
            const int* gmem_src = &B_ql[(size_t)g_row * (N * 2) + g_col];
            int* smem_dst = &sh_B_ql[dst_stage * B_QL_INTS_PER_KT
                                    + row_local * B_QL_COLS_PER_KT
                                    + col_local];
            // 4 bytes per int32 — use a 16-byte cp.async by grouping 4 int32 per call?
            // For simplicity here, use 16-byte cp.async only when we have 4 contiguous ints
            // to load; otherwise emit a scalar load. For BLOCK_N*2=128 cols, threads
            // 0..31 of a warp load 32 consecutive int32 = 128 bytes = 8 cp.async.16 calls
            // (one per 4-int group). Using one cp.async.16 per 4 threads.
            //
            // Phase B.3.2 takes the simpler path: one cp.async.16 per thread loading 4 int32
            // — but only the first thread of each 4-thread group issues, covering 4 elements.
            // Threads where (linear % 4) != 0 skip the cp.async (their bytes are covered
            // by the group leader). This keeps the load coalesced and 16-byte aligned.
            if ((linear & 0x3) == 0) {
                uint32_t smem_addr = cvt_to_smem_ptr(smem_dst);
                cp_async_16_cg(smem_addr, gmem_src);
            }
        }
    };

    // Lambda: issue cp.async for one kt's B_qh tile (Q6_K only) into the target stage.
    auto issue_B_qh_kt = [&](int kt_target, int dst_stage) {
        if (std::is_same<QuantTag, Q6K>::value && sh_B_qh != nullptr && B_qh != nullptr) {
            constexpr int B_QH_INTS_PER_KT_LOC = B_QH_ROWS_PER_KT * B_QH_COLS_PER_KT;  // 2 × 64 = 128 for BLOCK_N=64
            constexpr int B_QH_PASSES_PER_STAGE = (B_QH_INTS_PER_KT_LOC + THREADS_PER_CTA - 1) / THREADS_PER_CTA;
            #pragma unroll
            for (int pass = 0; pass < B_QH_PASSES_PER_STAGE; ++pass) {
                const int linear = tid + pass * THREADS_PER_CTA;
                if (linear >= B_QH_INTS_PER_KT_LOC) break;
                const int row_local = linear / B_QH_COLS_PER_KT;
                const int col_local = linear % B_QH_COLS_PER_KT;
                const int g_row = kt_target * B_QH_ROWS_PER_KT + row_local;
                const int g_col = tile_n_base + col_local;
                const int* gmem_src = &B_qh[(size_t)g_row * N + g_col];
                int* smem_dst = &sh_B_qh[dst_stage * B_QH_INTS_PER_KT_LOC
                                        + row_local * B_QH_COLS_PER_KT
                                        + col_local];
                if ((linear & 0x3) == 0) {
                    uint32_t smem_addr = cvt_to_smem_ptr(smem_dst);
                    cp_async_16_cg(smem_addr, gmem_src);
                }
            }
        }
    };

    // === Phase B.3.3: scales (and Q4_K mins) loader ===
    //
    // Scales / mins gmem layout: (K/GROUP_SIZE, N) DataT. For Q4_K with GROUP_SIZE=32 and
    // BLOCK_K=32, SCALES_PER_KT=1 → 1 × BLOCK_N=64 fp16 = 128 bytes per stage.
    // For Q6_K with GROUP_SIZE=16, SCALES_PER_KT=2 → 2 × 64 = 256 bytes per stage.
    //
    // cp.async.16 transfers 8 fp16 per call. For 64 fp16, 8 threads issue one call each.
    // Distribute across the warp; remaining threads skip.
    constexpr int SCALES_ELEMS_PER_KT = SCALES_PER_KT * BLOCK_N;
    constexpr int SCALES_ELEMS_PER_CALL = 8;
    constexpr int SCALES_CALLS_PER_KT = (SCALES_ELEMS_PER_KT + SCALES_ELEMS_PER_CALL - 1) / SCALES_ELEMS_PER_CALL;

    auto issue_scales_kt = [&](int kt_target, int dst_stage) {
        // gmem addr: eff_scales[(kt_target * SCALES_PER_KT + sc_row) * N + (tile_n_base + sc_col)]
        // For per-thread mapping: thread 0..SCALES_CALLS_PER_KT-1 each issue one call.
        if (tid < SCALES_CALLS_PER_KT) {
            const int sc_row = tid / (BLOCK_N / SCALES_ELEMS_PER_CALL);  // 0..SCALES_PER_KT-1
            const int sc_col_chunk = tid % (BLOCK_N / SCALES_ELEMS_PER_CALL);
            const int sc_col = sc_col_chunk * SCALES_ELEMS_PER_CALL;
            const int g_row = kt_target * SCALES_PER_KT + sc_row;
            const int g_col = tile_n_base + sc_col;
            const DataT* gmem_src = &eff_scales[(size_t)g_row * N + g_col];
            DataT* smem_dst = &sh_scales[dst_stage * SCALES_ELEMS_PER_KT
                                        + sc_row * BLOCK_N
                                        + sc_col];
            uint32_t smem_addr = cvt_to_smem_ptr(smem_dst);
            cp_async_16_cg(smem_addr, gmem_src);
        }
    };

    // Mins (Q4_K only) — same layout as scales, separate gmem array.
    auto issue_mins_kt = [&](int kt_target, int dst_stage) {
        if (std::is_same<QuantTag, Q4K>::value && sh_mins != nullptr && eff_mins != nullptr) {
            if (tid < SCALES_CALLS_PER_KT) {
                const int sc_row = tid / (BLOCK_N / SCALES_ELEMS_PER_CALL);
                const int sc_col_chunk = tid % (BLOCK_N / SCALES_ELEMS_PER_CALL);
                const int sc_col = sc_col_chunk * SCALES_ELEMS_PER_CALL;
                const int g_row = kt_target * SCALES_PER_KT + sc_row;
                const int g_col = tile_n_base + sc_col;
                const DataT* gmem_src = &eff_mins[(size_t)g_row * N + g_col];
                DataT* smem_dst = &sh_mins[dst_stage * SCALES_ELEMS_PER_KT
                                          + sc_row * BLOCK_N
                                          + sc_col];
                uint32_t smem_addr = cvt_to_smem_ptr(smem_dst);
                cp_async_16_cg(smem_addr, gmem_src);
            }
        }
    };

    // Lambda: issue cp.async for one kt's A tile into the target stage of sh_A.
    auto issue_A_kt = [&](int kt_target, int dst_stage) {
        const int kt_col_base = kt_target * BLOCK_K;
        #pragma unroll
        for (int pass = 0; pass < A_PASSES_PER_STAGE; ++pass) {
            const int linear = tid + pass * A_THREADS_PER_PASS;
            const int row_local = linear / A_CHUNKS_PER_ROW;
            const int col_local = (linear % A_CHUNKS_PER_ROW) * A_ELEMS_PER_CALL;
            const int g_row = tile_m_base + row_local;
            const int g_col = kt_col_base + col_local;
            // Per-CTA bounds: row_local < BLOCK_M always (we computed passes accordingly).
            // Per-tensor bounds: only the M boundary needs check (K is multiple of BLOCK_K).
            if (g_row < M) {
                const DataT* gmem_src = &A[(size_t)g_row * K + g_col];
                DataT* smem_dst = &sh_A[dst_stage * BLOCK_M * A_STRIDE
                                       + row_local * A_STRIDE
                                       + col_local];
                uint32_t smem_addr = cvt_to_smem_ptr(smem_dst);
                cp_async_16_cg(smem_addr, gmem_src);
            } else {
                // Out-of-bounds row: zero-fill the SMEM slot so MMA accumulator is correct.
                DataT* smem_dst = &sh_A[dst_stage * BLOCK_M * A_STRIDE
                                       + row_local * A_STRIDE
                                       + col_local];
                #pragma unroll
                for (int e = 0; e < A_ELEMS_PER_CALL; ++e) {
                    smem_dst[e] = float_to_data<DataT>(0.0f);
                }
            }
        }
    };

    // === Phase B.3.4: producer dequant ===
    //
    // 2 producer warps × 32 lanes = 64 producer threads. Each thread dequants
    // PAIRS_PER_PROD = 16 (k, n) pairs per kt. Each pair represents two adjacent
    // K positions × one N position; packed into a single uint32 in the
    // ldmatrix-friendly sh_B_dequant layout.
    //
    // Total pairs per kt: K_PAIRS_PER_TILE × BLOCK_N = (BLOCK_K/2) × BLOCK_N = 16 × 64 = 1024
    // 1024 / 64 producer threads = 16 pairs per thread.
    //
    // For each pair, the dequant uses the marlin permutation
    // (c_PERM_INV[1024]) to translate logical (n_tile_id, k_inner_in_kf, n_in_tile)
    // → physical (int_idx, bit_off) in the packed sh_B_ql.
    constexpr int K_PAIRS_PER_TILE = BLOCK_K / 2;                              // 16
    constexpr int TOTAL_PAIRS = K_PAIRS_PER_TILE * BLOCK_N;                    // 1024
    constexpr int PROD_THREADS = PRODUCER_WARPS * WARP_SIZE;                   // 64
    constexpr int PAIRS_PER_PROD = TOTAL_PAIRS / PROD_THREADS;                 // 16
    static_assert(PAIRS_PER_PROD * PROD_THREADS == TOTAL_PAIRS,
                  "Producer dequant: pairs must divide evenly among producer threads");
    constexpr int B_INTS_PER_KT_LOCAL = B_QL_ROWS_PER_KT * B_QL_COLS_PER_KT;   // 256
    constexpr int B_QH_INTS_PER_KT_LOCAL = std::is_same<QuantTag, Q6K>::value
                                              ? (B_QH_ROWS_PER_KT * B_QH_COLS_PER_KT)
                                              : 0;
    // Match BASIWAN v1's layout: BLOCK_N + 1 columns to avoid SMEM bank conflicts on the
    // consumer ldmatrix path. Without the +1 padding, lanes 0..7 of a warp all
    // hit the same SMEM bank when reading b0/b1 for the same n_in_BLOCK, causing
    // 32x serialized accesses per ldmatrix call.
    constexpr int B_PAIR_STRIDE_U32 = BLOCK_N + 1;
    constexpr int BPAIR_SINGLE_U32 = K_PAIRS_PER_TILE * B_PAIR_STRIDE_U32;
    constexpr int SCALES_ELEMS_PER_KT_LOCAL = SCALES_PER_KT * BLOCK_N;

    // COOPERATIVE dequant (all 256 threads participate, like BASIWAN v1 baseline).
    // Single-buffer sh_B_dequant (no parity — no producer/consumer overlap yet).
    // Phase D will add parity overlap once correctness is proven.
    constexpr int PAIRS_PER_THREAD_COOP = TOTAL_PAIRS / THREADS_PER_CTA;  // 1024/256 = 4
    static_assert(PAIRS_PER_THREAD_COOP * THREADS_PER_CTA == TOTAL_PAIRS,
                  "Cooperative dequant: pairs must divide evenly among CTA threads");

    // Phase D.2-FIX: cooperative unpack pass for Q6_K (the working 1.22× version).
    //
    // Tried D.2-FIX-v2 (forward-perm scatter via atomicOr to scatter from 128
    // contiguous sh_B_qh reads) — REJECTED: 9.94× v1 wall regression. atomicOr
    // serializes per-bank at ~20-30 cycles per op × 2048 ops/K-tile = ~50K cycles,
    // far worse than the conflict-prone reads it replaced. See memory memo
    // marlin_v2_phase_d2_fix_v2_atomicor_reject_2026-06-06.md.
    //
    // Each of 256 threads handles 4 pairs. Per pair: 2 conflict-prone sh_B_qh
    // reads + 1 plain sh_B_qh_unpacked write. The conflict cost is paid ONCE
    // per K-tile (instead of 8× distributed across the dequant pass).
    constexpr int BQH_UNPACK_INTS_PER_KT = K_PAIRS_PER_TILE * BLOCK_N;
    auto cooperative_unpack_qh_kt = [&](int kt_target) {
        if (!std::is_same<QuantTag, Q6K>::value) return;
        const int stage = kt_target % STAGES;
        for (int i = 0; i < PAIRS_PER_THREAD_COOP; ++i) {
            const int lin = tid * PAIRS_PER_THREAD_COOP + i;
            const int k_pair = lin / BLOCK_N;
            const int n_inner = lin % BLOCK_N;
            const int k_lo = 2 * k_pair;
            const int k_hi = k_lo + 1;
            auto extract_nib_qh = [&](int k_inner) -> uint32_t {
                const int kf = k_inner / 16;
                const int k_inner_in_kf = k_inner & 15;
                const int n_tile_id = n_inner / 16;
                const int n_in_tile = n_inner & 15;
                const int orig_flat = n_tile_id * 256 + k_inner_in_kf * 16 + n_in_tile;
                const int permuted = (int)c_PERM_INV[orig_flat];
                const int qh_int_idx = permuted / 16;
                const int qh_bit_off = (permuted & 15) * 2;
                const int packed_qh = sh_B_qh[stage * B_QH_INTS_PER_KT_LOCAL
                                               + kf * BLOCK_N
                                               + qh_int_idx];
                return (uint32_t)((packed_qh >> qh_bit_off) & 0x3);
            };
            const uint32_t nib_lo = extract_nib_qh(k_lo);
            const uint32_t nib_hi = extract_nib_qh(k_hi);
            sh_B_qh_unpacked[stage * BQH_UNPACK_INTS_PER_KT
                             + k_pair * BLOCK_N
                             + n_inner] = nib_lo | (nib_hi << 2);
        }
    };

    // === Phase D.3 producer/consumer warpspec attempt 2026-06-06 — REJECTED ===
    // Iter-0 prologue / current-kt-producer + previous-kt-consumer design.
    // Wall regression 28x (Q6_K) + correctness drift across all configs at large
    // K_TILES (production K=5120 → 160 K-tiles). Bug location unclear; the small
    // K_TILES (8-16) Phase C cases sometimes passed in earlier variants but not
    // production scale. Reverted to D.2-FIX cooperative pass for ship.
    // Memo: marlin_v2_phase_d3_warpspec_reject_2026-06-06.md (to write)
    auto cooperative_dequant_kt = [&](int kt_target) {
        const int stage = kt_target % STAGES;
        uint32_t* bpair_dst = reinterpret_cast<uint32_t*>(sh_B_dequant);
        // NOTE: NO #pragma unroll on this 4-iteration loop (same lesson as v1).
        for (int i = 0; i < PAIRS_PER_THREAD_COOP; ++i) {
            const int lin = tid * PAIRS_PER_THREAD_COOP + i;
            const int k_pair = lin / BLOCK_N;
            const int n_inner = lin % BLOCK_N;
            const int k_lo = 2 * k_pair;
            const int k_hi = k_lo + 1;
            // Phase D.2-FIX: Q6_K reads pre-unpacked qh nibbles from sh_B_qh_unpacked.
            // One uint32 per (k_pair, n_inner) holds both lo (bits 0-1) and hi (bits 2-3)
            // nibbles. 32 threads × 4 bytes = 32 banks accessed = ZERO conflict.
            const uint32_t qh_packed_pair = std::is_same<QuantTag, Q6K>::value
                ? sh_B_qh_unpacked[stage * BQH_UNPACK_INTS_PER_KT
                                    + k_pair * BLOCK_N + n_inner]
                : 0u;
            auto deq_one = [&](int k_inner) -> DataT {
                const int kf = k_inner / 16;
                const int k_inner_in_kf = k_inner & 15;
                const int n_tile_id = n_inner / 16;
                const int n_in_tile = n_inner & 15;
                const int orig_flat = n_tile_id * 256 + k_inner_in_kf * 16 + n_in_tile;
                const int permuted = (int)c_PERM_INV[orig_flat];
                const int int_idx = permuted / 8;
                const int bit_off = (permuted & 7) * 4;
                const int packed_ql = sh_B_ql[stage * B_INTS_PER_KT_LOCAL
                                              + kf * (BLOCK_N * 2)
                                              + int_idx];
                const int nib_ql = (packed_ql >> bit_off) & 0x0F;
                const int sub_id_rel = k_inner / GROUP_SIZE;
                const DataT eff_sc = sh_scales[stage * SCALES_ELEMS_PER_KT_LOCAL
                                                + sub_id_rel * BLOCK_N + n_inner];
                if (std::is_same<QuantTag, Q4K>::value) {
                    const DataT eff_mn = sh_mins[stage * SCALES_ELEMS_PER_KT_LOCAL
                                                  + sub_id_rel * BLOCK_N + n_inner];
                    return __hsub(__hmul(int_to_data<DataT>(nib_ql), eff_sc), eff_mn);
                } else {
                    // Pick nib_qh from the pre-unpacked pair using k_inner & 1:
                    // bits 0-1 hold lo (k_inner_in_pair=0), bits 2-3 hold hi.
                    const int is_hi = k_inner & 1;
                    const int nib_qh = (qh_packed_pair >> (is_hi * 2)) & 0x3;
                    const int q6 = nib_ql | (nib_qh << 4);
                    return __hmul(int_to_data<DataT>(q6 - 32), eff_sc);
                }
            };
            const DataT lo = deq_one(k_lo);
            const DataT hi = deq_one(k_hi);
            bpair_dst[k_pair * B_PAIR_STRIDE_U32 + n_inner] = pack_pair<DataT>(lo, hi);
        }
    };

    // === Phase B.3.5: consumer ldmatrix + mma.sync ===
    //
    // Per consumer warp per kt:
    //   - 2 ldmatrix.x4 calls (one per mfrag) load A fragments from sh_A.
    //   - 8 nfrags × 2 = 16 B fragment loads from sh_B_dequant (scalar uint32).
    //   - 2 × 8 = 16 mma.sync.m16n8k16 calls per kf; 2 kf per kt → 32 mma per kt per warp.
    //
    // K_FRAGS = BLOCK_K / MMA_K = 32/16 = 2.
    constexpr int K_FRAGS = BLOCK_K / MMA_K;
    static_assert(K_FRAGS * MMA_K == BLOCK_K, "K_FRAGS must tile BLOCK_K exactly");

    auto consumer_mma_kt = [&](int kt_current) {
        if (!is_consumer) return;
        const int cur_stage = kt_current % STAGES;
        uint32_t* bpair_src = reinterpret_cast<uint32_t*>(sh_B_dequant);
        for (int kf = 0; kf < K_FRAGS; ++kf) {
            const int warp_m_base = warp_m * WARP_TILE_M;
            uint32_t A_frags[MMA_M_PER_WARP][4];
            const int ld_mat_id = lane >> 3;
            const int ld_row_in_mat = lane & 7;
            const int ld_row_offset = (ld_mat_id & 1) ? 8 : 0;
            const int ld_col_offset = (ld_mat_id & 2) ? 8 : 0;
            #pragma unroll
            for (int mf = 0; mf < MMA_M_PER_WARP; ++mf) {
                const int row = warp_m_base + mf * 16 + ld_row_offset + ld_row_in_mat;
                const int col = kf * 16 + ld_col_offset;
                DataT* row_ptr = &sh_A[cur_stage * BLOCK_M * A_STRIDE + row * A_STRIDE + col];
                uint32_t smem_a = cvt_to_smem_ptr(row_ptr);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(A_frags[mf][0]), "=r"(A_frags[mf][1]),
                      "=r"(A_frags[mf][2]), "=r"(A_frags[mf][3])
                    : "r"(smem_a));
            }
            const int group_id = lane / 4;
            const int rep_id = lane & 3;
            const int kf_pair_base = kf * 8;
            const int b0_pair = kf_pair_base + rep_id;
            const int b1_pair = kf_pair_base + rep_id + 4;
            // NOTE: NO #pragma unroll on the nf loop. 8 nfrags fully unrolled
            // would inflate per-thread register pressure past 128. The compiler's
            // own loop iteration is fine — same lesson as Phase B.3.4 producer.
            for (int nf = 0; nf < MMA_N_PER_WARP; ++nf) {
                const int warp_n_base = warp_n * WARP_TILE_N + nf * 8;
                const int n_in_BLOCK = warp_n_base + group_id;
                const uint32_t b0 = bpair_src[b0_pair * B_PAIR_STRIDE_U32 + n_in_BLOCK];
                const uint32_t b1 = bpair_src[b1_pair * B_PAIR_STRIDE_U32 + n_in_BLOCK];
                #pragma unroll
                for (int mf = 0; mf < MMA_M_PER_WARP; ++mf) {
                    mma_m16n8k16_f32<DataT>(
                        acc[mf][nf][0], acc[mf][nf][1], acc[mf][nf][2], acc[mf][nf][3],
                        A_frags[mf][0], A_frags[mf][1], A_frags[mf][2], A_frags[mf][3],
                        b0, b1);
                }
            }
        }
    };

    // === Phase B.3.5: K-loop with consumer MMA ===
    //
    // For Phase B.3.1, we exercise the A loader through the prologue + K-loop
    // structure but don't yet load B or run MMA. The kernel writes zeros to C
    // (same fragment indexing as the B.2 placeholder) to confirm:
    //   - prologue cp.async commits without crash
    //   - K-loop sync pattern doesn't deadlock
    //   - SMEM accesses don't trigger out-of-bounds
    // Phase B.3.2 will add the B loader; B.3.4+ adds dequant + MMA.
    const int K_TILES = K / BLOCK_K;

    // Phase D.1: pipelined K-loop matching BASIWAN v1 baseline timing.
    //
    // Prologue: STAGES-1 commits (each issues all data for one kt). After
    // prologue, STAGES-1 cp.async groups are in flight. The first iter of
    // the main loop adds the STAGES'th commit then waits — this is the v1
    // baseline pattern and the timing that lets the pipeline keep STAGES
    // groups overlapped without races.
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < K_TILES) {
            issue_A_kt(s, s);
            issue_B_ql_kt(s, s);
            issue_B_qh_kt(s, s);
            issue_scales_kt(s, s);
            issue_mins_kt(s, s);
        }
        cp_async_commit();
    }

    // K-loop matches BASIWAN v1 baseline timing: prologue issues STAGES-1
    // commits, then each iter issues kt+STAGES-1 prefetch + commit + wait.
    // No producer/consumer parity overlap (Phase D); just cooperative dequant
    // then MMA, like v1 baseline.
    //
    // NB: rebuild prologue to match — only STAGES-1 commits (the issue_*
    // in iter 0..STAGES-2 of the loop above gave us 4 commits, but the v1
    // pattern wants STAGES-1=3 prologue commits + 1 from-loop commit per iter).
    // For now, accept that we have STAGES commits in prologue (same as before)
    // and adjust the loop accordingly.

    for (int kt = 0; kt < K_TILES; ++kt) {
        const int prefetch_kt = kt + STAGES - 1;
        if (prefetch_kt < K_TILES) {
            const int prefetch_stage = prefetch_kt % STAGES;
            issue_A_kt(prefetch_kt, prefetch_stage);
            issue_B_ql_kt(prefetch_kt, prefetch_stage);
            issue_B_qh_kt(prefetch_kt, prefetch_stage);
            issue_scales_kt(prefetch_kt, prefetch_stage);
            issue_mins_kt(prefetch_kt, prefetch_stage);
        }
        cp_async_commit();

        cp_async_wait_group<STAGES - 1>();
        __syncthreads();

        cooperative_unpack_qh_kt(kt);
        // No __syncthreads between unpack and dequant: each thread writes to
        // sh_B_qh_unpacked[stage*N + k_pair*BLOCK_N + n_inner] where (k_pair, n_inner)
        // is derived from `lin = tid*PAIRS_PER_THREAD_COOP + i`. The dequant pass
        // reads the SAME entries with the SAME formula. Within a thread, write→read
        // is serial. No cross-thread dependency.
        cooperative_dequant_kt(kt);
        __syncthreads();

        consumer_mma_kt(kt);
        __syncthreads();
    }

    // === Phase B.3.5 epilogue: consumer warps write C ===
    //
    // Each consumer lane owns 4 fp32 acc values per (mf, nf) fragment, mapped to
    // C as: (group_id = lane/4, tig = lane%4) → C[row = group_id ± 0/8, col = tig*2 + 0/1].
    // Scalar 4-store-per-(mf,nf) layout matches v1; packed coalescing was tried
    // in v1 (FFFv) and rejected for register-pressure cascade.
    if (is_consumer) {
        for (int mf = 0; mf < MMA_M_PER_WARP; ++mf) {
            for (int nf = 0; nf < MMA_N_PER_WARP; ++nf) {
                const int warp_m_base_ep = warp_m * WARP_TILE_M + mf * 16;
                const int warp_n_base_ep = warp_n * WARP_TILE_N + nf * 8;
                const int group_ep = (lane / 4);
                const int tig_ep = (lane & 3);
                const int g_row0 = tile_m_base + warp_m_base_ep + group_ep;
                const int g_row1 = tile_m_base + warp_m_base_ep + group_ep + 8;
                const int g_col0 = tile_n_base + warp_n_base_ep + tig_ep * 2;
                const int g_col1 = g_col0 + 1;
                if (g_row0 < M && g_col0 < N) C[g_row0 * N + g_col0] = float_to_data<DataT>(acc[mf][nf][0]);
                if (g_row0 < M && g_col1 < N) C[g_row0 * N + g_col1] = float_to_data<DataT>(acc[mf][nf][1]);
                if (g_row1 < M && g_col0 < N) C[g_row1 * N + g_col0] = float_to_data<DataT>(acc[mf][nf][2]);
                if (g_row1 < M && g_col1 < N) C[g_row1 * N + g_col1] = float_to_data<DataT>(acc[mf][nf][3]);
            }
        }
    }
}

// ---- Phase B.2 host launcher ----
//
// Allocates dynamic SMEM, sets attr, dispatches the kernel.
template<typename QuantTag, typename DataT,
         int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES, int GROUP_SIZE>
cudaError_t launch_basiwan_v2(
    const DataT* A,
    const int* B_ql, const int* B_qh,
    const DataT* eff_scales, const DataT* eff_mins,
    DataT* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    constexpr int A_STRIDE_LAUNCH = BLOCK_K + 8;
    constexpr int A_BYTES = STAGES * BLOCK_M * A_STRIDE_LAUNCH * sizeof(DataT);
    constexpr int B_QL_BYTES = STAGES * (BLOCK_K / 16) * (BLOCK_N * 2) * (int)sizeof(int);
    constexpr int B_QH_BYTES = std::is_same<QuantTag, Q6K>::value
                                   ? STAGES * (BLOCK_K / 16) * BLOCK_N * (int)sizeof(int)
                                   : 0;
    constexpr int SCALES_PER_KT_LAUNCH = (BLOCK_K + GROUP_SIZE - 1) / GROUP_SIZE;
    constexpr int SCALES_BYTES = STAGES * SCALES_PER_KT_LAUNCH * BLOCK_N * (int)sizeof(DataT);
    constexpr int MINS_BYTES = std::is_same<QuantTag, Q4K>::value ? SCALES_BYTES : 0;
    // sh_B_dequant: 2 (parity) × K_PAIRS_PER_TILE × (BLOCK_N+1) uint32.
    constexpr int B_DEQUANT_BYTES = 2 * (BLOCK_K / 2) * (BLOCK_N + 1) * (int)sizeof(uint32_t);
    // Phase D.2-FIX: sh_B_qh_unpacked = STAGES × K_PAIRS × BLOCK_N uint32 (Q6_K only).
    constexpr int BQH_UNPACK_BYTES = std::is_same<QuantTag, Q6K>::value
                                         ? STAGES * (BLOCK_K / 2) * BLOCK_N * (int)sizeof(uint32_t)
                                         : 0;
    constexpr int SMEM_BYTES = A_BYTES + B_QL_BYTES + B_QH_BYTES
                                + SCALES_BYTES + MINS_BYTES
                                + B_DEQUANT_BYTES + BQH_UNPACK_BYTES;

    auto kernel_fn = marlin_kernel_v2<QuantTag, DataT, BLOCK_M, BLOCK_N, BLOCK_K, STAGES, GROUP_SIZE>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        96 * 1024);  // Ada SMEM cap (per-CTA dynamic; 100 KB total with reserved)
    if (err != cudaSuccess) return err;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);
    dim3 block(THREADS_PER_CTA, 1, 1);
    kernel_fn<<<grid, block, SMEM_BYTES, stream>>>(
        A, B_ql, B_qh, eff_scales, eff_mins, C, M, N, K);
    return cudaGetLastError();
}

// ---- Phase B.2 explicit instantiations ----
//
// Target: BLOCK_M=192 (the 6+2 design point) for both Q4_K and Q6_K, both bf16 and half.
// Q4_K group_size=32; Q6_K group_size=16.
template cudaError_t launch_basiwan_v2<Q4K, half, 192, 64, 32, 1, 32>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan_v2<Q6K, half, 192, 64, 32, 1, 16>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan_v2<Q4K, __nv_bfloat16, 192, 64, 32, 1, 32>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan_v2<Q6K, __nv_bfloat16, 192, 64, 32, 1, 16>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*,
    int, int, int, cudaStream_t);

// ---- Phase A.4 stub: confirms scaffold + dequant helpers compile ----
__global__ void stub_kernel() {
    // Use the dequant helpers in a trivial way to keep them referenced (and to
    // exercise the build path so a syntax error in dequant would surface here).
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t q_packed = 0x12345678;
        half2 scale = __halves2half2(__float2half(1.0f), __float2half(1.0f));
        half2 min = __halves2half2(__float2half(0.0f), __float2half(0.0f));
        half2 q4_out[4];
        dequant_q4k_8_to_half2(q_packed, scale, min, q4_out);

        uint32_t ql = 0xAB;
        uint32_t qh = 0x12;
        half2 q6_out[4];
        dequant_q6k_8_to_half2(ql, qh, scale, q6_out);

        // Anti-DCE: write the sum of all values to a (non-)existent global. The
        // compiler won't fully eliminate the call chain if it can't prove the
        // results are unused. asm volatile with a fake register prevents DCE.
        half2 sum = __hadd2(__hadd2(q4_out[0], q4_out[1]), __hadd2(q6_out[0], q6_out[1]));
        asm volatile("" :: "r"(*reinterpret_cast<uint32_t*>(&sum)));
    }
}

// === Phase C: Python-callable GEMM entry point ===
//
// Signature mirrors v1's basiwan_q4k_q6k_gemm — same input layout, same caller
// allocates the output buffer. Initial implementation only handles the
// BLOCK_M=192 aligned case (M must be a multiple of 192). M-tail (M % 192 != 0)
// support comes in Phase D.
torch::Tensor basiwan_v2_gemm(
    torch::Tensor x,            // (M, K) fp16 or bf16
    torch::Tensor B_ql,         // (K/16, N*2) int32 — Q4_K packed nibbles or Q6_K low 4 bits
    torch::Tensor B_qh,         // (K/16, N) int32 — Q6_K only; empty for Q4_K
    torch::Tensor eff_scales,   // (K/g, N) DataT
    torch::Tensor eff_mins,     // (K/g, N) DataT — Q4_K only; empty for Q6_K
    torch::Tensor out,          // (M, N) DataT — caller-allocated, will be overwritten
    int64_t quant_type,         // 0 = Q4_K, 1 = Q6_K
    int64_t group_size
) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA");
    TORCH_CHECK(B_ql.is_cuda(), "B_ql must be CUDA");
    TORCH_CHECK(eff_scales.is_cuda(), "eff_scales must be CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be CUDA");
    TORCH_CHECK(
        x.dtype() == torch::kFloat16 || x.dtype() == torch::kBFloat16,
        "x must be fp16 or bf16");
    TORCH_CHECK(out.dtype() == x.dtype(), "out dtype must match x dtype");
    TORCH_CHECK(B_ql.dtype() == torch::kInt32, "B_ql must be int32");
    TORCH_CHECK(eff_scales.dtype() == x.dtype(), "eff_scales dtype must match x dtype");

    const int M = x.size(0);
    const int K = x.size(1);
    const int N = out.size(1);
    TORCH_CHECK(out.size(0) == M, "out.size(0) must equal M");
    TORCH_CHECK(M % 192 == 0, "marlin_v2 Phase C: M must be multiple of BLOCK_M=192 "
                              "(M-tail handling is Phase D)");
    TORCH_CHECK(N % 64 == 0, "N must be multiple of BLOCK_N=64");
    TORCH_CHECK(K % 32 == 0, "K must be multiple of BLOCK_K=32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaSuccess;
    const bool is_bf16 = (x.dtype() == torch::kBFloat16);

    #define LAUNCH_V2(QT, DT, GS) do { \
        DT* A_ptr = reinterpret_cast<DT*>(x.data_ptr()); \
        DT* C_ptr = reinterpret_cast<DT*>(out.data_ptr()); \
        const DT* sc_ptr = reinterpret_cast<const DT*>(eff_scales.data_ptr()); \
        const DT* mn_ptr = (eff_mins.numel() > 0) \
            ? reinterpret_cast<const DT*>(eff_mins.data_ptr()) : nullptr; \
        const int* qh_ptr = (B_qh.numel() > 0) ? B_qh.data_ptr<int>() : nullptr; \
        err = launch_basiwan_v2<QT, DT, 192, 64, 32, 1, GS>( \
            A_ptr, B_ql.data_ptr<int>(), qh_ptr, sc_ptr, mn_ptr, C_ptr, \
            M, N, K, stream); \
    } while(0)

    if (quant_type == 0) {
        TORCH_CHECK(eff_mins.is_cuda() && eff_mins.numel() > 0, "Q4_K requires eff_mins");
        TORCH_CHECK(group_size == 32, "Q4_K requires group_size=32");
        if (is_bf16) {
            LAUNCH_V2(Q4K, __nv_bfloat16, 32);
        } else {
            LAUNCH_V2(Q4K, half, 32);
        }
    } else if (quant_type == 1) {
        TORCH_CHECK(B_qh.is_cuda() && B_qh.numel() > 0, "Q6_K requires B_qh");
        TORCH_CHECK(B_qh.dtype() == torch::kInt32, "B_qh must be int32");
        TORCH_CHECK(group_size == 16, "Q6_K requires group_size=16");
        if (is_bf16) {
            LAUNCH_V2(Q6K, __nv_bfloat16, 16);
        } else {
            LAUNCH_V2(Q6K, half, 16);
        }
    } else {
        TORCH_CHECK(false, "Unsupported quant_type ", quant_type, " (expected 0=Q4_K, 1=Q6_K)");
    }
    #undef LAUNCH_V2
    TORCH_CHECK(err == cudaSuccess, "marlin_v2 kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

std::string stub_status() {
    dim3 grid(1, 1, 1);
    dim3 block(THREADS_PER_CTA, 1, 1);
    stub_kernel<<<grid, block>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return std::string("marlin_v2 stub FAILED: ") + cudaGetErrorString(err);
    }
    return "marlin_v2 Phase B.1 — dequant helpers ported + linked (Q4_K + Q6_K)";
}

}  // namespace basiwan_v2

// ---- pybind11 surface ----

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "BASIWAN v2 from-scratch Q4_K + Q6_K kernel for Wan2.2 (Phase C — correctness gate)";
    m.def("stub_status", &basiwan_v2::stub_status,
          "Scaffold check — confirms kernel + dequant helpers compile and link");
    m.def("basiwan_v2_gemm", &basiwan_v2::basiwan_v2_gemm,
          "BASIWAN v2 Q4_K + Q6_K GEMM (BLOCK_M=192 aligned, Phase C correctness gate)");
}
