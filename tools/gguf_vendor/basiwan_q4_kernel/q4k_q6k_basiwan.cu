/*
 * Native CUDA kernel for GGUF Q4_K + Q6_K matmul on sm_89 (RTX 4090).
 *
 * Reads packed weights produced by `pack.py:convert_q4k/q6k_gguf_to_marlin`
 * (verified bit-equal roundtrip vs ggml dequant, max-abs ~9e-5).
 *
 * Layout (per pack.py):
 *   Q4_K:
 *     B_packed   : (K/16, N*2)  int32  — 8 nibbles per int32, fragment order
 *     eff_scales : (K/32, N)    fp16   — d × sub_scale_6bit, _scale_perm'd
 *     eff_mins   : (K/32, N)    fp16   — dmin × sub_min_6bit, _scale_perm'd
 *   Q6_K:
 *     B_ql_packed: (K/16, N*2)  int32  — low 4 bits, 8 per int32, fragment order
 *     B_qh_packed: (K/16, N)    int32  — high 2 bits, 16 per int32, fragment order
 *     eff_scales : (K/16, N)    fp16   — d × scale_int8_signed (POSITIVE or NEGATIVE)
 *
 * Math:
 *   Q4_K row j, col i in group g:
 *     W[j,i] = nibble × eff_scales[g, j] - eff_mins[g, j]
 *   Q6_K row j, col i in group g:
 *     W[j,i] = ((ql | (qh << 4)) - 32) × eff_scales[g, j]
 *
 * Kernel: m16n8k16 mma.sync.row.col.f16.f16.f16.f16, 4-stage cp.async pipeline,
 * 256 threads/CTA, BLOCK_M × BLOCK_N × BLOCK_K templated, fp16 accumulator.
 *
 * No fallback paths. No GPTQ Marlin retrofit. No vLLM borrow.
 *
 * Design notes:
 * - cp.async lets us prefetch next K-stage into SMEM while current stage computes.
 * - 4 stages × (A tile + B tile) SMEM should fit ~64-80 KB at BLOCK_M=128.
 * - cudaFuncSetAttribute opts into 96 KB SMEM per CTA on Ada.
 * - mma.sync m16n8k16 produces 4×fp16 per thread per call.
 * - Each warp owns one m16n8k16 mma output tile; 8 warps × 16×8 = BLOCK_M=128, BLOCK_N=64 layout.
 */

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "perm_inverse.cuh"  // brings in c_PERM_INV[1024]

namespace q4k_q6k_basiwan {

// ---- Constants ----
constexpr int TILE   = 16;   // m16n8k16 fragment edge
constexpr int WARP_SIZE = 32;

// ---- cp.async wrappers (sm_80+) ----
//
// 16-byte (int4) async copy from global → shared. cp.async hides memory latency
// behind compute; the kernel issues multiple loads before waiting.
//
// dst_smem_ptr: shared-memory address (we pass int4* casted appropriately)
// src_gmem_ptr: global-memory address
__device__ __forceinline__ void cp_async_16(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "r"(smem_addr), "l"(gmem_ptr));
}

// 4-byte cp.async — for the 4-bit packed B_ql / 2-bit packed B_qh int32 loads
// (we can't coalesce these into 16 bytes per thread without idling 75% of warps).
__device__ __forceinline__ void cp_async_4(uint32_t smem_addr, const void* gmem_ptr) {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        :: "r"(smem_addr), "l"(gmem_ptr));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n");
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n");
}

// ---- Named-barrier wrappers (sm_50+, used by warp-spec kernel) ----
//
// PTX `bar.sync N, count` and `bar.arrive N, count` give us per-warp-group
// synchronization without paying the full-CTA __syncthreads() cost. Ada has
// 16 barriers per CTA (IDs 0-15); ID 0 = __syncthreads, CUTLASS reserves 1-7
// (per `arch/barrier.h:284,305`), so we use 9, 10, 11 for our producer/consumer
// split. Per research memo docs/r_2_2_marlin_warpspec_ada.md §B.
//
// bar.sync = wait until `count` threads have arrived (blocking).
// bar.arrive = signal arrival without waiting (non-blocking).
//
// Counts must match across all participating warps; mismatch → undefined.
__device__ __forceinline__ void named_bar_sync(int id, int count) {
    asm volatile("bar.sync %0, %1;\n" :: "r"(id), "r"(count));
}

__device__ __forceinline__ void named_bar_arrive(int id, int count) {
    asm volatile("bar.arrive %0, %1;\n" :: "r"(id), "r"(count));
}

// Warp-spec barrier IDs (avoid 0=__syncthreads, 1-7=CUTLASS reserved, 8=epilogue).
// We use FOUR barriers, two per parity, so consecutive iterations don't interfere
// (with a single shared barrier, the producer's bar.arrive triggers release
// before the consumer waits, resetting the counter and deadlocking the next iter).
constexpr int BAR_BPAIR_FULL_0  = 9;   // producer signals: parity 0 ready
constexpr int BAR_BPAIR_FULL_1  = 11;  // producer signals: parity 1 ready
constexpr int BAR_BPAIR_EMPTY_0 = 10;  // consumer signals: parity 0 free
constexpr int BAR_BPAIR_EMPTY_1 = 12;  // consumer signals: parity 1 free
// Backward-compat aliases (formerly used by single-barrier Phase 2 experiments).
constexpr int BAR_BPAIR_FULL  = BAR_BPAIR_FULL_0;
constexpr int BAR_BPAIR_EMPTY = BAR_BPAIR_EMPTY_0;
constexpr int BAR_CPASYNC     = 13;  // reserved for future cp.async producer/consumer split

__device__ __forceinline__ uint32_t cvt_smem_ptr(const void* p) {
    uint32_t addr;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }\n"
        : "=r"(addr) : "l"(p));
    return addr;
}

// ---- m16n8k16 mma.sync wrapper (fp16 × fp16 → fp32 accumulator) ----
//
// One call performs a 16×8 output × 16-K-element MMA.
// A fragment: 8 fp16 (4 × half2) per warp, distributed across 32 lanes.
// B fragment: 4 fp16 (2 × half2) per warp.
// C fragment: 4 fp32 (one per output element) per warp lane.
//
// Per PTX ISA m16n8k16.row.col.f32.f16.f16.f32 lane mapping for the C/D operands:
//   d[0] = C[groupID + 0, threadID_in_group * 2 + 0]
//   d[1] = C[groupID + 0, threadID_in_group * 2 + 1]
//   d[2] = C[groupID + 8, threadID_in_group * 2 + 0]
//   d[3] = C[groupID + 8, threadID_in_group * 2 + 1]
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
// Dispatch by DataT (selected at template instantiation).
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

// ---- Dtype-generic conversion helpers ----
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

// pack_floats helper (inline-PTX cvt.f16x2.f32 / cvt.bf16x2.f32) was tried for
// epilogue coalescing on 2026-06-06 but caused register-pressure spills on
// Q4_K BLOCK_M=128 (80→192-byte stack frame) and a +12% wall regression at
// p720_17f. Removed because not used; the scalar epilogue (4 single-element
// stores per (mf,nf)) remains the lower-spill / faster path on this kernel.

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

// ---- Q4_K inline dequant ----
//
// Input: q_packed = 8 4-bit nibbles in a uint32 (lo bits hold nibble 0).
//        eff_scale: half2 holding (sub_scale, sub_scale) for this sub-block.
//        eff_min:   half2 holding (sub_min,   sub_min)   for this sub-block.
// Output: 4 half2 values = 8 fp16 nibbles dequantized as (n × scale - min).
//
// We use Marlin's LOP3 trick: place the 4-bit nibble in the fp16 exponent
// field of `0x6400` (= 1024.0). Then `fp16(0x6400 | nibble) - fp16(1024.0)` gives
// the value as fp16 nibble in [0..15]. Multiply by eff_scale, subtract eff_min.
//
// Constants:
//   EX  = 0x64006400 (two fp16(1024) packed)
//   SUB = 0x64006400 (we will SUBTRACT to remove the 1024 base)
//
// Note: this is asymmetric (no -8 fold). The standard Marlin SUB = 0x64086408
// hides a "-8" symmetric offset; we DON'T want that for Q4_K because our
// scales already encode the asymmetry.
__device__ __forceinline__ void dequant_q4k_8_to_half2(
    uint32_t q_packed,
    half2 eff_scale,
    half2 eff_min,
    half2 out[4]
) {
    constexpr int LO = 0x000f000f;   // mask for low nibbles in two halves
    constexpr int HI = 0x00f000f0;   // mask for high nibbles (×2 shift each half)
    constexpr int EX = 0x64006400;   // fp16 1024.0 exponent base, 2 packed

    // Extract 4 nibbles per "low" pass (nibble 0,1 from byte 0 lo bits).
    // The 8 nibbles fan out into 4 half2 pairs.
    // Marlin's pattern: LOP3 ((a & b) | c) with lut 0xca = ((a & 0xf) << 0) | (1024 << 16).
    uint32_t lo0 = lop3<0xca>(q_packed,        LO, EX);  // nibbles 0, 1
    uint32_t hi0 = lop3<0xca>(q_packed,        HI, EX);  // nibbles 2, 3 (already in upper 4 bits)
    uint32_t lo1 = lop3<0xca>(q_packed >> 8,   LO, EX);  // nibbles 4, 5
    uint32_t hi1 = lop3<0xca>(q_packed >> 8,   HI, EX);  // nibbles 6, 7

    // Subtract the 1024.0 base (no -8 fold). Then FMA scale, subtract min.
    constexpr int SUB = 0x64006400;  // fp16 1024.0 base; subtraction recovers nibble in [0..15]
    half2 hsub = *reinterpret_cast<const half2*>(&SUB);

    half2 v0 = __hsub2(*reinterpret_cast<half2*>(&lo0), hsub);   // (n0, n1) fp16
    half2 v1 = __hsub2(*reinterpret_cast<half2*>(&hi0), hsub);   // (n2, n3); upper nibbles need /16
    half2 v2 = __hsub2(*reinterpret_cast<half2*>(&lo1), hsub);   // (n4, n5)
    half2 v3 = __hsub2(*reinterpret_cast<half2*>(&hi1), hsub);   // (n6, n7)

    // The HI mask extracted from positions 4-7 of each byte, so v1 and v3 are 16× too large.
    // Use a half2 multiplier 0x2c00 = fp16(1/16) to compensate.
    constexpr int INV16 = 0x2c002c00;  // fp16(1/16) packed
    half2 hinv16 = *reinterpret_cast<const half2*>(&INV16);
    v1 = __hmul2(v1, hinv16);
    v3 = __hmul2(v3, hinv16);

    // Apply asymmetric dequant: out = nibble × scale - min.
    half2 neg_min = __hneg2(eff_min);
    out[0] = __hfma2(v0, eff_scale, neg_min);
    out[1] = __hfma2(v1, eff_scale, neg_min);
    out[2] = __hfma2(v2, eff_scale, neg_min);
    out[3] = __hfma2(v3, eff_scale, neg_min);
}

// ---- Q6_K inline dequant ----
//
// Input: ql_packed = 8 low-4-bit values in a uint32 (lo bits hold value 0).
//        qh_packed = 8 high-2-bit values in a uint32 (lo bits hold value 0; 2 bits per value).
//        eff_scale: half2 holding (sub_scale, sub_scale).
// Output: 4 half2 = 8 fp16 dequantized as ((ql | qh<<4) - 32) × scale.
//
// Q6_K is symmetric — no min term. The -32 shift is folded into a SUB constant.
__device__ __forceinline__ void dequant_q6k_8_to_half2(
    uint32_t ql_packed,
    uint32_t qh_packed,
    half2 eff_scale,
    half2 out[4]
) {
    // ql: low 4 bits per value; we extract using LOP3 like Q4_K.
    // qh: high 2 bits per value; we extract masks 0x00030003 and shift to position 4.
    constexpr int LO = 0x000f000f;
    constexpr int HI = 0x00f000f0;
    constexpr int EX = 0x64006400;          // fp16 1024.0 base
    constexpr int MASK_QH = 0x00030003;     // 2-bit per-value mask, 2 packed
    constexpr int MASK_QH_HI = 0x000c000c;  // 2-bit per upper-pair, post-shift accommodation

    // Pull two pairs of low-4-bits.
    uint32_t lo0 = lop3<0xca>(ql_packed,      LO, EX);
    uint32_t hi0 = lop3<0xca>(ql_packed,      HI, EX);
    uint32_t lo1 = lop3<0xca>(ql_packed >> 8, LO, EX);
    uint32_t hi1 = lop3<0xca>(ql_packed >> 8, HI, EX);

    // Pull qh 2-bit values per packed pair (8 values total, packed 2-per-half within each int).
    // Each qh has 8 values × 2 bits = 16 bits. Lay out matching ql nibbles.
    // qh value n occupies bits [2n, 2n+1] of qh_packed.
    // For LOP3 with EX, we need to shift qh bits into the same position as ql bits (low nibble).
    // Since qh values are 0..3, shifting them << 4 places them in the high-nibble position of
    // a fp16 integer representation 0x6400|val. We assemble (ql | (qh << 4)) inside the LOP3.
    //
    // For simplicity here, decode qh into a half2 directly without LOP3 (the qh bits are sparse).
    // This avoids complex packed-bit gymnastics at the cost of a few extra muls/adds.

    constexpr int SUB_BASE  = 0x64006400;  // fp16 1024.0 — undoes the EX trick
    constexpr int SUB_SHIFT = 0x65006500;  // fp16 1280.0 — for combined value (1024 + 256 = 1280;
                                            //   256 = qh contribution for top-nibble base ×16 = +5
                                            //   so 6-bit value needs offset 32 baked separately).
    // Actually: we'll compose value = ql + (qh << 4) inside fp16 and subtract 32.
    // To avoid complex bit gymnastics, decode qh into a half2 multiplier separately.

    half2 hsub_base = *reinterpret_cast<const half2*>(&SUB_BASE);
    half2 v_ql0 = __hsub2(*reinterpret_cast<half2*>(&lo0), hsub_base); // ql lo nibble of pair 0
    half2 v_qh0 = __hsub2(*reinterpret_cast<half2*>(&hi0), hsub_base); // ql hi nibble (×16)
    half2 v_ql1 = __hsub2(*reinterpret_cast<half2*>(&lo1), hsub_base);
    half2 v_qh1 = __hsub2(*reinterpret_cast<half2*>(&hi1), hsub_base);

    // Now v_*_q* hold ql nibble values 0..15 in fp16.
    // For Q6_K we want: (ql | (qh<<4)) - 32. Since ql is 4 bits (0..15) and qh<<4 is 0..48,
    // combined is 0..63, and subtracting 32 gives signed [-32..31].
    //
    // qh contribution: extract per-value 2 bits, multiply by 16 (the <<4), and add to ql.
    // Build half2 with (qh_val_a × 16, qh_val_b × 16) for each pair.
    //
    // qh_packed has 8 2-bit values. For pair 0 (values 0,1): bits 0-1 and 2-3.
    uint32_t qh_bits01 = qh_packed & 0xFu;            // values 0,1 in low 4 bits (2 bits each)
    uint32_t qh_bits23 = (qh_packed >> 4) & 0xFu;
    uint32_t qh_bits45 = (qh_packed >> 8) & 0xFu;
    uint32_t qh_bits67 = (qh_packed >> 12) & 0xFu;
    auto build_qh16 = [](uint32_t pair) -> half2 {
        // pair bits [1:0] = value a, [3:2] = value b. Each is 0..3.
        int va = pair & 0x3;
        int vb = (pair >> 2) & 0x3;
        // Multiply by 16 (the <<4 shift) in fp16 directly.
        half ha = __float2half(static_cast<float>(va * 16));
        half hb = __float2half(static_cast<float>(vb * 16));
        return __halves2half2(ha, hb);
    };
    half2 qh16_01 = build_qh16(qh_bits01);
    half2 qh16_23 = build_qh16(qh_bits23);
    half2 qh16_45 = build_qh16(qh_bits45);
    half2 qh16_67 = build_qh16(qh_bits67);

    // Combine: combined_val = ql + qh16 - 32, all in fp16.
    constexpr int SUB_32 = 0xd000d000;  // fp16(-32) packed (negative 32)
    half2 hsub32 = *reinterpret_cast<const half2*>(&SUB_32);

    half2 v0 = __hadd2(__hadd2(v_ql0, qh16_01), hsub32);
    half2 v1 = __hadd2(__hadd2(v_qh0, qh16_23), hsub32);
    half2 v2 = __hadd2(__hadd2(v_ql1, qh16_45), hsub32);
    half2 v3 = __hadd2(__hadd2(v_qh1, qh16_67), hsub32);

    // Apply scale. Q6_K is symmetric, no min.
    out[0] = __hmul2(v0, eff_scale);
    out[1] = __hmul2(v1, eff_scale);
    out[2] = __hmul2(v2, eff_scale);
    out[3] = __hmul2(v3, eff_scale);
}

// ---- Quant-type tag ----
struct Q4K {};
struct Q6K {};

// ---- m16n8k16 load helpers from SMEM ----
//
// ldmatrix.sync loads 4 fragments of m16k16 A-tile (or n8k16 B-tile) from
// SMEM into per-lane registers. We use this instead of computing per-lane
// offsets manually — it's the canonical fragment-layout load.

// A fragment: 16×16 fp16 tile. ldmatrix.x4 returns 4 uint32 = 4 × half2 = 8 fp16 per lane.
__device__ __forceinline__ void ldmatrix_x4_A(
    uint32_t smem_addr,
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0, %1, %2, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        :  "r"(smem_addr));
}

// B fragment: 16×8 fp16 tile (k×n laid out for col-major). ldmatrix.x2 returns 2 uint32.
__device__ __forceinline__ void ldmatrix_x2_trans_B(
    uint32_t smem_addr,
    uint32_t& b0, uint32_t& b1
) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
        "{%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1)
        :  "r"(smem_addr));
}

// ---- The kernel ----
//
// Layout choice for first pass: BLOCK_M=64, BLOCK_N=64, BLOCK_K=32, 256 threads, 4 stages.
//
//   - 8 warps in (4 M, 2 N) layout. Each warp owns a 16×32 output sub-tile.
//   - Per warp per K-tile (BLOCK_K=32): 2 K-fragments × 1 m-frag × 2 n-frags = 4 mma calls.
//   - SMEM per stage: A=64×32×2=4 KB, B (Q4_K packed)=64×32/2=1 KB, scales~256 B, mins~256 B.
//   - 4 stages × ~5.5 KB = 22 KB → fits 96 KB cap with room.
//
// Per K iter we consume QK_K=256 K-elements (8 sub-blocks × 32 each). With BLOCK_K=32,
// we walk 8 BLOCK_K tiles per super-block. eff_scales / eff_mins index by sub-block.
//
// Output accumulator: fp32 per output element. Cast to fp16/bf16 at epilogue.

// WARPSPEC: when true, K-loop uses producer/consumer warp split per
// docs/r_2_2_marlin_warpspec_ada.md. Default false preserves the bit-exact
// symmetric-8-warp behavior of every existing instantiation.
//   Phase 1 scaffold (2026-06-05): WARPSPEC=true currently behaves identically
//   to WARPSPEC=false. Subsequent phases will introduce real producer/consumer
//   logic in `if constexpr (WARPSPEC)` branches gated by named bar.sync IDs
//   9 (full) and 10 (empty).
template<typename QuantTag, typename DataT, int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES, int GROUP_SIZE, bool WARPSPEC = false>
__global__ void marlin_kernel(
    const DataT* __restrict__ A,            // (M, K) DataT (half or __nv_bfloat16)
    const int*   __restrict__ B_ql,         // (K/16, N*2) int32 — Q4_K packed nibbles or Q6_K ql
    const int*   __restrict__ B_qh,         // (K/16, N) int32 — Q6_K only, else nullptr
    const DataT* __restrict__ eff_scales,   // (K/GROUP_SIZE, N) DataT
    const DataT* __restrict__ eff_mins,     // (K/GROUP_SIZE, N) DataT — Q4_K only, else nullptr
    DataT*       __restrict__ C,            // (M, N) DataT output
    int M, int N, int K
) {
    // GROUP_SIZE is now a template parameter (32 for Q4_K, 16 for Q6_K). Avoids runtime
    // division and lets the compiler unroll the per-kt scale-load.
    constexpr int SCALES_PER_KT = (BLOCK_K + GROUP_SIZE - 1) / GROUP_SIZE;
    static_assert(SCALES_PER_KT * GROUP_SIZE == BLOCK_K,
        "BLOCK_K must be an exact multiple of GROUP_SIZE for clean SMEM staging");
    // 1) Identify this CTA's output tile.
    const int pid_m = blockIdx.y;
    const int pid_n = blockIdx.x;
    const int tile_m_base = pid_m * BLOCK_M;
    const int tile_n_base = pid_n * BLOCK_N;
    if (tile_m_base >= M || tile_n_base >= N) return;

    // 2) Warp layout.
    //   Baseline: 8 warps (warps_m=4, warps_n=2). All 8 warps own MMA tiles.
    //   WARPSPEC: 4 producer (0..3) + 4 consumer (4..7). Consumers tile
    //             (warps_m=2, warps_n=2). Each consumer warp covers 2× baseline's
    //             WARP_TILE_M (same total MMA work as 8 baseline warps; producer
    //             warps overlap their dequant with consumer's MMA via prefetch).
    constexpr int WARPS_M = WARPSPEC ? 2 : 4;
    constexpr int WARPS_N = 2;
    constexpr int CONS_WARP_BASE_C = WARPSPEC ? 4 : 0;
    static_assert((WARPSPEC ? 8 : (WARPS_M * WARPS_N)) * 32 == 256, "warp count must yield 256 threads");
    static_assert(BLOCK_M % (WARPS_M * 16) == 0, "BLOCK_M must tile evenly across WARPS_M × 16");
    static_assert(BLOCK_N % (WARPS_N * 16) == 0, "BLOCK_N must tile evenly across WARPS_N × 16");
    constexpr int WARP_TILE_M = BLOCK_M / WARPS_M;       // baseline: 64/4=16, 128/4=32; ws: 64/2=32, 128/2=64
    constexpr int WARP_TILE_N = BLOCK_N / WARPS_N;       // 64/2 = 32

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid & 31;
    // For WARPSPEC, the producer warps map to a sentinel tile (warp_m, warp_n) that
    // is only consulted by consumer code paths. We guard MMA + epilogue with
    // `warp_id >= CONS_WARP_BASE_C` checks.
    const int cons_local_warp = WARPSPEC ? (warp_id - CONS_WARP_BASE_C) : warp_id;  // 0..3 for consumers, negative for producers (unused)
    const int warp_m = cons_local_warp / WARPS_N;
    const int warp_n = cons_local_warp % WARPS_N;

    // 3) Shared memory layout: stages × (A | B_ql | B_qh? | scales | mins?).
    // A: BLOCK_M × BLOCK_K fp16 = BLOCK_M × BLOCK_K × 2 bytes
    // B_ql: BLOCK_N × BLOCK_K / 2 bytes (4-bit packed for Q4_K and Q6_K low nibbles)
    // B_qh: only for Q6_K, BLOCK_N × BLOCK_K / 4 bytes (2-bit packed)
    extern __shared__ unsigned char smem[];
    // sh_A row stride: BLOCK_K + A_PAD. Padding breaks the 4-way bank conflict in
    // the mma A fragment read (DataT is 2 bytes for both half and bf16, so the
    // bank arithmetic is identical). Without padding: stride 32 × 2 = 64 bytes
    // = 16 banks; rows group=0..7 hit 4 distinct banks. With +8 pad: stride 80
    // bytes = 20 banks; 8 distinct banks, no conflict.
    constexpr int A_PAD = 8;
    constexpr int A_STRIDE = BLOCK_K + A_PAD;  // in DataT elements
    constexpr int A_STAGE_BYTES = BLOCK_M * A_STRIDE * sizeof(DataT);
    constexpr int B_QL_STAGE_BYTES = BLOCK_N * BLOCK_K / 2;
    // B_QH reserved only for Q6_K; Q4_K skips this region entirely to keep
    // SMEM_BYTES (in host launcher) consistent with pointer-math offsets.
    constexpr int B_QH_STAGE_BYTES = std::is_same<QuantTag, Q6K>::value
        ? (BLOCK_N * BLOCK_K / 4) : 0;

    DataT* sh_A   = reinterpret_cast<DataT*>(smem);
    int*   sh_B_ql = reinterpret_cast<int*>(
        reinterpret_cast<unsigned char*>(sh_A) + STAGES * A_STAGE_BYTES);
    int*   sh_B_qh = reinterpret_cast<int*>(reinterpret_cast<unsigned char*>(sh_B_ql) + STAGES * B_QL_STAGE_BYTES);
    // Scales/mins per-kt staging: SCALES_PER_KT × BLOCK_N DataT entries.
    // Stored AFTER B_qh (which is zero-sized for Q4_K).
    constexpr int SCALES_STAGE_BYTES = SCALES_PER_KT * BLOCK_N * sizeof(DataT);
    constexpr int MINS_STAGE_BYTES = (std::is_same<QuantTag, Q4K>::value) ? SCALES_STAGE_BYTES : 0;
    // sh_B_pair is a PTX-aligned K-pair layout: each entry is a packed half2 /
    // bf162 of (B[2*k_pair, n], B[2*k_pair+1, n]) — one mma B[0] fragment slot
    // for lane (rep_id, group_id). Row stride padded by +1 uint32 to break the
    // 4-way bank conflict between rep_id values.
    constexpr int B_PAIR_STRIDE_U32 = BLOCK_N + 1;  // uint32 entries per K-pair row
    constexpr int B_FP16_BYTES_SINGLE = (BLOCK_K / 2) * B_PAIR_STRIDE_U32 * sizeof(uint32_t);
    // When WARPSPEC, double-buffer sh_B_pair (+8 KiB) so producer writes one
    // parity while consumer reads the other. Phase 1 uses only parity 0 (no
    // overlap yet) — output stays bit-identical to baseline. Phase 3 introduces
    // actual producer/consumer parity exchange.
    constexpr int B_FP16_BYTES = WARPSPEC ? (2 * B_FP16_BYTES_SINGLE) : B_FP16_BYTES_SINGLE;
    DataT* sh_scales = reinterpret_cast<DataT*>(
        reinterpret_cast<unsigned char*>(sh_B_qh) + STAGES * B_QH_STAGE_BYTES);
    DataT* sh_mins = reinterpret_cast<DataT*>(
        reinterpret_cast<unsigned char*>(sh_scales) + SCALES_STAGE_BYTES);  // unused for Q6_K
    // Dequanted B tile, K-pair packed. Single buffer (not staged) — computed
    // at the start of each kt and consumed entirely before the next.
    uint32_t* sh_B_pair = reinterpret_cast<uint32_t*>(
        reinterpret_cast<unsigned char*>(sh_mins) + MINS_STAGE_BYTES);

    // 4) Accumulator fragments. Per warp: WARP_TILE_M / 16 × WARP_TILE_N / 8 mma output tiles.
    constexpr int MMA_M_PER_WARP = WARP_TILE_M / 16;  // 16/16 = 1
    constexpr int MMA_N_PER_WARP = WARP_TILE_N / 8;   // 32/8 = 4
    // FP32 accumulator: 4 floats per (mma_m, mma_n) per lane.
    float acc[MMA_M_PER_WARP][MMA_N_PER_WARP][4] = {};

    // [Phase 3 step C — WARPSPEC K-loop with producer/consumer + prefetch-by-one overlap]
    // Producer warps (0..3) dequant kt+1 while consumer warps (4..7) MMA kt.
    // Coordination via PTX named barriers IDs 9 (full) and 10 (empty).
    // Acc array is sized for consumer's 2× WARP_TILE_M (only consumer warps own
    // it; producer warps' acc is unused).
    if constexpr (WARPSPEC) {
        const int K_TILES_ws = K / BLOCK_K;
        constexpr int B_ROWS_PER_TILE_C_ws = BLOCK_K / 16;
        constexpr int B_INTS_PER_TILE_C_ws = B_ROWS_PER_TILE_C_ws * BLOCK_N * 2;
        constexpr int B_QH_INTS_PER_TILE_C_ws = B_ROWS_PER_TILE_C_ws * BLOCK_N;
        constexpr int BPAIR_SINGLE_U32 = (BLOCK_K / 2) * B_PAIR_STRIDE_U32;

        // cp.async issuer (full-CTA, same as baseline).
        auto issue_kt_async_ws = [&](int kt_target, int dst_stage) {
            {
                constexpr int A_ELEMS_PER_THREAD = (BLOCK_M * BLOCK_K) / 256;
                constexpr int A_CALLS_PER_THREAD = A_ELEMS_PER_THREAD / 8;
                #pragma unroll
                for (int c = 0; c < A_CALLS_PER_THREAD; ++c) {
                    const int linear_chunk = tid * A_CALLS_PER_THREAD + c;
                    const int row = (linear_chunk * 8) / BLOCK_K;
                    const int col = (linear_chunk * 8) % BLOCK_K;
                    const int gm = tile_m_base + row;
                    const int gk = kt_target * BLOCK_K + col;
                    DataT* src = const_cast<DataT*>(&A[gm * K + gk]);
                    DataT* dst = &sh_A[dst_stage * BLOCK_M * A_STRIDE + row * A_STRIDE + col];
                    cp_async_16(cvt_smem_ptr(dst), src);
                }
            }
            {
                const int linear_id = tid;
                const int row = linear_id / (BLOCK_N * 2);
                const int col = linear_id % (BLOCK_N * 2);
                const int gk_tile = (kt_target * BLOCK_K) / 16 + row;
                const int gn = tile_n_base * 2 + col;
                int* src = const_cast<int*>(&B_ql[gk_tile * (N * 2) + gn]);
                int* dst = &sh_B_ql[dst_stage * B_INTS_PER_TILE_C_ws + row * (BLOCK_N * 2) + col];
                cp_async_4(cvt_smem_ptr(dst), src);
            }
            if (std::is_same<QuantTag, Q6K>::value && B_qh != nullptr) {
                if (tid < B_QH_INTS_PER_TILE_C_ws) {
                    const int row = tid / BLOCK_N;
                    const int col = tid % BLOCK_N;
                    const int gk_tile = (kt_target * BLOCK_K) / 16 + row;
                    const int gn = tile_n_base + col;
                    int* src = const_cast<int*>(&B_qh[gk_tile * N + gn]);
                    int* dst = &sh_B_qh[dst_stage * B_QH_INTS_PER_TILE_C_ws + row * BLOCK_N + col];
                    cp_async_4(cvt_smem_ptr(dst), src);
                }
            }
        };

        // Full-CTA scales load for a given kt.
        auto load_scales_kt = [&](int kt_target) {
            constexpr int sc_loads = SCALES_PER_KT * BLOCK_N;
            const int sub_id_base = (kt_target * BLOCK_K) / GROUP_SIZE;
            for (int i = tid; i < sc_loads; i += 256) {
                int s_local = i / BLOCK_N;
                int n_local = i % BLOCK_N;
                int sc_idx = (sub_id_base + s_local) * N + tile_n_base + n_local;
                sh_scales[s_local * BLOCK_N + n_local] = eff_scales[sc_idx];
            }
            if (std::is_same<QuantTag, Q4K>::value && eff_mins != nullptr) {
                for (int i = tid; i < sc_loads; i += 256) {
                    int s_local = i / BLOCK_N;
                    int n_local = i % BLOCK_N;
                    int sc_idx = (sub_id_base + s_local) * N + tile_n_base + n_local;
                    sh_mins[s_local * BLOCK_N + n_local] = eff_mins[sc_idx];
                }
            }
        };

        // Producer-only dequant (128 threads, 2× pairs/thread) → sh_B_pair[parity_offset..].
        auto producer_dequant_kt = [&](int kt_target, int parity) {
            const int stage = kt_target % STAGES;
            uint32_t* bpair_dst = sh_B_pair + parity * BPAIR_SINGLE_U32;
            constexpr int K_PAIRS_PER_TILE = BLOCK_K / 2;
            constexpr int TOTAL_PAIRS = K_PAIRS_PER_TILE * BLOCK_N;
            constexpr int PROD_THREADS = 128;
            constexpr int PAIRS_PER_PROD = TOTAL_PAIRS / PROD_THREADS;
            const int prod_tid = tid;  // producer warps are 0..3, tid 0..127
            #pragma unroll
            for (int i = 0; i < PAIRS_PER_PROD; ++i) {
                const int lin = prod_tid * PAIRS_PER_PROD + i;
                const int k_pair = lin / BLOCK_N;
                const int n_inner = lin % BLOCK_N;
                const int k_lo = 2 * k_pair;
                const int k_hi = k_lo + 1;
                auto deq_one = [&](int k_inner) -> DataT {
                    const int kf = k_inner / 16;
                    const int k_inner_in_kf = k_inner & 15;
                    const int n_tile_id = n_inner / 16;
                    const int n_in_tile = n_inner & 15;
                    const int orig_flat = n_tile_id * 256 + k_inner_in_kf * 16 + n_in_tile;
                    const int permuted = (int)c_PERM_INV[orig_flat];
                    const int int_idx = permuted / 8;
                    const int bit_off = (permuted & 7) * 4;
                    const int packed_ql = sh_B_ql[stage * B_INTS_PER_TILE_C_ws + kf * (BLOCK_N * 2) + int_idx];
                    const int nib_ql = (packed_ql >> bit_off) & 0x0F;
                    const int sub_id_rel = k_inner / GROUP_SIZE;
                    const DataT eff_sc = sh_scales[sub_id_rel * BLOCK_N + n_inner];
                    if (std::is_same<QuantTag, Q4K>::value) {
                        const DataT eff_mn = sh_mins[sub_id_rel * BLOCK_N + n_inner];
                        return __hsub(__hmul(int_to_data<DataT>(nib_ql), eff_sc), eff_mn);
                    } else {
                        const int qh_int_idx = permuted / 16;
                        const int qh_bit_off = (permuted & 15) * 2;
                        const int packed_qh = sh_B_qh[stage * B_QH_INTS_PER_TILE_C_ws + kf * BLOCK_N + qh_int_idx];
                        const int nib_qh = (packed_qh >> qh_bit_off) & 0x3;
                        const int q6 = nib_ql | (nib_qh << 4);
                        return __hmul(int_to_data<DataT>(q6 - 32), eff_sc);
                    }
                };
                const DataT lo = deq_one(k_lo);
                const DataT hi = deq_one(k_hi);
                bpair_dst[k_pair * B_PAIR_STRIDE_U32 + n_inner] = pack_pair<DataT>(lo, hi);
            }
        };

        // Consumer-only MMA for kt reading from sh_B_pair[parity * SINGLE..].
        auto consumer_mma_kt = [&](int kt_current, int parity) {
            const int cur_stage = kt_current % STAGES;
            uint32_t* bpair_src = sh_B_pair + parity * BPAIR_SINGLE_U32;
            constexpr int K_FRAGS = BLOCK_K / 16;
            for (int kf = 0; kf < K_FRAGS; ++kf) {
                const int warp_m_base = warp_m * WARP_TILE_M;
                const int a_group = lane / 4;
                const int a_rep   = lane & 3;
                (void)a_group; (void)a_rep;
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
                    uint32_t smem_a = cvt_smem_ptr(row_ptr);
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
                #pragma unroll
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

        // === WARPSPEC K-loop ===
        // 1. cp.async prologue: issue stages 0..STAGES-2 (matches baseline pattern).
        //    Iter 0 PHASE A then prefetches kt+STAGES-1 = STAGES-1; after that commit,
        //    wait_group<STAGES-2> gives us 2 done = kt=0 AND kt=1.
        #pragma unroll
        for (int s = 0; s < STAGES - 1; ++s) {
            if (s < K_TILES_ws) issue_kt_async_ws(s, s);
            cp_async_commit();
        }
        // 2. Wait for kt=0 data (1 done after STAGES-1 commits + wait_group<STAGES-2>)
        cp_async_wait_group<STAGES - 2>();
        __syncthreads();
        // 3. Load scales for kt=0 (full-CTA)
        load_scales_kt(0);
        __syncthreads();
        // 4. Producer dequants kt=0 → parity 0. No bar.arrive — instead, the
        // __syncthreads below acts as the prologue→iter0 sync (data ready for
        // consumer iter 0 without a named barrier — iter 0 consumer skips its
        // bar.sync(FULL) for this reason).
        if (warp_id < CONS_WARP_BASE_C) {
            producer_dequant_kt(0, 0);
        }
        __syncthreads();
        // 5. Main K-loop with prefetch-by-one and full overlap.
        //   bar pattern: producer iter K writes parity (K+1)&1, signals BAR_FULL.
        //   consumer iter K reads parity K&1 (written by producer iter K-1 or prologue),
        //   then bar.arrive(BAR_EMPTY) so producer iter K+1 knows parity ((K+1)&1)==((K-1)&1)
        //   can be reused.
        //   - Consumer iter 0 skips bar.sync(FULL) because prologue+sync already prepared parity 0.
        //   - Producer iter 0 skips bar.sync(EMPTY) because no consumer has used parity 1 yet.
        //   - Producer iter k (k>=1) waits BAR_EMPTY before overwriting parity (k+1)&1 (== (k-1)&1).
        for (int kt = 0; kt < K_TILES_ws; ++kt) {
            // PHASE A: full-CTA prefetch for kt+STAGES-1 (one ahead of consumer).
            // For STAGES=3, iter 0 prefetches kt=2 → stage 2. Iter 1 prefetches kt=3 → stage 0
            // (overwrites kt=0 which is no longer needed).
            const int prefetch_kt = kt + STAGES - 1;
            if (prefetch_kt < K_TILES_ws) {
                issue_kt_async_ws(prefetch_kt, prefetch_kt % STAGES);
            }
            cp_async_commit();
            // PHASE B: wait. wait_group<STAGES-2> = at most STAGES-2 in flight → at least
            // 2 newest groups done. At iter k, that gives us kt=k and kt=k+1 ready.
            if (kt + 1 < K_TILES_ws) {
                cp_async_wait_group<STAGES - 2>();
            } else {
                cp_async_wait_group<STAGES - 1>();
            }
            __syncthreads();
            // PHASE C: load scales for kt+1 (full-CTA)
            if (kt + 1 < K_TILES_ws) {
                load_scales_kt(kt + 1);
            }
            __syncthreads();
            // PHASE D: role-split. Producer dequants kt+1 → parity (kt+1)&1 while
            // consumer MMAs kt → parity kt&1. Different SMEM parities, no conflict.
            // Loop-end __syncthreads provides cross-iter producer-write/consumer-read sync.
            if (warp_id < CONS_WARP_BASE_C) {
                if (kt + 1 < K_TILES_ws) {
                    producer_dequant_kt(kt + 1, (kt + 1) & 1);
                }
            } else {
                consumer_mma_kt(kt, kt & 1);
            }
            __syncthreads();  // sync producer write → next iter's consumer read
        }
        cp_async_wait_all();
        // === EPILOGUE: consumer warps write C === (scalar; see Epilogue coalescing REJECTED note in baseline epilogue)
        if (warp_id < CONS_WARP_BASE_C) return;
        #pragma unroll
        for (int mf_ep = 0; mf_ep < MMA_M_PER_WARP; ++mf_ep) {
            #pragma unroll
            for (int nf_ep = 0; nf_ep < MMA_N_PER_WARP; ++nf_ep) {
                const int warp_m_base_ep = warp_m * WARP_TILE_M + mf_ep * 16;
                const int warp_n_base_ep = warp_n * WARP_TILE_N + nf_ep * 8;
                const int group_ep = (lane / 4);
                const int tig_ep = (lane & 3);
                const int g_row0 = tile_m_base + warp_m_base_ep + group_ep;
                const int g_row1 = tile_m_base + warp_m_base_ep + group_ep + 8;
                const int g_col0 = tile_n_base + warp_n_base_ep + tig_ep * 2;
                const int g_col1 = g_col0 + 1;
                if (g_row0 < M && g_col0 < N) C[g_row0 * N + g_col0] = float_to_data<DataT>(acc[mf_ep][nf_ep][0]);
                if (g_row0 < M && g_col1 < N) C[g_row0 * N + g_col1] = float_to_data<DataT>(acc[mf_ep][nf_ep][1]);
                if (g_row1 < M && g_col0 < N) C[g_row1 * N + g_col0] = float_to_data<DataT>(acc[mf_ep][nf_ep][2]);
                if (g_row1 < M && g_col1 < N) C[g_row1 * N + g_col1] = float_to_data<DataT>(acc[mf_ep][nf_ep][3]);
            }
        }
        return;
    }

    // 5) Per-CTA K loop with cp.async pipeline (STAGES stages of overlap).
    //
    // Pattern:
    //   - Prologue: issue cp.async for kt=0 into stage 0, commit
    //   - For kt = 0..K_TILES-1:
    //       if kt+1 < K_TILES: issue cp.async for kt+1 into stage (kt+1)%STAGES, commit
    //       cp.async.wait_group<STAGES-1>: blocks until current stage's loads done
    //       __syncthreads
    //       Load scales/mins (sync, tiny)
    //       __syncthreads
    //       Compute mma using stage `kt % STAGES`
    //
    // This overlaps the HBM latency for the NEXT k-tile with COMPUTE for the current.
    const int K_TILES = K / BLOCK_K;

    constexpr int B_ROWS_PER_TILE_C = BLOCK_K / 16;
    constexpr int B_INTS_PER_TILE_C = B_ROWS_PER_TILE_C * BLOCK_N * 2;
    constexpr int B_QH_INTS_PER_TILE_C = B_ROWS_PER_TILE_C * BLOCK_N;

    // cp.async helper for issuing one kt's A and B (and B_qh for Q6) into a target stage.
    auto issue_kt_async = [&](int kt_target, int dst_stage) {
        // A load — each thread loads 8 fp16 = 16 bytes per cp_async_16 call.
        // BLOCK_M*BLOCK_K / 8 fp16-per-thread = chunks. With 256 threads:
        //   BLOCK_M=64,  BLOCK_K=32 → 8 fp16/thread = 1 cp_async_16 per thread.
        //   BLOCK_M=128, BLOCK_K=32 → 16 fp16/thread = 2 cp_async_16 per thread.
        // The loop handles either via a per-thread chunk count.
        {
            constexpr int A_ELEMS_PER_THREAD = (BLOCK_M * BLOCK_K) / 256;
            constexpr int A_CALLS_PER_THREAD = A_ELEMS_PER_THREAD / 8;
            constexpr int A_ELEMS_PER_CALL = 8;
            static_assert(A_CALLS_PER_THREAD * A_ELEMS_PER_CALL * 256
                          == BLOCK_M * BLOCK_K,
                          "A loader: BLOCK_M*BLOCK_K must factor as 256 × N × 8 fp16");
            #pragma unroll
            for (int c = 0; c < A_CALLS_PER_THREAD; ++c) {
                const int linear_chunk = tid * A_CALLS_PER_THREAD + c;
                const int row = (linear_chunk * A_ELEMS_PER_CALL) / BLOCK_K;
                const int col = (linear_chunk * A_ELEMS_PER_CALL) % BLOCK_K;
                const int gm = tile_m_base + row;
                const int gk = kt_target * BLOCK_K + col;
                DataT* src = const_cast<DataT*>(&A[gm * K + gk]);
                DataT* dst = &sh_A[dst_stage * BLOCK_M * A_STRIDE
                                   + row * A_STRIDE + col];
                cp_async_16(cvt_smem_ptr(dst), src);
            }
        }

        // B_ql load — 256 threads × 1 int each = 256 ints = B_INTS_PER_TILE_C.
        {
            const int linear_id = tid;
            const int row = linear_id / (BLOCK_N * 2);
            const int col = linear_id % (BLOCK_N * 2);
            const int gk_tile = (kt_target * BLOCK_K) / 16 + row;
            const int gn = tile_n_base * 2 + col;
            int* src = const_cast<int*>(&B_ql[gk_tile * (N * 2) + gn]);
            int* dst = &sh_B_ql[dst_stage * B_INTS_PER_TILE_C + row * (BLOCK_N * 2) + col];
            cp_async_4(cvt_smem_ptr(dst), src);
        }

        // B_qh load (Q6_K only) — 128 ints total; only first 128 threads participate.
        if (std::is_same<QuantTag, Q6K>::value && B_qh != nullptr) {
            if (tid < B_QH_INTS_PER_TILE_C) {
                const int row = tid / BLOCK_N;
                const int col = tid % BLOCK_N;
                const int gk_tile = (kt_target * BLOCK_K) / 16 + row;
                const int gn = tile_n_base + col;
                int* src = const_cast<int*>(&B_qh[gk_tile * N + gn]);
                int* dst = &sh_B_qh[dst_stage * B_QH_INTS_PER_TILE_C + row * BLOCK_N + col];
                cp_async_4(cvt_smem_ptr(dst), src);
            }
        }
    };

    // Prologue: prefetch the first STAGES-1 kt tiles. Each prefetch + commit = one group.
    // After prologue, STAGES-1 groups are in flight.
    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < K_TILES) {
            issue_kt_async(s, s);
        }
        cp_async_commit();
    }

    for (int kt = 0; kt < K_TILES; ++kt) {
        const int cur_stage = kt % STAGES;
        // The prefetch issued in iter kt loads the data for iter kt + STAGES - 1.
        const int prefetch_kt = kt + STAGES - 1;

        if (prefetch_kt < K_TILES) {
            const int prefetch_stage = prefetch_kt % STAGES;
            issue_kt_async(prefetch_kt, prefetch_stage);
        }
        cp_async_commit();

        // Wait until the current kt's loads are done (allow up to STAGES-1 in flight).
        cp_async_wait_group<STAGES - 1>();
        __syncthreads();

        // --- LOAD per-kt scales (and mins for Q4_K) — sync, small. ---
        {
            constexpr int sc_loads = SCALES_PER_KT * BLOCK_N;
            const int sub_id_base = (kt * BLOCK_K) / GROUP_SIZE;
            for (int i = tid; i < sc_loads; i += 256) {
                int s_local = i / BLOCK_N;
                int n_local = i % BLOCK_N;
                int sc_idx = (sub_id_base + s_local) * N + tile_n_base + n_local;
                sh_scales[s_local * BLOCK_N + n_local] = eff_scales[sc_idx];
            }
            if (std::is_same<QuantTag, Q4K>::value && eff_mins != nullptr) {
                for (int i = tid; i < sc_loads; i += 256) {
                    int s_local = i / BLOCK_N;
                    int n_local = i % BLOCK_N;
                    int sc_idx = (sub_id_base + s_local) * N + tile_n_base + n_local;
                    sh_mins[s_local * BLOCK_N + n_local] = eff_mins[sc_idx];
                }
            }
        }

        __syncthreads();

        // --- COOPERATIVE DEQUANT: B_packed → sh_B_pair (K-pair packed uint32 layout) ---
        //
        // Each thread writes 4 uint32 entries = 4 (k_pair, n_inner) packed fp16-pairs.
        // The lo half = B[2*k_pair, n_inner], hi half = B[2*k_pair+1, n_inner].
        // This matches the PTX m16n8k16 B fragment layout: lane (rep_id, group_id)
        // reads exactly 2 uint32 (one for the rep*2 K-pair, one for rep*2+8 K-pair).
        //
        // Amortizes the LOP3 + scale fmul + min fsub: per-tile cost = 8 ops/thread
        // (vs 25 ops × MMA_N_PER_WARP per mma per lane in the inline-dequant path).
        {
            constexpr int K_PAIRS_PER_TILE = BLOCK_K / 2;
            constexpr int TOTAL_PAIRS = K_PAIRS_PER_TILE * BLOCK_N;
            constexpr int PAIRS_PER_THREAD = TOTAL_PAIRS / 256;
            static_assert(PAIRS_PER_THREAD * 256 == TOTAL_PAIRS,
                          "BLOCK_K*BLOCK_N must be a multiple of 512");

            #pragma unroll
            for (int i = 0; i < PAIRS_PER_THREAD; ++i) {
                const int lin = tid * PAIRS_PER_THREAD + i;
                const int k_pair = lin / BLOCK_N;     // 0..K_PAIRS_PER_TILE-1
                const int n_inner = lin % BLOCK_N;
                const int k_lo = 2 * k_pair;
                const int k_hi = k_lo + 1;

                // Helper to dequantize one (k_inner, n_inner) → DataT.
                auto deq_one = [&](int k_inner) -> DataT {
                    const int kf = k_inner / 16;
                    const int k_inner_in_kf = k_inner & 15;
                    const int n_tile_id = n_inner / 16;
                    const int n_in_tile = n_inner & 15;
                    const int orig_flat = n_tile_id * 256 + k_inner_in_kf * 16 + n_in_tile;
                    const int permuted = (int)c_PERM_INV[orig_flat];
                    const int int_idx = permuted / 8;
                    const int bit_off = (permuted & 7) * 4;
                    const int packed_ql = sh_B_ql[cur_stage * B_INTS_PER_TILE_C
                                                  + kf * (BLOCK_N * 2) + int_idx];
                    const int nib_ql = (packed_ql >> bit_off) & 0x0F;
                    const int sub_id_rel = k_inner / GROUP_SIZE;
                    const DataT eff_sc = sh_scales[sub_id_rel * BLOCK_N + n_inner];
                    if (std::is_same<QuantTag, Q4K>::value) {
                        const DataT eff_mn = sh_mins[sub_id_rel * BLOCK_N + n_inner];
                        return __hsub(__hmul(int_to_data<DataT>(nib_ql), eff_sc), eff_mn);
                    } else {
                        const int qh_int_idx = permuted / 16;
                        const int qh_bit_off = (permuted & 15) * 2;
                        const int packed_qh = sh_B_qh[cur_stage * B_QH_INTS_PER_TILE_C
                                                       + kf * BLOCK_N + qh_int_idx];
                        const int nib_qh = (packed_qh >> qh_bit_off) & 0x3;
                        const int q6 = nib_ql | (nib_qh << 4);
                        return __hmul(int_to_data<DataT>(q6 - 32), eff_sc);
                    }
                };

                const DataT lo = deq_one(k_lo);
                const DataT hi = deq_one(k_hi);
                sh_B_pair[k_pair * B_PAIR_STRIDE_U32 + n_inner]
                    = pack_pair<DataT>(lo, hi);
            }
        }

        // [FFF Phase 2 runtime check] 2026-06-05: tested replacing this
        // __syncthreads() with named_bar_sync(BAR_BPAIR_FULL, 256) when
        // WARPSPEC=true. Correctness PASSED bit-identical (max_abs=0.0)
        // — verifies the PTX inline-asm wrapper + barrier ID 9 work at
        // runtime. BUT measured +14s e2e wall cost (after noise subtract)
        // with no functional gain — reverted. Phase 3 will introduce real
        // producer/consumer split where the partial arrival count enables
        // overlap; that's where the wall cost amortizes.
        __syncthreads();

        // --- MMA mainloop for this K tile ---
        //
        // For each k-fragment (k-dim is BLOCK_K split into 16-element fragments):
        //   - ldmatrix A fragment from SMEM (per warp's m-slice)
        //   - per n-fragment: ldmatrix B fragment, dequant, mma
        //
        constexpr int K_FRAGS = BLOCK_K / 16;  // BLOCK_K=32 → 2 k-fragments
        for (int kf = 0; kf < K_FRAGS; ++kf) {
            // Build the A fragments per-lane from SMEM (no ldmatrix; manual fp16 reads).
            // Per PTX ISA m16n8k16.row.col.f16 layout for A (16×16):
            //   Lane t = (group_id, rep_id) where group_id = t/4 in 0..7, rep_id = t%4 in 0..3
            //   a[0] = half2(A[group_id+0, rep_id*2+0], A[group_id+0, rep_id*2+1])
            //   a[1] = half2(A[group_id+8, rep_id*2+0], A[group_id+8, rep_id*2+1])
            //   a[2] = half2(A[group_id+0, rep_id*2+8], A[group_id+0, rep_id*2+9])
            //   a[3] = half2(A[group_id+8, rep_id*2+8], A[group_id+8, rep_id*2+9])
            //
            // For each m-fragment (mf): rows are warp_m_base + mf*16 + a_group{, +8}.
            // Hoist A loads OUTSIDE the nf loop so they're shared across n-fragments.
            const int warp_m_base = warp_m * WARP_TILE_M;
            const int a_group = lane / 4;       // 0..7
            const int a_rep   = lane & 3;        // 0..3

            // Build A fragments via ldmatrix.sync.x4 — one PTX instruction loads
            // the four 8x8 matrices of a 16x16 A-tile straight into the per-lane
            // half2 slots mma.m16n8k16 expects. Per lane t supplies the row
            // pointer for matrix (t/8), row (t%8). The mma A fragment ordering
            // is (TL, BL, TR, BR) — i.e. a[0]=top-left, a[1]=bottom-left,
            // a[2]=top-right, a[3]=bottom-right. So ldmatrix maps:
            //   Matrix 0 = TL  (rows 0..7,  cols 0..7)
            //   Matrix 1 = BL  (rows 8..15, cols 0..7)
            //   Matrix 2 = TR  (rows 0..7,  cols 8..15)
            //   Matrix 3 = BR  (rows 8..15, cols 8..15)
            // Per-mf base row is warp_m_base + mf*16.
            uint32_t A_frags[MMA_M_PER_WARP][4];
            const int ld_mat_id = lane >> 3;             // 0..3
            const int ld_row_in_mat = lane & 7;          // 0..7
            const int ld_row_offset = (ld_mat_id & 1) ? 8 : 0;  // mat 1,3 = bottom
            const int ld_col_offset = (ld_mat_id & 2) ? 8 : 0;  // mat 2,3 = right
            #pragma unroll
            for (int mf = 0; mf < MMA_M_PER_WARP; ++mf) {
                const int row = warp_m_base + mf * 16 + ld_row_offset + ld_row_in_mat;
                const int col = kf * 16 + ld_col_offset;
                DataT* row_ptr = &sh_A[cur_stage * BLOCK_M * A_STRIDE
                                       + row * A_STRIDE + col];
                uint32_t smem_a = cvt_smem_ptr(row_ptr);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(A_frags[mf][0]), "=r"(A_frags[mf][1]),
                      "=r"(A_frags[mf][2]), "=r"(A_frags[mf][3])
                    : "r"(smem_a)
                );
            }

            // Per-warp n-fragments. Each lane builds its B fragment from sh_B_pair
            // (the dequant-to-SMEM K-pair layout).
            //
            // PTX ISA §9.7.13.4.6 (mma.sync.m16n8k16.row.col.f16) lane mapping:
            //   group_id = lane / 4,  rep_id = lane % 4
            //   b[0] = (B[rep_id*2 + 0, group_id], B[rep_id*2 + 1, group_id])  → K-pair rep_id
            //   b[1] = (B[rep_id*2 + 8, group_id], B[rep_id*2 + 9, group_id])  → K-pair rep_id+4
            //
            // In sh_B_pair[k_pair][n_inner]: lo = B[2*k_pair, n_inner], hi = B[2*k_pair+1, n_inner].
            // So for k-fragment kf (BLOCK_K positions kf*16..kf*16+15) the absolute k_pair indices are:
            //   b[0] pair = kf*8 + rep_id
            //   b[1] pair = kf*8 + rep_id + 4
            const int group_id = lane / 4;
            const int rep_id = lane & 3;
            const int kf_pair_base = kf * 8;
            const int b0_pair = kf_pair_base + rep_id;
            const int b1_pair = kf_pair_base + rep_id + 4;
            #pragma unroll
            for (int nf = 0; nf < MMA_N_PER_WARP; ++nf) {
                const int warp_n_base = warp_n * WARP_TILE_N + nf * 8;
                const int n_in_BLOCK = warp_n_base + group_id;  // 0..BLOCK_N-1
                const uint32_t b0 = sh_B_pair[b0_pair * B_PAIR_STRIDE_U32 + n_in_BLOCK];
                const uint32_t b1 = sh_B_pair[b1_pair * B_PAIR_STRIDE_U32 + n_in_BLOCK];

                // Apply this B fragment against every m-fragment owned by the warp.
                #pragma unroll
                for (int mf = 0; mf < MMA_M_PER_WARP; ++mf) {
                    mma_m16n8k16_f32<DataT>(
                        acc[mf][nf][0], acc[mf][nf][1], acc[mf][nf][2], acc[mf][nf][3],
                        A_frags[mf][0], A_frags[mf][1], A_frags[mf][2], A_frags[mf][3],
                        b0, b1);
                }
            }
        }

        // Ensure all warps finish reading this stage's SMEM before the next iter's
        // prefetch issues writes to the same stage (STAGES=2 wraps around fast).
        __syncthreads();
    }
    cp_async_wait_all();

    // 6) Epilogue: write the accumulator to global C.
    //
    // Per-warp output: warp_m × 16 rows starting at tile_m_base + warp_m * WARP_TILE_M,
    // warp_n × WARP_TILE_N cols starting at tile_n_base + warp_n * WARP_TILE_N.
    //
    // Each lane writes 4 fp16 per mma fragment.
    //
    // [Epilogue coalescing REJECTED 2026-06-06] Tried pair-packing acc[0]+acc[1]
    // and acc[2]+acc[3] into uint32 stores via pack_pair / inline-PTX
    // cvt.f16x2.f32 / cvt.bf16x2.f32. Both bit-identical outputs but cause +12%
    // wall regression on Q4_K because the existing kernel is at the register
    // edge: introducing the packed-temporary triggers SMEM spill cascade
    // (Q4_K BLOCK_M=128 stack frame: 80 → 192 bytes, +44 bytes spill stores
    // per thread). Scalar stores cost more instructions but use fewer registers.
    #pragma unroll
    for (int mf = 0; mf < MMA_M_PER_WARP; ++mf) {
        #pragma unroll
        for (int nf = 0; nf < MMA_N_PER_WARP; ++nf) {
            // Per fp32 m16n8k16 accumulator, lane t owns 4 output values:
            //   acc[mf][nf][0] = C[group+0, tig*2+0]   (group = lane/4, tig = lane%4)
            //   acc[mf][nf][1] = C[group+0, tig*2+1]
            //   acc[mf][nf][2] = C[group+8, tig*2+0]
            //   acc[mf][nf][3] = C[group+8, tig*2+1]
            const int warp_m_base = warp_m * WARP_TILE_M + mf * 16;
            const int warp_n_base = warp_n * WARP_TILE_N + nf * 8;
            const int group = (lane / 4);
            const int tig   = (lane & 3);
            const int g_row0 = tile_m_base + warp_m_base + group;
            const int g_row1 = tile_m_base + warp_m_base + group + 8;
            const int g_col0 = tile_n_base + warp_n_base + tig * 2;
            const int g_col1 = g_col0 + 1;

            if (g_row0 < M && g_col0 < N) C[g_row0 * N + g_col0] = float_to_data<DataT>(acc[mf][nf][0]);
            if (g_row0 < M && g_col1 < N) C[g_row0 * N + g_col1] = float_to_data<DataT>(acc[mf][nf][1]);
            if (g_row1 < M && g_col0 < N) C[g_row1 * N + g_col0] = float_to_data<DataT>(acc[mf][nf][2]);
            if (g_row1 < M && g_col1 < N) C[g_row1 * N + g_col0 + 1] = float_to_data<DataT>(acc[mf][nf][3]);
        }
    }
}

// ---- Host launcher ----
//
// Picks the template instantiation based on quant_type and launches the kernel.
// Sets the SMEM dynamic limit to 96 KB via cudaFuncSetAttribute (Ada per-CTA cap).

template<typename QuantTag, typename DataT, int BLOCK_M, int BLOCK_N, int BLOCK_K, int STAGES, int GROUP_SIZE, bool WARPSPEC = false>
cudaError_t launch_basiwan(
    const DataT* A, const int* B_ql, const int* B_qh,
    const DataT* eff_scales, const DataT* eff_mins,
    DataT* C,
    int M, int N, int K,
    cudaStream_t stream
) {
    // A row stride must match the kernel's A_STRIDE (BLOCK_K + A_PAD=8).
    constexpr int A_PAD_LAUNCH = 8;
    constexpr int A_STRIDE_LAUNCH = BLOCK_K + A_PAD_LAUNCH;
    constexpr int A_BYTES  = BLOCK_M * A_STRIDE_LAUNCH * sizeof(DataT);
    constexpr int B_QL_BYTES = BLOCK_N * BLOCK_K / 2;
    constexpr int B_QH_BYTES = std::is_same<QuantTag, Q6K>::value ? (BLOCK_N * BLOCK_K / 4) : 0;
    constexpr int SCALES_PER_KT = (BLOCK_K + GROUP_SIZE - 1) / GROUP_SIZE;
    constexpr int SCALES_BYTES = SCALES_PER_KT * BLOCK_N * sizeof(DataT);
    constexpr int MINS_BYTES = std::is_same<QuantTag, Q4K>::value ? SCALES_BYTES : 0;
    // sh_B_pair: dequanted B tile, packed as K-pair uint32 entries (single buffer,
    // not staged). Stride padded by +1 uint32 per K-pair row to break the 4-way
    // bank conflict between rep_id values during the mma B fragment read.
    // WARPSPEC: 2× this for producer/consumer parity exchange (Phase 2+).
    constexpr int B_PAIR_STRIDE_U32_LAUNCH = BLOCK_N + 1;
    constexpr int B_FP16_BYTES_SINGLE_L = (BLOCK_K / 2) * B_PAIR_STRIDE_U32_LAUNCH * sizeof(uint32_t);
    constexpr int B_FP16_BYTES = WARPSPEC ? (2 * B_FP16_BYTES_SINGLE_L) : B_FP16_BYTES_SINGLE_L;
    constexpr int SMEM_BYTES = STAGES * (A_BYTES + B_QL_BYTES + B_QH_BYTES)
                             + SCALES_BYTES + MINS_BYTES
                             + B_FP16_BYTES;

    auto kernel_fn = marlin_kernel<QuantTag, DataT, BLOCK_M, BLOCK_N, BLOCK_K, STAGES, GROUP_SIZE, WARPSPEC>;
    cudaError_t err = cudaFuncSetAttribute(
        kernel_fn,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        96 * 1024);
    if (err != cudaSuccess) return err;

    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M, 1);
    dim3 block(256, 1, 1);
    kernel_fn<<<grid, block, SMEM_BYTES, stream>>>(
        A, B_ql, B_qh, eff_scales, eff_mins, C, M, N, K);
    return cudaGetLastError();
}

// Explicit instantiations — fp16 + bf16, BLOCK_M ∈ {64, 128}.
template cudaError_t launch_basiwan<Q4K, half, 64, 64, 32, 3, 32>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, half, 64, 64, 32, 3, 16>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q4K, half, 128, 64, 32, 3, 32>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, half, 128, 64, 32, 3, 16>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q4K, __nv_bfloat16, 64, 64, 32, 3, 32>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, __nv_bfloat16, 64, 64, 32, 3, 16>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q4K, __nv_bfloat16, 128, 64, 32, 3, 32>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, __nv_bfloat16, 128, 64, 32, 3, 16>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);

// ---- WARPSPEC=true instantiations (Phase 1 scaffold: identical body to
// WARPSPEC=false, exercised by BASIWAN_WARPSPEC=1 env gate.
// Subsequent phases will introduce actual producer/consumer logic). ----
template cudaError_t launch_basiwan<Q4K, half, 64, 64, 32, 3, 32, true>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, half, 64, 64, 32, 3, 16, true>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q4K, half, 128, 64, 32, 3, 32, true>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, half, 128, 64, 32, 3, 16, true>(
    const half*, const int*, const int*, const half*, const half*, half*,
    int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q4K, __nv_bfloat16, 64, 64, 32, 3, 32, true>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, __nv_bfloat16, 64, 64, 32, 3, 16, true>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q4K, __nv_bfloat16, 128, 64, 32, 3, 32, true>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);
template cudaError_t launch_basiwan<Q6K, __nv_bfloat16, 128, 64, 32, 3, 16, true>(
    const __nv_bfloat16*, const int*, const int*, const __nv_bfloat16*, const __nv_bfloat16*,
    __nv_bfloat16*, int, int, int, cudaStream_t);

} // namespace q4k_q6k_basiwan


// ---- PyTorch C++ extension entry ----

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

torch::Tensor basiwan_q4k_q6k_gemm(
    torch::Tensor x,           // (M, K) fp16
    torch::Tensor B_ql,        // (K/16, N*2) int32 — Q4_K packed nibbles OR Q6_K low 4 bits
    torch::Tensor B_qh,        // (K/16, N) int32 — Q6_K high 2 bits; empty for Q4_K
    torch::Tensor eff_scales,  // (K/g, N) fp16
    torch::Tensor eff_mins,    // (K/g, N) fp16 — Q4_K only; empty for Q6_K
    torch::Tensor out,         // (M, N) fp16 — caller-allocated, persistent
    int64_t quant_type,        // 0 = Q4_K, 1 = Q6_K
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
    TORCH_CHECK(M % 64 == 0, "M must be multiple of BLOCK_M=64 for this implementation");
    TORCH_CHECK(N % 64 == 0, "N must be multiple of BLOCK_N=64");
    TORCH_CHECK(K % 32 == 0, "K must be multiple of BLOCK_K=32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaSuccess;
    // BLOCK_M selection: BLOCK_M=128 has ~30% lower per-mma SMEM bandwidth and
    // halves the wave count, but the kernel itself does not handle partial M
    // tiles. We split M into a 128-multiple head + a ≤127-row tail (which the
    // BLOCK_M=64 path then handles in one or two tiles). This way every M ≥ 128
    // benefits, including M=32448 (= 253*128 + 64) and M=7808 (= 61*128 + 0).
    const int M_high = (M / 128) * 128;
    const int M_tail = M - M_high;

    const bool is_bf16 = (x.dtype() == torch::kBFloat16);

    // Dispatch by (DataT, QuantTag). The two-call head+tail pattern is the same
    // for every type; only the template DataT changes.
    #define LAUNCH_HIGH_LOW(QT, DT) do { \
        DT* A_ptr = reinterpret_cast<DT*>(x.data_ptr()); \
        DT* C_ptr = reinterpret_cast<DT*>(out.data_ptr()); \
        DT* A_tail_ptr = A_ptr + (size_t)M_high * K; \
        DT* C_tail_ptr = C_ptr + (size_t)M_high * N; \
        const DT* sc_ptr = reinterpret_cast<const DT*>(eff_scales.data_ptr()); \
        const DT* mn_ptr = (eff_mins.numel() > 0) \
            ? reinterpret_cast<const DT*>(eff_mins.data_ptr()) : nullptr; \
        const int* qh_ptr = (B_qh.numel() > 0) ? B_qh.data_ptr<int>() : nullptr; \
        constexpr int GS = std::is_same<QT, q4k_q6k_basiwan::Q4K>::value ? 32 : 16; \
        if (M_high > 0) { \
            err = q4k_q6k_basiwan::launch_basiwan<QT, DT, 128, 64, 32, 3, GS>( \
                A_ptr, B_ql.data_ptr<int>(), qh_ptr, sc_ptr, mn_ptr, C_ptr, \
                M_high, N, K, stream); \
        } \
        if (err == cudaSuccess && M_tail > 0) { \
            err = q4k_q6k_basiwan::launch_basiwan<QT, DT, 64, 64, 32, 3, GS>( \
                A_tail_ptr, B_ql.data_ptr<int>(), qh_ptr, sc_ptr, mn_ptr, C_tail_ptr, \
                M_tail, N, K, stream); \
        } \
    } while(0)

    if (quant_type == 0) {  // Q4_K
        TORCH_CHECK(eff_mins.is_cuda() && eff_mins.numel() > 0, "Q4_K requires eff_mins");
        TORCH_CHECK(group_size == 32, "Q4_K requires group_size=32");
        if (is_bf16) {
            LAUNCH_HIGH_LOW(q4k_q6k_basiwan::Q4K, __nv_bfloat16);
        } else {
            LAUNCH_HIGH_LOW(q4k_q6k_basiwan::Q4K, half);
        }
    } else if (quant_type == 1) {  // Q6_K
        TORCH_CHECK(B_qh.is_cuda() && B_qh.numel() > 0, "Q6_K requires B_qh");
        TORCH_CHECK(B_qh.dtype() == torch::kInt32, "B_qh must be int32");
        TORCH_CHECK(group_size == 16, "Q6_K requires group_size=16");
        if (is_bf16) {
            LAUNCH_HIGH_LOW(q4k_q6k_basiwan::Q6K, __nv_bfloat16);
        } else {
            LAUNCH_HIGH_LOW(q4k_q6k_basiwan::Q6K, half);
        }
    } else {
        TORCH_CHECK(false, "Unsupported quant_type ", quant_type, " (expected 0=Q4_K, 1=Q6_K)");
    }
    #undef LAUNCH_HIGH_LOW
    TORCH_CHECK(err == cudaSuccess, "marlin kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

// Same as basiwan_q4k_q6k_gemm but dispatches with WARPSPEC=true (separate
// kernel instantiation). Phase 1: bit-identical output (no warp-spec logic
// added yet — the if constexpr (WARPSPEC) branches are empty/no-op).
// Gated by BASIWAN_WARPSPEC=1 in the Python wrapper.
torch::Tensor basiwan_q4k_q6k_gemm_ws(
    torch::Tensor x,
    torch::Tensor B_ql,
    torch::Tensor B_qh,
    torch::Tensor eff_scales,
    torch::Tensor eff_mins,
    torch::Tensor out,
    int64_t quant_type,
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
    TORCH_CHECK(M % 64 == 0, "M must be multiple of BLOCK_M=64");
    TORCH_CHECK(N % 64 == 0, "N must be multiple of BLOCK_N=64");
    TORCH_CHECK(K % 32 == 0, "K must be multiple of BLOCK_K=32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    cudaError_t err = cudaSuccess;
    const int M_high = (M / 128) * 128;
    const int M_tail = M - M_high;
    const bool is_bf16 = (x.dtype() == torch::kBFloat16);

    #define LAUNCH_HIGH_LOW_WS(QT, DT) do { \
        DT* A_ptr = reinterpret_cast<DT*>(x.data_ptr()); \
        DT* C_ptr = reinterpret_cast<DT*>(out.data_ptr()); \
        DT* A_tail_ptr = A_ptr + (size_t)M_high * K; \
        DT* C_tail_ptr = C_ptr + (size_t)M_high * N; \
        const DT* sc_ptr = reinterpret_cast<const DT*>(eff_scales.data_ptr()); \
        const DT* mn_ptr = (eff_mins.numel() > 0) \
            ? reinterpret_cast<const DT*>(eff_mins.data_ptr()) : nullptr; \
        const int* qh_ptr = (B_qh.numel() > 0) ? B_qh.data_ptr<int>() : nullptr; \
        constexpr int GS = std::is_same<QT, q4k_q6k_basiwan::Q4K>::value ? 32 : 16; \
        if (M_high > 0) { \
            err = q4k_q6k_basiwan::launch_basiwan<QT, DT, 128, 64, 32, 3, GS, true>( \
                A_ptr, B_ql.data_ptr<int>(), qh_ptr, sc_ptr, mn_ptr, C_ptr, \
                M_high, N, K, stream); \
        } \
        if (err == cudaSuccess && M_tail > 0) { \
            err = q4k_q6k_basiwan::launch_basiwan<QT, DT, 64, 64, 32, 3, GS, true>( \
                A_tail_ptr, B_ql.data_ptr<int>(), qh_ptr, sc_ptr, mn_ptr, C_tail_ptr, \
                M_tail, N, K, stream); \
        } \
    } while(0)

    if (quant_type == 0) {
        TORCH_CHECK(eff_mins.is_cuda() && eff_mins.numel() > 0, "Q4_K requires eff_mins");
        TORCH_CHECK(group_size == 32, "Q4_K requires group_size=32");
        if (is_bf16) { LAUNCH_HIGH_LOW_WS(q4k_q6k_basiwan::Q4K, __nv_bfloat16); }
        else         { LAUNCH_HIGH_LOW_WS(q4k_q6k_basiwan::Q4K, half);          }
    } else if (quant_type == 1) {
        TORCH_CHECK(B_qh.is_cuda() && B_qh.numel() > 0, "Q6_K requires B_qh");
        TORCH_CHECK(B_qh.dtype() == torch::kInt32, "B_qh must be int32");
        TORCH_CHECK(group_size == 16, "Q6_K requires group_size=16");
        if (is_bf16) { LAUNCH_HIGH_LOW_WS(q4k_q6k_basiwan::Q6K, __nv_bfloat16); }
        else         { LAUNCH_HIGH_LOW_WS(q4k_q6k_basiwan::Q6K, half);          }
    } else {
        TORCH_CHECK(false, "Unsupported quant_type ", quant_type);
    }
    #undef LAUNCH_HIGH_LOW_WS
    TORCH_CHECK(err == cudaSuccess, "marlin_ws kernel launch failed: ", cudaGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("basiwan_q4k_q6k_gemm", &basiwan_q4k_q6k_gemm,
          "Native Marlin-pattern Q4_K + Q6_K matmul (no vLLM GPTQ borrow, no fallback)");
    m.def("basiwan_q4k_q6k_gemm_ws", &basiwan_q4k_q6k_gemm_ws,
          "Warp-specialized variant (FFF/r_2_2_marlin_warpspec_ada). Phase 1 = "
          "bit-identical scaffold; further phases introduce producer/consumer split.");
}

