"""CPU reference decoder for GGUF Q4_K and Q6_K super-blocks.

Pure-Python decode matching ggml-quants.c's row functions. Used as ground truth
when validating the offline pack adapter (`pack.py`) AND the CUDA Marlin kernel.

Layouts confirmed against llama.cpp `ggml-common.h`:

Q4_K (144 bytes per 256 values):
    d       fp16   (2 B)   super-block scale
    dmin    fp16   (2 B)   super-block min-scale
    scales  u8[12]         8 sub-scales + 8 sub-mins (6-bit packed)
    qs      u8[128]        256 × 4-bit nibbles

Q6_K (210 bytes per 256 values):
    ql      u8[128]        low 4 bits per value
    qh      u8[64]         high 2 bits per value (packed 4-per-byte)
    scales  i8[16]         signed 8-bit sub-scale per 16-value sub-block
    d       fp16   (2 B)   super-block scale
"""
from __future__ import annotations

import numpy as np


# ---- Q4_K ----

def get_scale_min_k4(j: int, scales: np.ndarray) -> tuple[int, int]:
    """Unpack the j-th 6-bit sub-scale and sub-min from a Q4_K super-block's 12-byte
    `scales` field. Returns (sc, mn), both 0..63.

    Matches `ggml-quants.c:get_scale_min_k4`.
    """
    if j < 4:
        sc = scales[j] & 63
        mn = scales[j + 4] & 63
    else:
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4)
        mn = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
    return int(sc), int(mn)


def decode_q4k_super_block(block_bytes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one Q4_K super-block (144 bytes) into 256 fp32 weights.

    Returns:
        weights: (256,) float32 — the dequantized values
        sub_scales: (8,) float32 — eff_scale per sub-block (d * sc_6bit), for the Marlin pack
        sub_mins:   (8,) float32 — eff_min per sub-block (dmin * mn_6bit), for the Marlin pack
    """
    assert len(block_bytes) == 144, f"Q4_K block must be 144 bytes, got {len(block_bytes)}"
    arr = np.frombuffer(block_bytes, dtype=np.uint8)
    d = np.frombuffer(arr[0:2].tobytes(), dtype=np.float16).item()
    dmin = np.frombuffer(arr[2:4].tobytes(), dtype=np.float16).item()
    scales = arr[4:16]
    qs = arr[16:144]

    weights = np.empty(256, dtype=np.float32)
    sub_scales = np.empty(8, dtype=np.float32)
    sub_mins = np.empty(8, dtype=np.float32)
    is_min = -1.0  # mins are subtracted

    # Each sub-block covers 32 contiguous values. Across 8 sub-blocks: 256 total.
    # Quantized layout per sub-block uses pairs of bytes (low nibble for one set,
    # high nibble for the next). Matches ggml's `dequantize_row_q4_K` loop.
    for sub in range(8):
        sc6, mn6 = get_scale_min_k4(sub, scales)
        eff_sc = d * sc6
        eff_mn = dmin * mn6
        sub_scales[sub] = eff_sc
        sub_mins[sub] = eff_mn

        # Within each pair of consecutive sub-blocks, the 32 quants are interleaved
        # as low/high nibbles of the same 16 bytes. sub indexes pairs (sub//2) and
        # nibble half (sub & 1).
        pair_idx = sub // 2
        half = sub & 1
        qs_pair = qs[pair_idx * 32:(pair_idx + 1) * 32]  # 32 bytes shared by 2 sub-blocks
        if half == 0:
            nibbles = (qs_pair & 0x0F).astype(np.int32)
        else:
            nibbles = (qs_pair >> 4).astype(np.int32)

        # Q4_K formula: w = nibble * eff_sc - eff_mn
        weights[sub * 32:(sub + 1) * 32] = nibbles.astype(np.float32) * eff_sc - eff_mn

    return weights, sub_scales, sub_mins


# ---- Q6_K ----

def decode_q6k_super_block(block_bytes: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Decode one Q6_K super-block (210 bytes) into 256 fp32 weights.

    Returns:
        weights: (256,) float32
        sub_scales: (16,) float32 — eff_scale per sub-block (d * scale_int8_signed)
    """
    assert len(block_bytes) == 210, f"Q6_K block must be 210 bytes, got {len(block_bytes)}"
    ql = np.frombuffer(block_bytes[0:128], dtype=np.uint8)
    qh = np.frombuffer(block_bytes[128:192], dtype=np.uint8)
    scales_i8 = np.frombuffer(block_bytes[192:208], dtype=np.int8)
    d = np.frombuffer(block_bytes[208:210], dtype=np.float16).item()

    weights = np.empty(256, dtype=np.float32)
    sub_scales = np.empty(16, dtype=np.float32)
    for sub in range(16):
        sc = float(scales_i8[sub])
        sub_scales[sub] = d * sc

    # Q6_K's ggml layout interleaves two 128-value halves. Within each half,
    # 64 ql bytes hold low-4-bit pairs; 32 qh bytes hold high-2-bit fours.
    # Two sub-blocks per "halfpair" of 32 values each (?) — match the C code:
    #   for (i = 0; i < 256; i+=128)
    #     for (l = 0; l < 32; l++):
    #        q1 = ((ql[l]    & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32
    #        q2 = ((ql[l+32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32
    #        q3 = ((ql[l]    >> 4)   | (((qh[l] >> 4) & 3) << 4)) - 32
    #        q4 = ((ql[l+32] >> 4)   | (((qh[l] >> 6) & 3) << 4)) - 32
    #        y[l+ 0] = d * sc[is+0] * q1
    #        y[l+32] = d * sc[is+2] * q2
    #        y[l+64] = d * sc[is+4] * q3
    #        y[l+96] = d * sc[is+6] * q4
    #        is starts at 0, increments by 8 for the second i=128 half
    for i_half, half_start in enumerate((0, 128)):
        ql_half = ql[i_half * 64:(i_half + 1) * 64]
        qh_half = qh[i_half * 32:(i_half + 1) * 32]
        is_off = i_half * 8  # 0 or 8 — selects between the two 8-scale halves
        for l in range(32):
            # is = l // 16 → 0 for first 16 lanes, 1 for next 16 (within this half)
            is_in_half = l // 16
            sc_base = is_off + is_in_half
            q1 = int((ql_half[l]      & 0x0F) | (((qh_half[l] >> 0) & 0x3) << 4)) - 32
            q2 = int((ql_half[l + 32] & 0x0F) | (((qh_half[l] >> 2) & 0x3) << 4)) - 32
            q3 = int((ql_half[l]      >> 4)   | (((qh_half[l] >> 4) & 0x3) << 4)) - 32
            q4 = int((ql_half[l + 32] >> 4)   | (((qh_half[l] >> 6) & 0x3) << 4)) - 32
            weights[half_start + l +  0] = d * float(scales_i8[sc_base + 0]) * q1
            weights[half_start + l + 32] = d * float(scales_i8[sc_base + 2]) * q2
            weights[half_start + l + 64] = d * float(scales_i8[sc_base + 4]) * q3
            weights[half_start + l + 96] = d * float(scales_i8[sc_base + 6]) * q4

    return weights, sub_scales


# ---- self-test ----

if __name__ == "__main__":
    # Compare against llama.cpp's dequantize via gguf-python by loading a real tensor.
    import sys
    from pathlib import Path

    import os
    GGUF_PATH = Path(
        os.environ.get("BASIWAN_REF_GGUF", "")
        or (Path(os.environ.get("BASIWAN_CKPT_DIR", "checkpoints"))
            / "gguf/models--QuantStack--Wan2.2-T2V-A14B-GGUF/snapshots"
            / "73eafba53a1a8f29254e4c77f92e74ea27d7cd6f"
            / "HighNoise/Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"))
    if not GGUF_PATH.exists():
        print(f"skip: {GGUF_PATH} not present", file=sys.stderr)
        sys.exit(0)

    import gguf  # type: ignore

    reader = gguf.GGUFReader(str(GGUF_PATH))

    # Pick a Q4_K tensor and a Q6_K tensor
    q4k_tensor = None
    q6k_tensor = None
    for t in reader.tensors:
        if t.tensor_type == gguf.GGMLQuantizationType.Q4_K and q4k_tensor is None:
            q4k_tensor = t
        elif t.tensor_type == gguf.GGMLQuantizationType.Q6_K and q6k_tensor is None:
            q6k_tensor = t
        if q4k_tensor is not None and q6k_tensor is not None:
            break

    # ---- Q4_K self-check ----
    print(f"Q4_K tensor: {q4k_tensor.name}, shape={tuple(q4k_tensor.shape)}")
    raw = q4k_tensor.data.tobytes()
    # Decode first super-block manually
    first_block_bytes = raw[:144]
    w_manual, sc_manual, mn_manual = decode_q4k_super_block(first_block_bytes)
    # Decode via ggml's reference
    w_ref = gguf.quants.dequantize(q4k_tensor.data, q4k_tensor.tensor_type)
    w_ref_first = w_ref.flatten()[:256].astype(np.float32)
    diff = np.abs(w_manual - w_ref_first).max()
    print(f"  first 256 values: manual vs ggml max-abs diff = {diff:.6g}  "
          f"({'PASS' if diff < 1e-3 else 'FAIL'})")
    print(f"  sub_scales[0..3] = {sc_manual[:4]}")
    print(f"  sub_mins[0..3]   = {mn_manual[:4]}")

    # ---- Q6_K self-check ----
    print(f"\nQ6_K tensor: {q6k_tensor.name}, shape={tuple(q6k_tensor.shape)}")
    raw = q6k_tensor.data.tobytes()
    first_block_bytes = raw[:210]
    w_manual, sc_manual = decode_q6k_super_block(first_block_bytes)
    w_ref = gguf.quants.dequantize(q6k_tensor.data, q6k_tensor.tensor_type)
    w_ref_first = w_ref.flatten()[:256].astype(np.float32)
    diff = np.abs(w_manual - w_ref_first).max()
    print(f"  first 256 values: manual vs ggml max-abs diff = {diff:.6g}  "
          f"({'PASS' if diff < 1e-3 else 'FAIL'})")
    print(f"  sub_scales[0..3] = {sc_manual[:4]}")
