# Independent reimplementation informed by the Marlin kernel design
# (IST-DASLab/marlin and the vLLM port, both Apache-2.0). Permutation
# tables and the LOP3 fp16 dequant idiom follow Marlin. Citation per the
# authors' request: Frantar et al., arXiv:2408.11743. See CREDITS.md.
"""Offline pack adapter: GGUF Q4_K / Q6_K → our native Marlin-pattern layout.

DESIGN — this is NOT a vLLM-GPTQ retrofit.

The kernel reads quantized weights directly in a m16n8k16 fragment-order layout
that mirrors Marlin's geometric arrangement. We use the SAME `_perm` permutation
table as upstream Marlin (verified bit-equal in `basiwan_perms.py`) because that
order is dictated by Ada's m16n8k16 fragment ownership pattern, not by the GPTQ
algorithm. What we do NOT inherit from Marlin:
  - the symmetric "-8" zero-point fold (Q4_K is asymmetric)
  - the single fp16 scale per group (Q4_K needs scale AND min per sub-block)
  - the int4-only assumption (Q6_K has 6 bits split across two tensors)

Output layouts:

Q4_K (asymmetric, sub-block-32):
  B_packed  : (k // TILE_K, n * TILE_K // 8) int32
              — 4-bit nibbles in fragment-order, 8 per int32
  eff_scales: (k // 32, n) fp16, permuted by _scale_perm
              — per-sub-block effective scale: d × sub_scale_6bit
  eff_mins  : (k // 32, n) fp16, permuted by _scale_perm
              — per-sub-block effective min:   dmin × sub_min_6bit
              (stored POSITIVE; kernel applies `q × scale - min`)

Q6_K (symmetric, sub-block-16):
  B_ql_packed: (k // TILE_K, n * TILE_K // 8) int32
               — low 4 bits in fragment-order, 8 per int32
  B_qh_packed: (k // TILE_K, n * TILE_K // 16) int32
               — high 2 bits in fragment-order, 16 per int32 (4 per byte)
  eff_scales : (k // 16, n) fp16, permuted by _scale_perm
               — per-sub-block: d × scale_int8_signed (no +127/127 normalization;
               kernel multiplies by signed int8 cast to fp16 directly)
  (No mins — Q6_K is symmetric, but inline subtracts 32 to recover signed value)

The m16n8k16 fragment layout uses TILE_K = TILE_N = 16. After the per-row
reshape into (k/16, 16, n/16, 16) and the (k_tile, n_tile, 16, 16) permute, the
final 256-element tile is permuted by Marlin's `_perm` (1024 elements total
across the 4 K-super-tiles).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import gguf
import numpy as np
import torch

from .basiwan_perms import get_perms
from .ref_decode import decode_q4k_super_block, decode_q6k_super_block, get_scale_min_k4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TILE = 16                       # m16n8k16 fragment edge
QK_K = 256                      # super-block size
Q4_K_BLOCK_BYTES = 144          # ggml block_q4_K
Q6_K_BLOCK_BYTES = 210          # ggml block_q6_K
Q4_K_GROUP_SIZE = 32            # Q4_K sub-block (per scale)
Q6_K_GROUP_SIZE = 16            # Q6_K sub-block (per scale)

_PERM_T, _SCALE_PERM_LIST, _SCALE_PERM_SINGLE_LIST = get_perms()
_PERM = _PERM_T  # 1024-element permutation, bit-equal to Marlin upstream
_SCALE_PERM = torch.tensor(_SCALE_PERM_LIST, dtype=torch.long)


class QuantType:
    Q4_K = 0
    Q6_K = 1


@dataclass
class PackedBasiwanWeight:
    """Packed weight + scales for one Linear's matmul.

    `weight`     is the 4-bit packed B for Q4_K, or the low-4-bits ql for Q6_K
    `weight_hi`  is None for Q4_K, or the 2-bit packed qh for Q6_K
    `scales`     per-group effective fp16 scale (already permuted)
    `mins`       per-group effective fp16 min for Q4_K; None for Q6_K
    """
    quant_type: int
    weight: torch.Tensor
    weight_hi: Optional[torch.Tensor]
    scales: torch.Tensor
    mins: Optional[torch.Tensor]
    n: int
    k: int
    group_size: int


# ---------------------------------------------------------------------------
# Q4_K conversion
# ---------------------------------------------------------------------------

def convert_q4k_gguf_to_marlin(
    raw_q4k_bytes: bytes | memoryview | np.ndarray | torch.Tensor,
    n: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one Q4_K weight tensor (shape (n, k) logically) into our packed
    fragment-order layout. Returns (B_packed_int32, eff_scales_fp16, eff_mins_fp16)."""
    if k % QK_K != 0:
        raise ValueError(f"K={k} must be divisible by {QK_K}")
    if n % TILE != 0:
        raise ValueError(f"N={n} must be divisible by {TILE} (fragment tile edge)")

    raw = _as_np_u8(raw_q4k_bytes)
    blocks_per_row = k // QK_K
    expected = n * blocks_per_row * Q4_K_BLOCK_BYTES
    if raw.size != expected:
        raise ValueError(
            f"Q4_K raw bytes size mismatch: expected {expected}, got {raw.size} "
            f"(n={n}, blocks_per_row={blocks_per_row}, bytes/block={Q4_K_BLOCK_BYTES})"
        )

    raw = raw.reshape(n, blocks_per_row, Q4_K_BLOCK_BYTES)
    # Vectorized super-block decode — replaces the per-(row, blk, sub) Python triple-loop.
    # ~100× faster on numpy than the loop version (measured: 30+ min → ~30 s for a Wan
    # expert's full Q4_K weight set on this hardware).
    # Bytes 0:2  → d  (fp16)
    # Bytes 2:4  → dmin (fp16)
    # Bytes 4:16 → 12-byte meta (8 sub-scales × 6 bits, packed per ggml-quants.c)
    # Bytes 16:144 → 128-byte qs (paired low/high nibbles)
    d_f32   = raw[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(n, blocks_per_row)
    dmin_f32 = raw[:, :, 2:4].copy().view(np.float16).astype(np.float32).reshape(n, blocks_per_row)
    meta = raw[:, :, 4:16]                       # (n, bpr, 12)
    qs   = raw[:, :, 16:144]                     # (n, bpr, 128)

    # 6-bit sub-scale unpack — see ggml-quants.c::get_scale_min_k4.
    # For sub 0..3:  sc = meta[sub] & 63;             mn = meta[sub+4] & 63
    # For sub 4..7:  sc = (meta[sub+4]&0xF) | ((meta[sub-4]>>6)<<4)
    #                mn = (meta[sub+4]>>4)  | ((meta[sub  ]>>6)<<4)
    meta_lo = meta[:, :, 0:4].astype(np.int32)   # sub 0..3 raw
    meta_md = meta[:, :, 4:8].astype(np.int32)   # sub-mins low / sub 4..7 raw bits
    meta_hi = meta[:, :, 8:12].astype(np.int32)  # sub 4..7 low-4 bits

    sc6 = np.empty((n, blocks_per_row, 8), dtype=np.int32)
    mn6 = np.empty((n, blocks_per_row, 8), dtype=np.int32)
    sc6[:, :, 0:4] = meta_lo & 63
    mn6[:, :, 0:4] = meta_md & 63
    sc6[:, :, 4:8] = (meta_hi & 0x0F) | ((meta_lo >> 6) << 4)
    mn6[:, :, 4:8] = (meta_hi >> 4)   | ((meta_md >> 6) << 4)

    # eff_scale = d * sc6 ;  eff_min = dmin * mn6 — all per (n, bpr, sub) at once.
    scales = (d_f32[:, :, None] * sc6.astype(np.float32)).astype(np.float16)    # (n, bpr, 8)
    mins   = (dmin_f32[:, :, None] * mn6.astype(np.float32)).astype(np.float16) # (n, bpr, 8)
    scales = scales.reshape(n, blocks_per_row * 8)
    mins   = mins.reshape(n, blocks_per_row * 8)

    # Nibble decode. qs is laid out as 4 byte-pairs of 32 bytes each. Sub-blocks
    # come in pairs: even-sub uses low nibbles of pair, odd-sub uses high nibbles.
    qs_pairs = qs.reshape(n, blocks_per_row, 4, 32)               # (n, bpr, pair, 32)
    nib_lo = (qs_pairs & 0x0F).astype(np.uint8)                   # even sub of each pair
    nib_hi = (qs_pairs >> 4).astype(np.uint8)                     # odd sub of each pair
    # Interleave (lo, hi) along the pair axis → per-sub 32-nibble blocks in order
    # sub=0(lo), 1(hi), 2(lo), 3(hi), …, 7(hi).
    q_super = np.empty((n, blocks_per_row, 8, 32), dtype=np.uint8)
    q_super[:, :, 0::2, :] = nib_lo
    q_super[:, :, 1::2, :] = nib_hi
    q_kn = q_super.reshape(n, blocks_per_row * QK_K)              # (n, k)

    # q_kn has shape (n, k) — transpose to (k, n) for fragment layout (matches Marlin's _pack).
    q_kn_T = q_kn.T.copy()  # (k, n)
    packed = _pack_4bit_marlin(q_kn_T)
    # Scales/mins are stored in raw (k/g, n) order. The kernel reads them by
    # (sub_id, n) — no permutation. Marlin's `_scale_perm` is for its specific
    # scale-loading register pattern, which we don't replicate.
    eff_scales = torch.from_numpy(scales.T.copy()).to(torch.float16).contiguous()
    eff_mins = torch.from_numpy(mins.T.copy()).to(torch.float16).contiguous()
    return packed, eff_scales, eff_mins


# ---------------------------------------------------------------------------
# Q6_K conversion
# ---------------------------------------------------------------------------

def convert_q6k_gguf_to_marlin(
    raw_q6k_bytes: bytes | memoryview | np.ndarray | torch.Tensor,
    n: int,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert one Q6_K weight tensor (shape (n, k)) into our packed
    fragment-order layout. Returns (B_ql_packed, B_qh_packed, eff_scales_fp16).

    Q6_K is symmetric: weight = (q - 32) × (d × scale_int8). The kernel
    re-derives the -32 shift inline. We pack the low 4 bits (ql) and high
    2 bits (qh) into separate int32 tensors so the kernel can recombine.
    """
    if k % QK_K != 0:
        raise ValueError(f"K={k} must be divisible by {QK_K}")
    if n % TILE != 0:
        raise ValueError(f"N={n} must be divisible by {TILE}")

    raw = _as_np_u8(raw_q6k_bytes)
    blocks_per_row = k // QK_K
    expected = n * blocks_per_row * Q6_K_BLOCK_BYTES
    if raw.size != expected:
        raise ValueError(
            f"Q6_K raw bytes size mismatch: expected {expected}, got {raw.size}"
        )

    raw = raw.reshape(n, blocks_per_row, Q6_K_BLOCK_BYTES)
    # Vectorized super-block decode. ql / qh / sc_i8 / d are sliced across the
    # whole (n, bpr) grid via numpy views; the previous per-row, per-blk,
    # per-sub Python loop was ~3M Python ops per FFN-tier Linear.
    ql = raw[:, :, 0:128]                                 # (n, bpr, 128) uint8
    qh = raw[:, :, 128:192]                               # (n, bpr, 64)  uint8
    sc_i8 = raw[:, :, 192:208].copy().view(np.int8)       # (n, bpr, 16)  int8
    d_f32 = raw[:, :, 208:210].copy().view(np.float16).astype(np.float32).reshape(n, blocks_per_row)

    # eff_scale = d * sc_i8 (signed) — apply across all sub-blocks at once.
    scales = (d_f32[:, :, None] * sc_i8.astype(np.float32)).astype(np.float16)
    scales = scales.reshape(n, blocks_per_row * 16)

    # Layout per ref_decode.py: for each half (0 → super-block bytes 0..127,
    # 1 → 128..255), positions l+0/l+32/l+64/l+96 come from ql_h[l], ql_h[l+32]
    # (low nibble), ql_h[l] (high nibble), ql_h[l+32] (high nibble). qh
    # contributes 2 bits per position from qh_h[l]'s bits (>>0, >>2, >>4, >>6).
    # Reshape into half × column-major slots: (n, bpr, 2, 64) for ql, (n, bpr, 2, 32) for qh.
    ql_h = ql.reshape(n, blocks_per_row, 2, 64)           # halves index dim 2
    qh_h = qh.reshape(n, blocks_per_row, 2, 32)

    # ql_lo[..., 0:32] = ql_h[..., 0:32] & 0x0F  → goes to slot l+0
    # ql_lo[..., 32:64] = ql_h[..., 32:64] & 0x0F → slot l+32
    # ql_hi[..., 0:32] = ql_h[..., 0:32] >> 4    → slot l+64
    # ql_hi[..., 32:64] = ql_h[..., 32:64] >> 4  → slot l+96
    q_ql_half = np.empty((n, blocks_per_row, 2, 128), dtype=np.uint8)
    q_ql_half[:, :, :,   0: 32] = ql_h[:, :, :, 0:32]  & 0x0F
    q_ql_half[:, :, :,  32: 64] = ql_h[:, :, :, 32:64] & 0x0F
    q_ql_half[:, :, :,  64: 96] = ql_h[:, :, :, 0:32]  >> 4
    q_ql_half[:, :, :,  96:128] = ql_h[:, :, :, 32:64] >> 4
    q_ql = q_ql_half.reshape(n, blocks_per_row * QK_K)   # (n, k)

    # qh: 4 bit-pair slots per 8-bit byte → 4 sub-positions per row of qh_h[..., l].
    q_qh_half = np.empty((n, blocks_per_row, 2, 128), dtype=np.uint8)
    q_qh_half[:, :, :,   0: 32] = (qh_h >> 0) & 0x3
    q_qh_half[:, :, :,  32: 64] = (qh_h >> 2) & 0x3
    q_qh_half[:, :, :,  64: 96] = (qh_h >> 4) & 0x3
    q_qh_half[:, :, :,  96:128] = (qh_h >> 6) & 0x3
    q_qh = q_qh_half.reshape(n, blocks_per_row * QK_K)

    q_ql_T = q_ql.T.copy()  # (k, n)
    q_qh_T = q_qh.T.copy()  # (k, n)
    ql_packed = _pack_4bit_marlin(q_ql_T)
    qh_packed = _pack_2bit_marlin(q_qh_T)
    # Raw (k/g, n) — no _scale_perm; kernel reads by (sub_id, n).
    eff_scales = torch.from_numpy(scales.T.copy()).to(torch.float16).contiguous()
    return ql_packed, qh_packed, eff_scales


# ---------------------------------------------------------------------------
# Shared bit-pack helpers
# ---------------------------------------------------------------------------

def _pack_4bit_marlin(q_kn: np.ndarray) -> torch.Tensor:
    """Reshape into m16n8k16 fragment tiles, apply Marlin _perm, pack 8 nibbles per int32.

    Input  : q_kn (k, n) uint8, values in [0, 15]
    Output : (k // 16, n * 16 // 8) int32
    """
    k, n = q_kn.shape
    w = torch.from_numpy(q_kn.astype(np.int32, copy=False))
    # (k/16, 16, n/16, 16) → (k/16, n/16, 16, 16) → (k/16, n × 16)
    w = w.reshape(k // TILE, TILE, n // TILE, TILE)
    w = w.permute(0, 2, 1, 3).reshape(k // TILE, n * TILE)
    # Apply fragment-order permutation in 1024-element strides.
    w = w.reshape(-1, _PERM.numel())[:, _PERM].reshape_as(w)
    # Pack 8 4-bit values per int32 (column-interleaved).
    packed = torch.zeros((w.shape[0], w.shape[1] // 8), dtype=torch.int32)
    for i in range(8):
        packed |= (w[:, i::8] & 0x0F) << (4 * i)
    return packed.contiguous()


def _pack_2bit_marlin(q_kn: np.ndarray) -> torch.Tensor:
    """Pack 16 2-bit values per int32 in the same fragment order.

    Input  : q_kn (k, n) uint8, values in [0, 3]
    Output : (k // 16, n * 16 // 16) int32
    """
    k, n = q_kn.shape
    w = torch.from_numpy(q_kn.astype(np.int32, copy=False))
    w = w.reshape(k // TILE, TILE, n // TILE, TILE)
    w = w.permute(0, 2, 1, 3).reshape(k // TILE, n * TILE)
    w = w.reshape(-1, _PERM.numel())[:, _PERM].reshape_as(w)
    packed = torch.zeros((w.shape[0], w.shape[1] // 16), dtype=torch.int32)
    for i in range(16):
        packed |= (w[:, i::16] & 0x03) << (2 * i)
    return packed.contiguous()


def _permute_group_scales(scales_gn: torch.Tensor) -> torch.Tensor:
    """Apply Marlin's scale_perm to the N axis of a (groups, n) scale tensor."""
    g, n = scales_gn.shape
    if n % _SCALE_PERM.numel() != 0:
        # n must be a multiple of 64 for the standard scale permutation.
        # Wan2.2 dims: 5120 % 64 = 0, 13824 % 64 = 0, 2560 % 64 = 0 → all pass.
        raise ValueError(f"n={n} must be a multiple of {_SCALE_PERM.numel()}")
    return scales_gn.reshape(g, -1, _SCALE_PERM.numel())[:, :, _SCALE_PERM].reshape(g, n).contiguous()


def _as_np_u8(raw) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        return np.ascontiguousarray(raw.view(np.uint8))
    if isinstance(raw, torch.Tensor):
        return raw.detach().cpu().contiguous().view(torch.uint8).numpy()
    return np.frombuffer(raw, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def pack_basiwan_weight(weight) -> PackedBasiwanWeight:
    """Convert a GGMLTensor weight (Q4_K or Q6_K) to the packed Marlin layout."""
    qtype = getattr(weight, "tensor_type", None)
    shape = tuple(int(v) for v in getattr(weight, "tensor_shape", weight.shape))
    if len(shape) != 2:
        raise RuntimeError(f"BASIWAN pack requires rank-2 weight, got shape={shape}")
    n, k = shape
    raw = weight.detach().cpu().contiguous().view(torch.uint8).numpy()
    name = getattr(qtype, "name", str(qtype))

    if name == "Q4_K":
        packed, scales, mins = convert_q4k_gguf_to_marlin(raw, n=n, k=k)
        return PackedBasiwanWeight(
            quant_type=QuantType.Q4_K,
            weight=packed,
            weight_hi=None,
            scales=scales,
            mins=mins,
            n=n,
            k=k,
            group_size=Q4_K_GROUP_SIZE,
        )
    if name == "Q6_K":
        ql_packed, qh_packed, scales = convert_q6k_gguf_to_marlin(raw, n=n, k=k)
        return PackedBasiwanWeight(
            quant_type=QuantType.Q6_K,
            weight=ql_packed,
            weight_hi=qh_packed,
            scales=scales,
            mins=None,
            n=n,
            k=k,
            group_size=Q6_K_GROUP_SIZE,
        )
    raise RuntimeError(f"Unsupported quant type for native BASIWAN pack: {name}")


# ---------------------------------------------------------------------------
# Reference dequantizer: invert the packing for correctness validation
# ---------------------------------------------------------------------------

def dequantize_q4k_marlin_reference(
    B_packed: torch.Tensor,
    eff_scales: torch.Tensor,
    eff_mins: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Unpack + apply scale/min to recover fp32 weights. Compare to ggml dequant."""
    # Inverse of _pack_4bit_marlin + _permute_group_scales.
    packed = B_packed.to(torch.int64)
    # Unpack 8 nibbles per int32 back into (k_tiles, n*16) uint8.
    k_tiles = packed.shape[0]
    unpacked = torch.empty((k_tiles, packed.shape[1] * 8), dtype=torch.int32)
    for i in range(8):
        unpacked[:, i::8] = (packed >> (4 * i)) & 0x0F
    # Invert _perm.
    inv_perm = torch.empty_like(_PERM)
    inv_perm[_PERM] = torch.arange(_PERM.numel(), dtype=_PERM.dtype)
    unpacked = unpacked.reshape(-1, _PERM.numel())[:, inv_perm].reshape(k_tiles, n * TILE)
    # Inverse of the (k/16, n/16, 16, 16) permute + reshape:
    # (k_tiles, n × 16) → (k_tiles, n/16, 16, 16) → (k_tiles, 16, n/16, 16) → (k, n).
    nib = unpacked.reshape(k_tiles, n // TILE, TILE, TILE).permute(0, 2, 1, 3).reshape(k, n)

    # Scales/mins are now stored raw (no _scale_perm applied in pack).
    sc = eff_scales.to(torch.float32)
    mn = eff_mins.to(torch.float32)

    # Reconstruct: weight[k, n] = nib[k, n] × sc[k/g, n] - mn[k/g, n].
    # sc/mn shape is (g, n) with g = k/group_size. Expand to (k, n).
    g, n_ = sc.shape
    group_size = k // g
    sc_kn = sc.repeat_interleave(group_size, dim=0)  # (k, n)
    mn_kn = mn.repeat_interleave(group_size, dim=0)
    return nib.to(torch.float32) * sc_kn - mn_kn


def dequantize_q6k_marlin_reference(
    B_ql_packed: torch.Tensor,
    B_qh_packed: torch.Tensor,
    eff_scales: torch.Tensor,
    n: int,
    k: int,
) -> torch.Tensor:
    """Unpack Q6_K pack + apply scale (symmetric, no mins). Compare to ggml dequant."""
    # Unpack ql (4 bits) and qh (2 bits) separately.
    packed_l = B_ql_packed.to(torch.int64)
    packed_h = B_qh_packed.to(torch.int64)
    k_tiles = packed_l.shape[0]

    ql = torch.empty((k_tiles, packed_l.shape[1] * 8), dtype=torch.int32)
    for i in range(8):
        ql[:, i::8] = (packed_l >> (4 * i)) & 0x0F
    qh = torch.empty((k_tiles, packed_h.shape[1] * 16), dtype=torch.int32)
    for i in range(16):
        qh[:, i::16] = (packed_h >> (2 * i)) & 0x03

    # Invert _perm.
    inv_perm = torch.empty_like(_PERM)
    inv_perm[_PERM] = torch.arange(_PERM.numel(), dtype=_PERM.dtype)
    ql = ql.reshape(-1, _PERM.numel())[:, inv_perm].reshape(k_tiles, n * TILE)
    qh = qh.reshape(-1, _PERM.numel())[:, inv_perm].reshape(k_tiles, n * TILE)
    ql = ql.reshape(k_tiles, n // TILE, TILE, TILE).permute(0, 2, 1, 3).reshape(k, n)
    qh = qh.reshape(k_tiles, n // TILE, TILE, TILE).permute(0, 2, 1, 3).reshape(k, n)

    # Combine + sign-shift: q_signed = (ql | (qh << 4)) - 32, range [-32, 31].
    q_signed = (ql | (qh << 4)).to(torch.int32) - 32

    # Scales now stored raw (no _scale_perm applied in pack).
    sc = eff_scales.to(torch.float32)

    g, _ = sc.shape
    group_size = k // g
    sc_kn = sc.repeat_interleave(group_size, dim=0)
    return q_signed.to(torch.float32) * sc_kn
