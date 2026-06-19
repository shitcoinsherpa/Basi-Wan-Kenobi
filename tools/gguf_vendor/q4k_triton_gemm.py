"""Triton fused GGUF Q4_K linear layers with inline decode.

This path keeps the GGUF Q4_K payload in its original packed layout:

- 256 logical weights per super-block
- 16 header bytes per super-block:
  - 2 bytes `d`
  - 2 bytes `dmin`
  - 12 bytes packed scale/min metadata
- 128 packed quant bytes per super-block

The Triton kernel loads the raw bytes directly inside the K loop, decodes the
two 4-bit nibbles for each packed byte, applies the per-32-value scale/min,
accumulates in fp32, and stores the requested output dtype. No prepared cache
is attached to the GGMLTensor.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings
from typing import Sequence

import gguf
import torch
import torch.nn.functional as F

try:
    from .dequant import K_SCALE_SIZE, QK_K, dequantize
except ImportError:  # pragma: no cover - supports direct script execution
    from dequant import K_SCALE_SIZE, QK_K, dequantize

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import failure is handled at runtime
    triton = None
    tl = None


Q4_K_QTYPE = gguf.GGMLQuantizationType.Q4_K
Q4_K_BLOCK_SIZE, Q4_K_TYPE_SIZE = gguf.GGML_QUANT_SIZES[Q4_K_QTYPE]
Q4_K_META_BYTES = 2 + 2 + K_SCALE_SIZE
Q4_K_SUBBLOCK = 32
Q4_K_SUBBLOCKS_PER_SUPERBLOCK = QK_K // Q4_K_SUBBLOCK
Q4_K_QBYTES_PER_SUPERBLOCK = QK_K // 2
SUPPORTED_X_DTYPES = {torch.float16, torch.bfloat16, torch.float32}
# BLOCK_M is fixed at 32 for M>32 — measured 2026-06-02 on RTX 4090 at real M=7800:
#   BLOCK_M=32 → 193.6s end-to-end (the prior bench)
#   BLOCK_M=64 → 234.0s end-to-end (worse — more bandwidth per CTA, no CTA-thrash gain)
#   BLOCK_M=128+ → SMEM OoR (Required 136 KB vs Ada limit 99 KB)
# The v1.1 fused-Q4 kernel cannot beat baseline cuBLAS dequant+matmul (177.2s) on Ada at
# our M=7800 production point. Documented in q4_v1.1_ceiling_2026-06-02 memory.
# v1.1 stays opt-in via BASIWAN_Q4_FUSED_GEMM=1; default = baseline dequant+linear.
BLOCK_M_CHOICES = (16, 32)
LARGE_M_GRID_CAP = 32


def _truthy_env(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def fused_q4k_env_enabled() -> bool:
    return _truthy_env("BASIWAN_Q4_FUSED_GEMM")


def _triton_runtime_available_for_device(device: torch.device) -> bool:
    if triton is None:
        return False
    if device.type == "cuda":
        return torch.cuda.is_available()
    return device.type == "cpu" and _truthy_env("TRITON_INTERPRET")


def _tensor_is_q4k(weight: torch.Tensor) -> bool:
    qtype = getattr(weight, "tensor_type", None)
    if qtype == Q4_K_QTYPE:
        return True
    return getattr(qtype, "name", None) == "Q4_K"


def _logical_shape(
    logical_shape: Sequence[int] | torch.Size | None,
) -> tuple[int, int]:
    if logical_shape is None or len(logical_shape) != 2:
        raise ValueError(f"expected a 2D logical shape, got {logical_shape!r}")
    return (int(logical_shape[0]), int(logical_shape[1]))


def _validate_q4k_raw_weight(
    raw_weight: torch.Tensor,
    logical_shape: Sequence[int] | torch.Size,
) -> tuple[int, int, int]:
    n, k = _logical_shape(logical_shape)
    if raw_weight.ndim != 2:
        raise ValueError(f"expected packed weight to be 2D, got {tuple(raw_weight.shape)}")
    if raw_weight.dtype != torch.uint8:
        raise ValueError(f"expected packed weight dtype=torch.uint8, got {raw_weight.dtype}")
    if k % QK_K != 0:
        raise ValueError(f"Q4_K requires K multiple of {QK_K}, got {k}")
    packed_cols = (k // QK_K) * Q4_K_TYPE_SIZE
    if tuple(raw_weight.shape) != (n, packed_cols):
        raise ValueError(
            f"packed weight shape mismatch: expected {(n, packed_cols)}, got {tuple(raw_weight.shape)}"
        )
    return n, k, packed_cols


def _select_launch_config(m_size: int, n_size: int) -> tuple[int, int, int, int]:
    block_n = 64
    if m_size <= 32:
        return 16, block_n, 4, 2

    # Keep the tiny-M prefill tile, then scale M-tiles so the launch grid
    # does not explode on long-sequence forward passes.
    block_m = BLOCK_M_CHOICES[-1]
    for candidate in BLOCK_M_CHOICES[1:]:
        if (m_size + candidate - 1) // candidate <= LARGE_M_GRID_CAP:
            block_m = candidate
            break
    # num_warps=4 stays — BLOCK_M ≤ 64 doesn't need 8 warps (register pressure was the
    # original justification at BLOCK_M ≥ 256, which we no longer reach).
    return block_m, block_n, 4, 2


if triton is not None:

    @triton.jit
    def _q4k_inline_gemm_kernel(
        x_ptr,
        w_ptr,
        out_ptr,
        m_size,
        n_size,
        k_size,
        stride_xm,
        stride_xk,
        stride_wn,
        stride_wk,
        stride_outm,
        stride_outn,
        qk_k: tl.constexpr,
        type_size: tl.constexpr,
        meta_bytes: tl.constexpr,
        subblock: tl.constexpr,
        qbytes_per_superblock: tl.constexpr,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * block_m + tl.arange(0, block_m)
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        mask_m = offs_m < m_size
        mask_n = offs_n < n_size

        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        sub_idx = tl.arange(0, subblock)
        for superblock_idx in range(0, tl.cdiv(k_size, qk_k)):
            superblock_k_base = superblock_idx * qk_k
            superblock_byte_base = superblock_idx * type_size

            row_base = offs_n * stride_wn + superblock_byte_base * stride_wk
            d_bits = (
                tl.load(w_ptr + row_base + 0 * stride_wk, mask=mask_n, other=0).to(tl.uint16)
                | (
                    tl.load(w_ptr + row_base + 1 * stride_wk, mask=mask_n, other=0).to(tl.uint16)
                    << 8
                )
            )
            dmin_bits = (
                tl.load(w_ptr + row_base + 2 * stride_wk, mask=mask_n, other=0).to(tl.uint16)
                | (
                    tl.load(w_ptr + row_base + 3 * stride_wk, mask=mask_n, other=0).to(tl.uint16)
                    << 8
                )
            )
            d = tl.cast(d_bits, tl.float16, bitcast=True).to(tl.float32)
            dmin = tl.cast(dmin_bits, tl.float16, bitcast=True).to(tl.float32)

            s0 = tl.load(w_ptr + row_base + 4 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            s1 = tl.load(w_ptr + row_base + 5 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            s2 = tl.load(w_ptr + row_base + 6 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            s3 = tl.load(w_ptr + row_base + 7 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            m0 = tl.load(w_ptr + row_base + 8 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            m1 = tl.load(w_ptr + row_base + 9 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            m2 = tl.load(w_ptr + row_base + 10 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            m3 = tl.load(w_ptr + row_base + 11 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            sm0 = tl.load(w_ptr + row_base + 12 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            sm1 = tl.load(w_ptr + row_base + 13 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            sm2 = tl.load(w_ptr + row_base + 14 * stride_wk, mask=mask_n, other=0).to(tl.int32)
            sm3 = tl.load(w_ptr + row_base + 15 * stride_wk, mask=mask_n, other=0).to(tl.int32)

            eff_scales = (
                (s0 & 0x3F).to(tl.float32) * d,
                (s1 & 0x3F).to(tl.float32) * d,
                (s2 & 0x3F).to(tl.float32) * d,
                (s3 & 0x3F).to(tl.float32) * d,
                ((sm0 & 0x0F) | ((s0 >> 2) & 0x30)).to(tl.float32) * d,
                ((sm1 & 0x0F) | ((s1 >> 2) & 0x30)).to(tl.float32) * d,
                ((sm2 & 0x0F) | ((s2 >> 2) & 0x30)).to(tl.float32) * d,
                ((sm3 & 0x0F) | ((s3 >> 2) & 0x30)).to(tl.float32) * d,
            )
            eff_mins = (
                (m0 & 0x3F).to(tl.float32) * dmin,
                (m1 & 0x3F).to(tl.float32) * dmin,
                (m2 & 0x3F).to(tl.float32) * dmin,
                (m3 & 0x3F).to(tl.float32) * dmin,
                ((sm0 >> 4) | ((m0 >> 2) & 0x30)).to(tl.float32) * dmin,
                ((sm1 >> 4) | ((m1 >> 2) & 0x30)).to(tl.float32) * dmin,
                ((sm2 >> 4) | ((m2 >> 2) & 0x30)).to(tl.float32) * dmin,
                ((sm3 >> 4) | ((m3 >> 2) & 0x30)).to(tl.float32) * dmin,
            )

            # Fully unrolled — 4 qbyte chunks × 2 nibbles = 8 sub-blocks per
            # superblock. We emit each chunk explicitly so `eff_scales[i]`
            # is a Python-int literal lookup that the Triton AST parser
            # accepts. Loops with tuple destructuring or tl.static_range
            # both got rejected (parser sees runtime tuple indexing).

            # === chunk 0 (sub-blocks 0,1) ===
            q_offsets = row_base[:, None] + (meta_bytes + 0 * subblock + sub_idx)[None, :] * stride_wk
            q_packed = tl.load(w_ptr + q_offsets, mask=mask_n[:, None], other=0).to(tl.int32)
            low_k_base = superblock_k_base + 0 * subblock
            x_low_offsets = offs_m[:, None] * stride_xm + (low_k_base + sub_idx)[None, :] * stride_xk
            x_high_offsets = offs_m[:, None] * stride_xm + (low_k_base + subblock + sub_idx)[None, :] * stride_xk
            x_low = tl.load(x_ptr + x_low_offsets, mask=mask_m[:, None], other=0.0)
            x_high = tl.load(x_ptr + x_high_offsets, mask=mask_m[:, None], other=0.0)
            w_low = (q_packed & 0x0F).to(tl.float32) * eff_scales[0][:, None] - eff_mins[0][:, None]
            w_high = ((q_packed >> 4) & 0x0F).to(tl.float32) * eff_scales[1][:, None] - eff_mins[1][:, None]
            acc += tl.dot(x_low, tl.trans(w_low.to(x_low.dtype)), out_dtype=tl.float32)
            acc += tl.dot(x_high, tl.trans(w_high.to(x_high.dtype)), out_dtype=tl.float32)

            # === chunk 1 (sub-blocks 2,3) ===
            q_offsets = row_base[:, None] + (meta_bytes + 1 * subblock + sub_idx)[None, :] * stride_wk
            q_packed = tl.load(w_ptr + q_offsets, mask=mask_n[:, None], other=0).to(tl.int32)
            low_k_base = superblock_k_base + 2 * subblock
            x_low_offsets = offs_m[:, None] * stride_xm + (low_k_base + sub_idx)[None, :] * stride_xk
            x_high_offsets = offs_m[:, None] * stride_xm + (low_k_base + subblock + sub_idx)[None, :] * stride_xk
            x_low = tl.load(x_ptr + x_low_offsets, mask=mask_m[:, None], other=0.0)
            x_high = tl.load(x_ptr + x_high_offsets, mask=mask_m[:, None], other=0.0)
            w_low = (q_packed & 0x0F).to(tl.float32) * eff_scales[2][:, None] - eff_mins[2][:, None]
            w_high = ((q_packed >> 4) & 0x0F).to(tl.float32) * eff_scales[3][:, None] - eff_mins[3][:, None]
            acc += tl.dot(x_low, tl.trans(w_low.to(x_low.dtype)), out_dtype=tl.float32)
            acc += tl.dot(x_high, tl.trans(w_high.to(x_high.dtype)), out_dtype=tl.float32)

            # === chunk 2 (sub-blocks 4,5) ===
            q_offsets = row_base[:, None] + (meta_bytes + 2 * subblock + sub_idx)[None, :] * stride_wk
            q_packed = tl.load(w_ptr + q_offsets, mask=mask_n[:, None], other=0).to(tl.int32)
            low_k_base = superblock_k_base + 4 * subblock
            x_low_offsets = offs_m[:, None] * stride_xm + (low_k_base + sub_idx)[None, :] * stride_xk
            x_high_offsets = offs_m[:, None] * stride_xm + (low_k_base + subblock + sub_idx)[None, :] * stride_xk
            x_low = tl.load(x_ptr + x_low_offsets, mask=mask_m[:, None], other=0.0)
            x_high = tl.load(x_ptr + x_high_offsets, mask=mask_m[:, None], other=0.0)
            w_low = (q_packed & 0x0F).to(tl.float32) * eff_scales[4][:, None] - eff_mins[4][:, None]
            w_high = ((q_packed >> 4) & 0x0F).to(tl.float32) * eff_scales[5][:, None] - eff_mins[5][:, None]
            acc += tl.dot(x_low, tl.trans(w_low.to(x_low.dtype)), out_dtype=tl.float32)
            acc += tl.dot(x_high, tl.trans(w_high.to(x_high.dtype)), out_dtype=tl.float32)

            # === chunk 3 (sub-blocks 6,7) ===
            q_offsets = row_base[:, None] + (meta_bytes + 3 * subblock + sub_idx)[None, :] * stride_wk
            q_packed = tl.load(w_ptr + q_offsets, mask=mask_n[:, None], other=0).to(tl.int32)
            low_k_base = superblock_k_base + 6 * subblock
            x_low_offsets = offs_m[:, None] * stride_xm + (low_k_base + sub_idx)[None, :] * stride_xk
            x_high_offsets = offs_m[:, None] * stride_xm + (low_k_base + subblock + sub_idx)[None, :] * stride_xk
            x_low = tl.load(x_ptr + x_low_offsets, mask=mask_m[:, None], other=0.0)
            x_high = tl.load(x_ptr + x_high_offsets, mask=mask_m[:, None], other=0.0)
            w_low = (q_packed & 0x0F).to(tl.float32) * eff_scales[6][:, None] - eff_mins[6][:, None]
            w_high = ((q_packed >> 4) & 0x0F).to(tl.float32) * eff_scales[7][:, None] - eff_mins[7][:, None]
            acc += tl.dot(x_low, tl.trans(w_low.to(x_low.dtype)), out_dtype=tl.float32)
            acc += tl.dot(x_high, tl.trans(w_high.to(x_high.dtype)), out_dtype=tl.float32)


        out_offsets = offs_m[:, None] * stride_outm + offs_n[None, :] * stride_outn
        out_mask = mask_m[:, None] & mask_n[None, :]
        tl.store(out_ptr + out_offsets, acc.to(out_ptr.type.element_ty), mask=out_mask)


def q4k_gemm(
    x: torch.Tensor,
    raw_weight: torch.Tensor,
    logical_shape: Sequence[int] | torch.Size,
    *,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got {tuple(x.shape)}")
    if x.dtype not in SUPPORTED_X_DTYPES:
        raise TypeError(f"unsupported activation dtype for fused Q4_K GEMM: {x.dtype}")
    n_size, k_size, _ = _validate_q4k_raw_weight(raw_weight, logical_shape)
    if x.shape[1] != k_size:
        raise ValueError(f"x K mismatch: expected {k_size}, got {x.shape[1]}")
    if not _triton_runtime_available_for_device(x.device):
        raise RuntimeError(
            f"Triton runtime is not available for device {x.device}; "
            "use CUDA or set TRITON_INTERPRET=1 for CPU interpretation"
        )

    target_dtype = out_dtype or x.dtype
    out = torch.empty((int(x.shape[0]), n_size), device=x.device, dtype=target_dtype)
    return q4k_gemm_out(x, raw_weight, logical_shape, out=out)


def q4k_gemm_out(
    x: torch.Tensor,
    raw_weight: torch.Tensor,
    logical_shape: Sequence[int] | torch.Size,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is not available")
    if x.ndim != 2:
        raise ValueError(f"expected x to be 2D, got {tuple(x.shape)}")
    if x.dtype not in SUPPORTED_X_DTYPES:
        raise TypeError(f"unsupported activation dtype for fused Q4_K GEMM: {x.dtype}")
    n_size, k_size, _ = _validate_q4k_raw_weight(raw_weight, logical_shape)
    if x.shape[1] != k_size:
        raise ValueError(f"x K mismatch: expected {k_size}, got {x.shape[1]}")
    if not _triton_runtime_available_for_device(x.device):
        raise RuntimeError(
            f"Triton runtime is not available for device {x.device}; "
            "use CUDA or set TRITON_INTERPRET=1 for CPU interpretation"
        )

    expected_shape = (int(x.shape[0]), n_size)
    if tuple(out.shape) != expected_shape:
        raise ValueError(f"out shape mismatch: expected {expected_shape}, got {tuple(out.shape)}")
    if out.device != x.device:
        raise ValueError(f"out device mismatch: expected {x.device}, got {out.device}")
    if out.dtype not in SUPPORTED_X_DTYPES:
        raise TypeError(f"unsupported out dtype for fused Q4_K GEMM: {out.dtype}")

    packed = raw_weight if raw_weight.device == x.device else raw_weight.to(x.device)
    packed = packed.contiguous()
    x_in = x.contiguous()
    x_kernel = x_in
    if x_in.device.type == "cpu" and _truthy_env("TRITON_INTERPRET") and x_in.dtype != torch.float32:
        x_kernel = x_in.to(torch.float32)
    m_size = int(x_kernel.shape[0])
    block_m, block_n, num_warps, num_stages = _select_launch_config(m_size, n_size)
    out_kernel = out.contiguous()

    grid = (triton.cdiv(m_size, block_m), triton.cdiv(n_size, block_n))
    _q4k_inline_gemm_kernel[grid](
        x_kernel,
        packed,
        out_kernel,
        m_size,
        n_size,
        k_size,
        x_kernel.stride(0),
        x_kernel.stride(1),
        packed.stride(0),
        packed.stride(1),
        out_kernel.stride(0),
        out_kernel.stride(1),
        qk_k=QK_K,
        type_size=Q4_K_TYPE_SIZE,
        meta_bytes=Q4_K_META_BYTES,
        subblock=Q4_K_SUBBLOCK,
        qbytes_per_superblock=Q4_K_QBYTES_PER_SUPERBLOCK,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if out_kernel.data_ptr() != out.data_ptr():
        out.copy_(out_kernel)
    return out


def q4k_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if triton is None or not _tensor_is_q4k(weight):
        return None
    if x.ndim < 2 or x.dtype not in SUPPORTED_X_DTYPES:
        return None

    x2d = x.reshape(-1, x.shape[-1])
    if not _triton_runtime_available_for_device(x2d.device):
        return None

    raw_weight = getattr(weight, "data", weight)
    logical_shape_attr = getattr(weight, "tensor_shape", None)
    if logical_shape_attr is None:
        logical_shape_attr = weight.shape
    logical_shape = _logical_shape(logical_shape_attr)
    out = q4k_gemm(x2d, raw_weight, logical_shape, out_dtype=x.dtype)
    if bias is not None:
        out = out + bias.to(device=out.device, dtype=out.dtype)
    return out.reshape(*x.shape[:-1], logical_shape[0])


def reference_q4k_linear(
    x: torch.Tensor,
    raw_weight: torch.Tensor,
    logical_shape: Sequence[int] | torch.Size,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    weight = dequantize(raw_weight, Q4_K_QTYPE, logical_shape, dtype=x.dtype)
    return F.linear(x, weight, bias)


def load_gguf_tensor(
    gguf_path: str | os.PathLike[str],
    tensor_name: str,
) -> tuple[torch.Tensor, tuple[int, int]]:
    reader = gguf.GGUFReader(str(gguf_path))
    for tensor in reader.tensors:
        if tensor.name != tensor_name:
            continue
        logical_shape = tuple(int(v) for v in tensor.shape)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            raw = torch.from_numpy(tensor.data)
        return raw, _logical_shape(logical_shape)
    raise KeyError(f"tensor not found in GGUF: {tensor_name}")


def slice_q4k_weight(
    raw_weight: torch.Tensor,
    logical_shape: Sequence[int] | torch.Size,
    *,
    n_rows: int | None = None,
    k_cols: int | None = None,
) -> tuple[torch.Tensor, tuple[int, int]]:
    n, k = _logical_shape(logical_shape)
    n_rows = n if n_rows is None else int(n_rows)
    k_cols = k if k_cols is None else int(k_cols)
    if n_rows <= 0 or n_rows > n:
        raise ValueError(f"n_rows must be in [1, {n}], got {n_rows}")
    if k_cols <= 0 or k_cols > k or k_cols % QK_K != 0:
        raise ValueError(f"k_cols must be in [256, {k}] and multiple of {QK_K}, got {k_cols}")
    packed_cols = (k_cols // QK_K) * Q4_K_TYPE_SIZE
    return raw_weight[:n_rows, :packed_cols].contiguous(), (n_rows, k_cols)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _bench(fn, *, device: torch.device, warmup: int, iters: int) -> tuple[float, int | None]:
    for _ in range(warmup):
        fn()
    _sync(device)

    peak = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    _sync(device)
    elapsed = (time.perf_counter() - t0) / max(iters, 1)

    if device.type == "cuda":
        peak = int(torch.cuda.max_memory_allocated(device))
    return elapsed, peak


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True, help="Path to a GGUF file")
    parser.add_argument(
        "--tensor",
        default="blocks.0.self_attn.q.weight",
        help="Tensor name inside the GGUF file",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=17,
        help="Number of input rows to benchmark",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=5120,
        help="Optional row slice for correctness/benchmarking",
    )
    parser.add_argument(
        "--k-cols",
        type=int,
        default=5120,
        help="Optional K slice, must be a multiple of 256",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp16", "bf16", "fp32"),
        default="bf16",
        help="Activation dtype",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Target device for the benchmark",
    )
    args = parser.parse_args(argv)

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    raw_weight, logical_shape = load_gguf_tensor(args.gguf, args.tensor)
    raw_weight, logical_shape = slice_q4k_weight(
        raw_weight,
        logical_shape,
        n_rows=args.n_rows,
        k_cols=args.k_cols,
    )

    n_size, k_size = logical_shape
    x = torch.randn(args.m, k_size, dtype=dtype, device=device)
    bias = torch.randn(n_size, dtype=dtype, device=device)
    raw_weight_dev = raw_weight if device.type == "cpu" else raw_weight.to(device)

    ref = reference_q4k_linear(x, raw_weight_dev, logical_shape, bias)
    fused = q4k_gemm(x, raw_weight_dev, logical_shape, out_dtype=dtype) + bias

    diff = (fused - ref).to(torch.float32)
    max_abs = float(diff.abs().max().item())
    rms = float(diff.square().mean().sqrt().item())
    block_m, block_n, num_warps, num_stages = _select_launch_config(args.m, n_size)

    print(f"tensor: {args.tensor}")
    print(f"logical_shape: {logical_shape}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(
        "launch_config: "
        f"block_m={block_m} block_n={block_n} block_k={QK_K} "
        f"num_warps={num_warps} num_stages={num_stages}"
    )
    print(f"max_abs_diff: {max_abs:.8f}")
    print(f"rms_diff: {rms:.8f}")

    if device.type != "cuda":
        print("cuda_benchmark: skipped (CUDA not available on this machine)")
        return 0

    ref_ms, ref_peak = _bench(
        lambda: reference_q4k_linear(x, raw_weight_dev, logical_shape, bias),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    fused_ms, fused_peak = _bench(
        lambda: q4k_gemm(x, raw_weight_dev, logical_shape, out_dtype=dtype) + bias,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    print(f"baseline_ms: {ref_ms * 1000.0:.3f}")
    print(f"baseline_peak_bytes: {ref_peak}")
    print(f"fused_ms: {fused_ms * 1000.0:.3f}")
    print(f"fused_peak_bytes: {fused_peak}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
