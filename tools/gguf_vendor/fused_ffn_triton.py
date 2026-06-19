"""Chunk-local fused Q4_K FFN path for Wan GGUF inference.

This is the low-risk memory fix for long-sequence Wan FFNs:

1. Run Q4_K `linear1` into a reusable scratch buffer.
2. Apply GELU(approximate="tanh") in place.
3. Run Q4_K `linear2` into a reusable output buffer.
4. Apply forward-time LoRA in tiled slices so no full hidden/output-sized
   LoRA delta is ever materialized.

The design goal here is allocator behavior, not raw speed. We remove the
full `(M, 13824)` `linear1` transient that fragments a 24 GB Ada card at
`p720_33f`, while keeping the existing Q4_K inline-dequant Triton matmul as
the core compute primitive.
"""
from __future__ import annotations

import argparse
import time
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .dequant import dequantize_tensor, is_quantized
    from .q4k_triton_gemm import (
        SUPPORTED_X_DTYPES,
        _logical_shape,
        _tensor_is_q4k,
        _triton_runtime_available_for_device,
        load_gguf_tensor,
        q4k_gemm_out,
        reference_q4k_linear,
        triton,
        tl,
    )
except ImportError:  # pragma: no cover - supports direct script execution
    from dequant import dequantize_tensor, is_quantized
    from q4k_triton_gemm import (
        SUPPORTED_X_DTYPES,
        _logical_shape,
        _tensor_is_q4k,
        _triton_runtime_available_for_device,
        load_gguf_tensor,
        q4k_gemm_out,
        reference_q4k_linear,
        triton,
        tl,
    )


DEFAULT_SCRATCH_LIMIT_BYTES = 96 * 1024 * 1024
DEFAULT_LORA_TILE_COLS = 1024
GELU_TANH_COEFF = 0.7978845608028654  # sqrt(2 / pi)
GELU_TANH_CUBIC = 0.044715


if triton is not None:

    @triton.jit
    def _gelu_tanh_inplace_kernel(
        x_ptr,
        n_elements,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * block_size + tl.arange(0, block_size)
        mask = offs < n_elements
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x3 = x * x * x
        y = 0.5 * x * (1.0 + tl.tanh(GELU_TANH_COEFF * (x + GELU_TANH_CUBIC * x3)))
        tl.store(x_ptr + offs, y.to(x_ptr.type.element_ty), mask=mask)


def _prepare_bias(bias: torch.Tensor | None, ref: torch.Tensor) -> torch.Tensor | None:
    if bias is None:
        return None
    if is_quantized(bias):
        bias = dequantize_tensor(bias, dtype=ref.dtype)
    elif bias.dtype != ref.dtype:
        bias = bias.to(ref.dtype)
    if bias.device != ref.device:
        bias = bias.to(ref.device)
    return bias


def _extract_q4k_weight(weight: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if isinstance(weight, tuple) and len(weight) == 2:
        raw_weight, logical_shape = weight
        if not isinstance(raw_weight, torch.Tensor):
            raise TypeError(f"expected tuple[Tensor, logical_shape], got {type(raw_weight)!r}")
        return raw_weight, _logical_shape(logical_shape)
    raw_weight = getattr(weight, "data", weight)
    logical_shape_attr = getattr(weight, "tensor_shape", None)
    if logical_shape_attr is None:
        logical_shape_attr = weight.shape
    return raw_weight, _logical_shape(logical_shape_attr)


def _weight_is_supported_q4k(weight: torch.Tensor | tuple[torch.Tensor, Sequence[int]]) -> bool:
    if isinstance(weight, tuple) and len(weight) == 2 and isinstance(weight[0], torch.Tensor):
        return True
    return _tensor_is_q4k(weight)


def max_chunk_rows_for_hidden(
    hidden_dim: int,
    dtype: torch.dtype,
    *,
    max_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
) -> int:
    bytes_per_elem = torch.tensor([], dtype=dtype).element_size()
    if hidden_dim <= 0 or bytes_per_elem <= 0:
        raise ValueError(f"invalid hidden scratch shape: hidden_dim={hidden_dim} dtype={dtype}")
    return max(1, int(max_bytes // (hidden_dim * bytes_per_elem)))


def _ensure_2d_buffer(
    buf: torch.Tensor | None,
    rows: int,
    cols: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    expected = (rows, cols)
    if (
        buf is None
        or tuple(buf.shape) != expected
        or buf.dtype != dtype
        or buf.device != device
        or not buf.is_contiguous()
    ):
        return torch.empty(expected, device=device, dtype=dtype)
    return buf


def _gelu_tanh_inplace(x: torch.Tensor) -> None:
    if triton is None or not _triton_runtime_available_for_device(x.device):
        x.copy_(F.gelu(x, approximate="tanh"))
        return

    flat = x.reshape(-1)
    block_size = 1024
    grid = (triton.cdiv(flat.numel(), block_size),)
    _gelu_tanh_inplace_kernel[grid](flat, flat.numel(), block_size=block_size, num_warps=4)


def _apply_tiled_lora_delta_(
    out: torch.Tensor,
    source: torch.Tensor,
    down: torch.Tensor | None,
    up: torch.Tensor | None,
    scale: float,
    *,
    tile_cols: int = DEFAULT_LORA_TILE_COLS,
) -> None:
    if down is None or up is None or scale == 0.0:
        return

    if down.device != source.device or down.dtype != torch.float32:
        down = down.to(device=source.device, dtype=torch.float32)
    if up.device != source.device or up.dtype != torch.float32:
        up = up.to(device=source.device, dtype=torch.float32)

    src_fp32 = source.to(torch.float32)
    proj = src_fp32 @ down.T
    if scale != 1.0:
        proj.mul_(float(scale))

    out_cols = out.shape[1]
    for start in range(0, out_cols, tile_cols):
        end = min(start + tile_cols, out_cols)
        delta = proj @ up[start:end].T
        out[:, start:end].add_(delta.to(out.dtype))


def reference_q4_ffn(
    x: torch.Tensor,
    w1_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    *,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    lora1_down: torch.Tensor | None = None,
    lora1_up: torch.Tensor | None = None,
    lora1_scale: float = 0.0,
    lora2_down: torch.Tensor | None = None,
    lora2_up: torch.Tensor | None = None,
    lora2_scale: float = 0.0,
    lora_tile_cols: int = DEFAULT_LORA_TILE_COLS,
) -> torch.Tensor:
    x2d = x.reshape(-1, x.shape[-1])
    raw_w1, logical_shape1 = _extract_q4k_weight(w1_packed)
    raw_w2, logical_shape2 = _extract_q4k_weight(w2_packed)
    raw_w1 = raw_w1 if raw_w1.device == x2d.device else raw_w1.to(x2d.device)
    raw_w2 = raw_w2 if raw_w2.device == x2d.device else raw_w2.to(x2d.device)

    hidden = reference_q4k_linear(x2d, raw_w1, logical_shape1, _prepare_bias(bias1, x2d))
    _apply_tiled_lora_delta_(
        hidden,
        x2d,
        lora1_down,
        lora1_up,
        lora1_scale,
        tile_cols=lora_tile_cols,
    )
    hidden = F.gelu(hidden, approximate="tanh")
    out = reference_q4k_linear(hidden, raw_w2, logical_shape2, _prepare_bias(bias2, hidden))
    _apply_tiled_lora_delta_(
        out,
        hidden,
        lora2_down,
        lora2_up,
        lora2_scale,
        tile_cols=lora_tile_cols,
    )
    return out.reshape(*x.shape[:-1], logical_shape2[0])


def fused_q4_ffn(
    x: torch.Tensor,
    w1_packed: torch.Tensor,
    w2_packed: torch.Tensor,
    *,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    lora1_down: torch.Tensor | None = None,
    lora1_up: torch.Tensor | None = None,
    lora1_scale: float = 0.0,
    lora2_down: torch.Tensor | None = None,
    lora2_up: torch.Tensor | None = None,
    lora2_scale: float = 0.0,
    chunk_size: int = 4096,
    out: torch.Tensor | None = None,
    scratch: torch.Tensor | None = None,
    lora_tile_cols: int = DEFAULT_LORA_TILE_COLS,
    max_scratch_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
) -> torch.Tensor | None:
    if triton is None:
        return None
    if x.ndim < 2 or x.dtype not in SUPPORTED_X_DTYPES:
        return None
    if not _weight_is_supported_q4k(w1_packed) or not _weight_is_supported_q4k(w2_packed):
        return None

    x2d = x.reshape(-1, x.shape[-1])
    if not _triton_runtime_available_for_device(x2d.device):
        return None

    raw_w1, logical_shape1 = _extract_q4k_weight(w1_packed)
    raw_w2, logical_shape2 = _extract_q4k_weight(w2_packed)
    hidden_dim = logical_shape1[0]
    out_dim = logical_shape2[0]
    rows = int(x2d.shape[0])
    if logical_shape1[1] != int(x2d.shape[1]):
        raise ValueError(f"linear1 input mismatch: expected {logical_shape1[1]}, got {x2d.shape[1]}")
    if logical_shape2[1] != hidden_dim:
        raise ValueError(f"linear2 input mismatch: expected {hidden_dim}, got {logical_shape2[1]}")

    capped_chunk = max_chunk_rows_for_hidden(hidden_dim, x2d.dtype, max_bytes=max_scratch_bytes)
    effective_chunk = max(1, min(int(chunk_size), rows, capped_chunk))

    out2d = _ensure_2d_buffer(out, rows, out_dim, dtype=x2d.dtype, device=x2d.device)
    scratch2d = _ensure_2d_buffer(
        scratch,
        effective_chunk,
        hidden_dim,
        dtype=x2d.dtype,
        device=x2d.device,
    )

    bias1_ready = _prepare_bias(bias1, x2d)
    bias2_ready = _prepare_bias(bias2, x2d)

    for start in range(0, rows, effective_chunk):
        end = min(start + effective_chunk, rows)
        x_chunk = x2d[start:end]
        hidden = scratch2d[: end - start]
        q4k_gemm_out(x_chunk, raw_w1, logical_shape1, out=hidden)
        if bias1_ready is not None:
            hidden.add_(bias1_ready)
        _apply_tiled_lora_delta_(
            hidden,
            x_chunk,
            lora1_down,
            lora1_up,
            lora1_scale,
            tile_cols=lora_tile_cols,
        )
        _gelu_tanh_inplace(hidden)

        out_chunk = out2d[start:end]
        q4k_gemm_out(hidden, raw_w2, logical_shape2, out=out_chunk)
        if bias2_ready is not None:
            out_chunk.add_(bias2_ready)
        _apply_tiled_lora_delta_(
            out_chunk,
            hidden,
            lora2_down,
            lora2_up,
            lora2_scale,
            tile_cols=lora_tile_cols,
        )

    return out2d.reshape(*x.shape[:-1], out_dim)


class FusedChunkedQ4FFN(nn.Module):
    """Drop-in FFN wrapper for Wan GGUF blocks."""

    def __init__(
        self,
        orig: nn.Sequential,
        *,
        chunk_size: int,
        threshold: int = 8000,
        max_scratch_bytes: int = DEFAULT_SCRATCH_LIMIT_BYTES,
        lora_tile_cols: int = DEFAULT_LORA_TILE_COLS,
    ):
        super().__init__()
        self.orig = orig
        self.chunk_size = int(chunk_size)
        self.threshold = int(threshold)
        self.max_scratch_bytes = int(max_scratch_bytes)
        self.lora_tile_cols = int(lora_tile_cols)
        self._out_buf = None
        self._hidden_buf = None
        self._fallback_out_buf = None

    def _fallback_chunked(self, x: torch.Tensor) -> torch.Tensor:
        rows = int(x.reshape(-1, x.shape[-1]).shape[0])
        cols = int(x.shape[-1])
        out2d = _ensure_2d_buffer(
            self._fallback_out_buf,
            rows,
            cols,
            dtype=x.dtype,
            device=x.device,
        )
        self._fallback_out_buf = out2d
        out = out2d.reshape_as(x)
        n = x.shape[-2]
        for start in range(0, n, self.chunk_size):
            end = min(start + self.chunk_size, n)
            ch = x[..., start:end, :]
            for layer in self.orig:
                ch = layer(ch)
            out[..., start:end, :].copy_(ch)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2 or x.shape[-2] <= self.threshold:
            return self.orig(x)

        if (
            not isinstance(self.orig, nn.Sequential)
            or len(self.orig) != 3
            or not isinstance(self.orig[1], nn.GELU)
        ):
            return self._fallback_chunked(x)

        linear1 = self.orig[0]
        linear2 = self.orig[2]
        if (
            not hasattr(linear1, "weight")
            or not hasattr(linear2, "weight")
            or not _tensor_is_q4k(linear1.weight)
            or not _tensor_is_q4k(linear2.weight)
        ):
            return self._fallback_chunked(x)

        x2d = x.reshape(-1, x.shape[-1])
        hidden_dim = int(getattr(linear1, "out_features"))
        out_dim = int(getattr(linear2, "out_features"))
        effective_chunk = max(
            1,
            min(
                self.chunk_size,
                x2d.shape[0],
                max_chunk_rows_for_hidden(
                    hidden_dim,
                    x2d.dtype,
                    max_bytes=self.max_scratch_bytes,
                ),
            ),
        )
        self._out_buf = _ensure_2d_buffer(
            self._out_buf,
            int(x2d.shape[0]),
            out_dim,
            dtype=x.dtype,
            device=x.device,
        )
        self._hidden_buf = _ensure_2d_buffer(
            self._hidden_buf,
            effective_chunk,
            hidden_dim,
            dtype=x.dtype,
            device=x.device,
        )

        out = fused_q4_ffn(
            x,
            linear1.weight,
            linear2.weight,
            bias1=getattr(linear1, "bias", None),
            bias2=getattr(linear2, "bias", None),
            lora1_down=getattr(linear1, "lora_down", None),
            lora1_up=getattr(linear1, "lora_up", None),
            lora1_scale=float(getattr(linear1, "lora_scale", 0.0)),
            lora2_down=getattr(linear2, "lora_down", None),
            lora2_up=getattr(linear2, "lora_up", None),
            lora2_scale=float(getattr(linear2, "lora_scale", 0.0)),
            chunk_size=effective_chunk,
            out=self._out_buf,
            scratch=self._hidden_buf,
            lora_tile_cols=self.lora_tile_cols,
            max_scratch_bytes=self.max_scratch_bytes,
        )
        if out is None:
            return self._fallback_chunked(x)
        return out


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
    parser.add_argument("--tensor-w1", default="blocks.0.ffn.0.weight")
    parser.add_argument("--tensor-w2", default="blocks.0.ffn.2.weight")
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="bf16")
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--lora-rank", type=int, default=0)
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

    w1_raw, logical_shape1 = load_gguf_tensor(args.gguf, args.tensor_w1)
    w2_raw, logical_shape2 = load_gguf_tensor(args.gguf, args.tensor_w2)
    if logical_shape2[1] != logical_shape1[0]:
        raise ValueError(
            f"FFN shape mismatch: linear1 out={logical_shape1[0]} linear2 in={logical_shape2[1]}"
        )

    x = torch.randn(args.m, logical_shape1[1], dtype=dtype, device=device)
    bias1 = torch.randn(logical_shape1[0], dtype=dtype, device=device)
    bias2 = torch.randn(logical_shape2[0], dtype=dtype, device=device)
    w1_raw = w1_raw if device.type == "cpu" else w1_raw.to(device)
    w2_raw = w2_raw if device.type == "cpu" else w2_raw.to(device)
    w1 = (w1_raw, logical_shape1)
    w2 = (w2_raw, logical_shape2)

    lora1_down = lora1_up = lora2_down = lora2_up = None
    if args.lora_rank > 0:
        rank = int(args.lora_rank)
        lora1_down = torch.randn(rank, logical_shape1[1], dtype=torch.float32, device=device)
        lora1_up = torch.randn(logical_shape1[0], rank, dtype=torch.float32, device=device)
        lora2_down = torch.randn(rank, logical_shape2[1], dtype=torch.float32, device=device)
        lora2_up = torch.randn(logical_shape2[0], rank, dtype=torch.float32, device=device)

    ref = reference_q4_ffn(
        x,
        w1,
        w2,
        bias1=bias1,
        bias2=bias2,
        lora1_down=lora1_down,
        lora1_up=lora1_up,
        lora1_scale=1.0 if lora1_down is not None else 0.0,
        lora2_down=lora2_down,
        lora2_up=lora2_up,
        lora2_scale=1.0 if lora2_down is not None else 0.0,
    )
    fused = fused_q4_ffn(
        x,
        w1,
        w2,
        bias1=bias1,
        bias2=bias2,
        lora1_down=lora1_down,
        lora1_up=lora1_up,
        lora1_scale=1.0 if lora1_down is not None else 0.0,
        lora2_down=lora2_down,
        lora2_up=lora2_up,
        lora2_scale=1.0 if lora2_down is not None else 0.0,
        chunk_size=args.chunk_size,
    )
    if fused is None:
        print("fused_q4_ffn: unavailable on this machine/device")
        return 1

    diff = (fused - ref).to(torch.float32)
    max_abs = float(diff.abs().max().item())
    rms = float(diff.square().mean().sqrt().item())
    scratch_rows = min(
        args.chunk_size,
        args.m,
        max_chunk_rows_for_hidden(logical_shape1[0], dtype),
    )
    scratch_mb = scratch_rows * logical_shape1[0] * torch.tensor([], dtype=dtype).element_size() / (1024**2)

    print(f"tensor_w1: {args.tensor_w1}")
    print(f"tensor_w2: {args.tensor_w2}")
    print(f"logical_shape1: {logical_shape1}")
    print(f"logical_shape2: {logical_shape2}")
    print(f"device: {device}")
    print(f"dtype: {dtype}")
    print(f"effective_chunk_rows: {scratch_rows}")
    print(f"hidden_scratch_mb: {scratch_mb:.2f}")
    print(f"max_abs_diff: {max_abs:.8f}")
    print(f"rms_diff: {rms:.8f}")

    if device.type != "cuda":
        print("cuda_benchmark: skipped (CUDA not available on this machine)")
        return 0

    fused_elapsed, fused_peak = _bench(
        lambda: fused_q4_ffn(
            x,
            w1,
            w2,
            bias1=bias1,
            bias2=bias2,
            lora1_down=lora1_down,
            lora1_up=lora1_up,
            lora1_scale=1.0 if lora1_down is not None else 0.0,
            lora2_down=lora2_down,
            lora2_up=lora2_up,
            lora2_scale=1.0 if lora2_down is not None else 0.0,
            chunk_size=args.chunk_size,
        ),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    ref_elapsed, ref_peak = _bench(
        lambda: reference_q4_ffn(
            x,
            w1,
            w2,
            bias1=bias1,
            bias2=bias2,
            lora1_down=lora1_down,
            lora1_up=lora1_up,
            lora1_scale=1.0 if lora1_down is not None else 0.0,
            lora2_down=lora2_down,
            lora2_up=lora2_up,
            lora2_scale=1.0 if lora2_down is not None else 0.0,
        ),
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(f"fused_ms: {fused_elapsed * 1e3:.3f}")
    print(f"ref_ms: {ref_elapsed * 1e3:.3f}")
    print(f"fused_peak_bytes: {fused_peak}")
    print(f"ref_peak_bytes: {ref_peak}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(run_cli())
