"""Fused per-token LayerNorm + AdaLN-Zero modulation Triton kernel for Wan2.2.

The Wan2.2 transformer block computes, per attention/FFN modulation site:

    out[b, l, c] = LayerNorm(x[b, l, :])[c] * (1 + scale[b, l, c]) + shift[b, l, c]

Where:
- x has shape (B, L, C) — Wan inference is B=1, L=18000 (p720_17f), C=5120
- scale and shift have shape (B, L, C) — PER-TOKEN modulation (not per-batch)

Stock eager path runs 3-4 separate kernels: LayerNorm, broadcast add (1+scale),
broadcast mul, broadcast add shift. Each pass touches the full (B, L, C) tensor
(~150 MB at p720_17f). This kernel fuses all four ops into a single pass.

The standard F.layer_norm fusion (passing weight=(1+scale), bias=shift) does
NOT work here because F.layer_norm requires weight/bias of shape (C,) — but
Wan's modulation is per-token (B, L, C). See the FFFm attempt memo for the
exact RuntimeError. This Triton kernel handles the per-token broadcast
correctly via explicit row indexing.

Each program processes one (b, l) row. Reads x_row, scale_row, shift_row of
length C. Computes mean/var/normalize/mul/add in fp32 internally for
numerical stability (matches PyTorch F.layer_norm semantics). Writes the
output row in the input dtype.

Gate: BASIWAN_FUSED_NORM_MOD=1 (opt-in). Default off until the wall and
quality A/B confirms the win.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAVE_TRITON = True
except Exception:
    _HAVE_TRITON = False


if _HAVE_TRITON:

    @triton.jit
    def _fused_ln_mod_kernel(
        x_ptr,              # (B*L, C) input
        scale_ptr,          # (B*L, C) per-token scale
        shift_ptr,          # (B*L, C) per-token shift
        out_ptr,            # (B*L, C) output
        C,                  # int — channel count
        stride_x_row,
        stride_scale_row,
        stride_shift_row,
        stride_out_row,
        eps,
        BLOCK_C: tl.constexpr,
    ):
        # Each program handles one (b, l) row.
        row_id = tl.program_id(0)
        cols = tl.arange(0, BLOCK_C)
        mask = cols < C

        x_offs = x_ptr + row_id * stride_x_row + cols
        x_row = tl.load(x_offs, mask=mask, other=0.0).to(tl.float32)

        # LayerNorm statistics in fp32.
        x_for_stat = tl.where(mask, x_row, 0.0)
        mean = tl.sum(x_for_stat, axis=0) / C
        diff = tl.where(mask, x_row - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        norm = diff * rstd

        scale = tl.load(scale_ptr + row_id * stride_scale_row + cols, mask=mask, other=0.0).to(tl.float32)
        shift = tl.load(shift_ptr + row_id * stride_shift_row + cols, mask=mask, other=0.0).to(tl.float32)
        out = norm * (1.0 + scale) + shift

        tl.store(out_ptr + row_id * stride_out_row + cols, out, mask=mask)


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def fused_layernorm_modulation(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute LayerNorm(x) * (1 + scale) + shift in a single fused kernel.

    All three tensors must have the same shape (B, L, C) and dtype (bf16/fp16/fp32).
    Returns a fresh tensor (same shape, same dtype). The kernel runs the LN
    statistics in fp32 internally — matching torch.nn.functional.layer_norm
    semantics — and casts the output back to the input dtype.

    Restrictions:
    - Last dim C must fit in BLOCK_C (next pow-2 of C, max 16384).
    - All inputs must be CUDA contiguous in the last dim.
    """
    if not _HAVE_TRITON:
        raise RuntimeError(
            "fused_layernorm_modulation requires triton; install triton or "
            "fall back to the eager LayerNorm + modulation path"
        )
    if x.dim() != 3 or scale.shape != x.shape or shift.shape != x.shape:
        raise ValueError(
            f"fused_layernorm_modulation: expected x/scale/shift to be (B,L,C) "
            f"with matching shape; got x={tuple(x.shape)} scale={tuple(scale.shape)} shift={tuple(shift.shape)}"
        )
    if x.dtype != scale.dtype or x.dtype != shift.dtype:
        raise ValueError(
            f"fused_layernorm_modulation: x/scale/shift dtype must match; "
            f"got x={x.dtype} scale={scale.dtype} shift={shift.dtype}"
        )
    if not x.is_cuda:
        raise ValueError("fused_layernorm_modulation: x must be on CUDA")

    B, L, C = x.shape
    BLOCK_C = _next_pow2(C)
    if BLOCK_C > 16384:
        raise ValueError(
            f"fused_layernorm_modulation: C={C} too large; BLOCK_C={BLOCK_C} exceeds 16384"
        )

    x_2d = x.reshape(B * L, C)
    scale_2d = scale.reshape(B * L, C)
    shift_2d = shift.reshape(B * L, C)

    # Output: fresh tensor, same dtype as input.
    out_2d = torch.empty_like(x_2d)

    # Stride along the row dim (number of elements per row).
    stride_x_row = x_2d.stride(0)
    stride_scale_row = scale_2d.stride(0)
    stride_shift_row = shift_2d.stride(0)
    stride_out_row = out_2d.stride(0)

    grid = (B * L,)
    _fused_ln_mod_kernel[grid](
        x_2d, scale_2d, shift_2d, out_2d,
        C,
        stride_x_row, stride_scale_row, stride_shift_row, stride_out_row,
        eps,
        BLOCK_C=BLOCK_C,
    )
    return out_2d.reshape(B, L, C)


def selftest():
    """Synthetic correctness test against the eager LN + modulation path.

    Returns (max_abs_error, passed).
    """
    import os as _os
    if not torch.cuda.is_available() or not _HAVE_TRITON:
        print("[fused-norm-mod selftest] skipping (no CUDA or no triton)")
        return None, False
    device = "cuda:0"
    torch.manual_seed(0)
    B, L, C = 1, 1024, 5120
    eps = 1e-6
    # Tolerances calibrated to bf16/fp16 rounding noise. Triton sum reduction
    # order differs from torch's, producing up to 1-2 ULP at the cast boundary.
    # bf16: mantissa step ~0.5%; for output magnitudes ~3, abs error ~1.5e-2 is at noise floor.
    # fp16: ~1e-3 abs at typical magnitudes.
    for dtype, atol in [(torch.bfloat16, 2.5e-2), (torch.float16, 4e-3)]:
        x = torch.randn(B, L, C, device=device, dtype=dtype) * 0.5
        scale = torch.randn(B, L, C, device=device, dtype=dtype) * 0.1
        shift = torch.randn(B, L, C, device=device, dtype=dtype) * 0.1

        # Reference: fp32 internal everything, single cast at end (matches the
        # Triton kernel's behavior). NOTE: this is NOT a bit-perfect match for the
        # eager Wan path, which does 3 bf16 quantizations and produces slightly
        # different outputs. See audit_FFFw_*_2026-06-06.md for the e2e finding.
        x_fp32 = x.float()
        mean = x_fp32.mean(dim=-1, keepdim=True)
        var = x_fp32.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x_fp32 - mean) / torch.sqrt(var + eps)
        ref = x_norm * (1 + scale.float()) + shift.float()
        ref = ref.to(dtype)

        out = fused_layernorm_modulation(x, scale, shift, eps=eps)
        max_abs = (out.float() - ref.float()).abs().max().item()
        passed = max_abs < atol
        print(
            f"[fused-norm-mod selftest] {dtype} B={B} L={L} C={C}: "
            f"max_abs={max_abs:.4e} tol={atol} {'OK' if passed else 'FAIL'}"
        )
    return max_abs, passed


if __name__ == "__main__":
    selftest()
