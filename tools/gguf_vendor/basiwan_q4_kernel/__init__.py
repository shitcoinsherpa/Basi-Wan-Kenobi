"""Native Marlin-pattern Q4_K + Q6_K matmul for GGUF on Ada (sm_89).

Public entry points (used by bare_gguf.py):
  pack_basiwan_weight(weight) → PackedBasiwanWeight
  basiwan_q4k_linear(x, packed, *, bias, runtime) → out
  BasiwanRuntimeBuffers — per-Linear persistent scratch (output, etc.)

No silent fallbacks. The kernel either runs and is correct, or we raise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from .pack import (
    PackedBasiwanWeight,
    QuantType,
    pack_basiwan_weight,
)

_HERE = Path(__file__).resolve().parent
_EXT = None  # lazy-built torch C++ extension


def _load_extension():
    """JIT-build the CUDA extension on first call. Cached afterwards."""
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Marlin Q4_K/Q6_K kernel requires CUDA; refusing CPU fallback "
            "(no shitty fallbacks per feedback_no_lazy_impl.md)"
        )
    from torch.utils.cpp_extension import load

    # Build name encodes the regcount so different builds don't clobber each
    # other's .so. Override via BASIWAN_MAXRREG (default 128 = 2 CTAs/SM
    # at BLOCK_M=128 / 256 thr / 32K regs each on Ada's 64K-reg SM file).
    import os as _os
    maxrreg = int(_os.environ.get("BASIWAN_MAXRREG", "128"))
    # 2026-06-04 drift-bisect: set BASIWAN_NO_FAST_MATH=1 to build a
    # separate .so without --use_fast_math (different cache key = fresh build,
    # original .so untouched). Tests whether fast-math nvcc optimizations are
    # the source of the today-vs-ship composite Q drift.
    _no_fast_math = _os.environ.get("BASIWAN_NO_FAST_MATH") == "1"
    suffix = "_nofm" if _no_fast_math else ""
    name = f"wan_q4k_q6k_basiwan_ext_r{maxrreg}{suffix}"
    cflags = [
        "-O3",
        "-std=c++17",
        "-gencode=arch=compute_89,code=sm_89",
        f"--maxrregcount={maxrreg}",
    ]
    if not _no_fast_math:
        cflags.insert(2, "--use_fast_math")
    # 2026-06-08: Windows VS 2022 BuildTools (MSVC 14.44+) is newer than
    # CUDA 12.2's host_config.h whitelist. The compiler is ABI-compatible
    # in practice — bypass the version check on Windows.
    if _os.name == "nt":
        cflags.append("-allow-unsupported-compiler")
    # 2026-06-08: torch.utils.cpp_extension.load() re-validates the FULL
    # build chain (ninja, cl, nvcc) on every call, even when the cached
    # .pyd is already on disk. On Windows that requires a vcvarsall'd
    # shell which isn't typical. Skip the validation by direct-loading
    # the cached .pyd via importlib if it exists.
    if _os.name == "nt":
        # Match torch.utils.cpp_extension's default cache layout.
        _ext_root = Path(
            _os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                Path(_os.environ.get("LOCALAPPDATA", "")) / "torch_extensions" / "torch_extensions" / "Cache",
            )
        )
        _torch_ver = torch.__version__.split("+")[0].replace(".", "")
        _cuda_ver = (torch.version.cuda or "").replace(".", "")
        _pyd_dir = _ext_root / f"py{sys.version_info.major}{sys.version_info.minor}_cu{_cuda_ver}" / name
        _pyd_path = _pyd_dir / f"{name}.pyd"
        if _pyd_path.exists():
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(name, str(_pyd_path))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _EXT = _mod
            return _EXT
    _EXT = load(
        name=name,
        sources=[str(_HERE / "q4k_q6k_basiwan.cu")],
        extra_include_paths=[str(_HERE)],
        extra_cuda_cflags=cflags,
        verbose=True,
    )
    return _EXT


@dataclass
class BasiwanRuntimeBuffers:
    """Per-Linear persistent scratch — DEPRECATED in favor of the shared
    `_MARLIN_RUNTIME_POOL` (below) but kept for backward-compatibility of any
    callers passing `runtime=BasiwanRuntimeBuffers()` explicitly to
    `basiwan_q4k_linear`. New code should pass `runtime=None` and let the
    shared pool handle output buffer reuse across Linears.

    Rationale for the shared pool (2026-06-02): at p720_17f (M=18000) each
    per-Linear out buffer is up to 487 MB (FFN[0]). With block-swap cycling
    72 unique Linears per step, the sticky per-Linear pile reaches ~18 GB
    inside one diffusion step — leaving < 6 GB for everything else and
    OOMing at the start of step 2 (`time_projection.F.linear`). Sharing
    by (M_round, N, dtype, device) collapses to ~10 unique shapes →
    ~5 GB total.
    """
    out: Optional[torch.Tensor] = None
    max_m: int = 0
    n: int = 0
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float16

    def ensure(self, m: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Return a (M, N) buffer of the requested dtype/device. Re-allocate
        only when needed; otherwise slice the existing one."""
        if (
            self.out is None
            or self.device != device
            or self.dtype != dtype
            or self.n != n
            or self.max_m < m
        ):
            new_max_m = ((m + 63) // 64) * 64
            new_max_m = max(new_max_m, self.max_m)
            self.out = torch.empty((new_max_m, n), device=device, dtype=dtype)
            self.max_m = new_max_m
            self.n = n
            self.device = device
            self.dtype = dtype
        return self.out[:m]


# ---------------------------------------------------------------------------
# Shared runtime pool — one buffer per (M_round, N, dtype, device) tuple.
# Replaces the per-Linear sticky allocation that piled up to ~18 GB at p720_17f.
# ---------------------------------------------------------------------------

class _MarlinRuntimePool:
    """Process-wide pool of Marlin output buffers, keyed by (M_round, N, dtype,
    device). All BareGGMLLinears that share the same output shape share one
    backing tensor — the buffer is only ever in use during a single
    `basiwan_q4k_linear` call (a kernel launch + epilogue), so the same
    physical tensor can be slice-viewed across many Linears as long as no two
    Marlin calls execute concurrently (which they don't in our pipeline:
    forward is serialized at the Python level).

    Risk if violated: two concurrent Marlin calls overlapping on the same
    buffer would race. We DO NOT support that. If you ever batch-launch
    Marlin in parallel, the pool must be augmented to allocate a per-stream
    or per-caller buffer.
    """
    def __init__(self):
        # key: (m_round, n, dtype.str, device.str) → torch.Tensor (m_round, n)
        self._pool: dict = {}

    def get(self, m: int, n: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        m_round = ((m + 63) // 64) * 64
        key = (m_round, n, str(dtype), str(device))
        t = self._pool.get(key)
        if t is None:
            t = torch.empty((m_round, n), device=device, dtype=dtype)
            self._pool[key] = t
        return t[:m]

    def clear(self) -> None:
        """Free all pool tensors. Call between major shapes (e.g. between
        p480 and p720 benches) if you want a clean slate."""
        self._pool.clear()


_MARLIN_RUNTIME_POOL = _MarlinRuntimePool()


def get_marlin_runtime_pool() -> _MarlinRuntimePool:
    """Expose the singleton for callers that want to clear it between benches."""
    return _MARLIN_RUNTIME_POOL


class _MarlinInputPool:
    """Process-wide pool of M-padded input buffers, keyed by (M_padded, K,
    dtype, device). Mirrors `_MarlinRuntimePool` but for the kernel INPUT side:
    Wan production has non-64-aligned M (7800, 18000, 32400) so every Marlin
    call allocates a fresh padded buffer at `basiwan_q4k_linear` entry. At
    p720_33f attention sites that's 332 MB (M=32448, K=5120 bf16) per call,
    × 4 sites × 80 blocks × 2 experts per step — a major fragmentation source
    that broke p720_33f at the boundary of step 3 (runs #4 and #5,
    2026-06-02). Pooling collapses ~10 unique (M_padded, K) shapes across the
    whole model.

    Correctness: the pool yields a buffer of shape (M_padded, K). Caller
    writes x2d into rows [:M] and explicitly zeros rows [M:M_padded] on every
    call so the kernel's dot products over those rows are zero (kernel
    requires M%64==0).
    """
    def __init__(self):
        self._pool: dict = {}

    def get(self, m_padded: int, k: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (m_padded, k, str(dtype), str(device))
        t = self._pool.get(key)
        if t is None:
            # zeros so very first use has zero pad rows; subsequent calls
            # explicitly re-zero pad rows.
            t = torch.zeros((m_padded, k), device=device, dtype=dtype)
            self._pool[key] = t
        return t

    def clear(self) -> None:
        self._pool.clear()


_MARLIN_INPUT_POOL = _MarlinInputPool()


def get_marlin_input_pool() -> _MarlinInputPool:
    return _MARLIN_INPUT_POOL


def _move_packed_to_device(packed: PackedBasiwanWeight, device: torch.device, dtype: torch.dtype) -> PackedBasiwanWeight:
    """Ensure all tensors in PackedBasiwanWeight are on the right device + scale dtype matches.

    Mutates `packed` in place — fine because the runtime caches the packed
    representation per-Linear and we want device transitions to be sticky.
    """
    if packed.weight.device != device:
        packed.weight = packed.weight.to(device)
    if packed.weight_hi is not None and packed.weight_hi.device != device:
        packed.weight_hi = packed.weight_hi.to(device)
    if packed.scales.device != device or packed.scales.dtype != dtype:
        packed.scales = packed.scales.to(device=device, dtype=dtype)
    if packed.mins is not None and (packed.mins.device != device or packed.mins.dtype != dtype):
        packed.mins = packed.mins.to(device=device, dtype=dtype)
    return packed


def basiwan_q4k_linear(
    x: torch.Tensor,
    packed: PackedBasiwanWeight,
    *,
    bias: Optional[torch.Tensor] = None,
    runtime: Optional[BasiwanRuntimeBuffers] = None,
) -> torch.Tensor:
    """Matmul x @ dequant(packed_weight).T + bias, using the native CUDA kernel.

    x:        (M, K) fp16 or bf16
    packed:   PackedBasiwanWeight from pack_basiwan_weight(weight)
    bias:     optional (N,) — coerced to x.dtype if needed
    runtime:  BasiwanRuntimeBuffers — if None, a fresh one is allocated per call (slow)

    Returns: (..., N) tensor with the same dtype as x. The kernel supports
    both fp16 and bf16 via native mma.sync paths (m16n8k16.row.col.f32.{f16|bf16}.{f16|bf16}.f32);
    no silent dtype conversion is performed.
    """
    if not x.is_cuda:
        raise RuntimeError("BASIWAN path requires CUDA tensor; refusing CPU fallback")
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(
            f"BASIWAN path requires fp16 or bf16 input, got {x.dtype}."
        )

    x2d = x.reshape(-1, x.shape[-1]).contiguous()
    M = x2d.shape[0]
    K = packed.k
    N = packed.n
    if x2d.shape[1] != K:
        raise RuntimeError(f"x.K={x2d.shape[1]} does not match packed.k={K}")

    # BASIWAN v2 routing: when BASIWAN_V2=1, route Q4_K and Q6_K to the v2 kernel.
    # See memory/marlin_v2_stages_1_huge_win_2026-06-06.md.
    import os as _os
    _use_basiwan_v2 = (
        _os.environ.get("BASIWAN_V2") == "1"
        and packed.quant_type in (QuantType.Q4_K, QuantType.Q6_K)
    )

    # M-alignment: v1 requires M%64=0, v2 requires M%192=0. Both pad with zeros
    # (zero rows produce zero accumulators, exact for real rows). Pad cost is
    # negligible (<1% extra rows at production shapes).
    _align = 192 if _use_basiwan_v2 else 64
    M_padded = ((M + _align - 1) // _align) * _align
    pad_rows = M_padded - M
    if pad_rows > 0:
        # Pool the padded input buffer (shape-keyed). Eliminates 332-881 MB
        # per-call fresh allocations that fragmented the allocator at p720_33f
        # (run #5, 2026-06-02). Pad rows must be re-zeroed every call —
        # the kernel computes dot products over all M_padded rows.
        x2d_padded = _MARLIN_INPUT_POOL.get(M_padded, K, x.device, x.dtype)
        x2d_padded[:M].copy_(x2d)
        x2d_padded[M:].zero_()
        x2d = x2d_padded

    # Output buffer: when caller passes an explicit `runtime`, use its
    # per-Linear buffer (legacy path). When `runtime is None`, use the
    # shared `_MARLIN_RUNTIME_POOL` — collapses ~72 per-Linear sticky
    # buffers (~18 GB at p720_17f) down to ~10 shape-keyed buffers
    # (~5 GB). The pool is safe because Marlin calls are serialized at
    # the Python forward layer (no concurrent kernel launches).
    if runtime is None:
        out2d = _MARLIN_RUNTIME_POOL.get(M_padded, N, x.device, x.dtype)
    else:
        out2d = runtime.ensure(M_padded, N, x.device, x.dtype)

    packed = _move_packed_to_device(packed, x.device, x.dtype)
    ext = _load_extension()

    # The extension function expects:
    #   x, B_ql, B_qh, eff_scales, eff_mins, out, quant_type, group_size
    # B_qh and eff_mins must be valid tensors even when unused (we pass empty placeholders).
    empty_int = torch.empty(0, dtype=torch.int32, device=x.device)
    empty_dt = torch.empty(0, dtype=x.dtype, device=x.device)

    # BASIWAN_WARPSPEC=1 routes to the warp-specialized kernel variant
    # (FFF/r_2_2_marlin_warpspec_ada.md). Phase 1 scaffold is bit-identical
    # to baseline; future phases introduce producer/consumer K-loop split.
    # Default OFF.
    _gemm = (ext.basiwan_q4k_q6k_gemm_ws
             if _os.environ.get("BASIWAN_WARPSPEC") == "1"
             else ext.basiwan_q4k_q6k_gemm)

    if _use_basiwan_v2:
        # BASIWAN v2 path — Q4_K AND Q6_K (STAGES=1 makes v2 faster than v1 on both).
        # Layout is bit-identical to v1's PackedBasiwanWeight (proven in Phase C).
        from basiwan_v2_kernel import _load_extension as _load_v2_ext
        _v2 = _load_v2_ext()
        if packed.quant_type == QuantType.Q4_K:
            _v2.basiwan_v2_gemm(
                x2d, packed.weight, empty_int,
                packed.scales, packed.mins,
                out2d,
                0, packed.group_size,
            )
        else:  # Q6_K
            _v2.basiwan_v2_gemm(
                x2d, packed.weight, packed.weight_hi,
                packed.scales, empty_dt,
                out2d,
                1, packed.group_size,
            )
    elif packed.quant_type == QuantType.Q4_K:
        _gemm(
            x2d, packed.weight, empty_int,
            packed.scales, packed.mins,
            out2d,
            0, packed.group_size,
        )
    elif packed.quant_type == QuantType.Q6_K:
        _gemm(
            x2d, packed.weight, packed.weight_hi,
            packed.scales, empty_dt,
            out2d,
            1, packed.group_size,
        )
    else:
        raise RuntimeError(
            f"Unsupported quant_type {packed.quant_type}; expected Q4_K(0) or Q6_K(1)"
        )

    # Slice off the padded rows before returning.
    if pad_rows > 0:
        out2d = out2d[:M]

    if bias is not None:
        if bias.device != out2d.device:
            bias = bias.to(out2d.device)
        if bias.dtype != out2d.dtype:
            bias = bias.to(out2d.dtype)
        # In-place add to avoid a fresh (M, N) allocation. The BasiwanRuntimeBuffers
        # output IS the destination; out2d is a slice view of it. `add_` broadcasts
        # bias over M without materializing.
        out2d.add_(bias)

    return out2d.reshape(*x.shape[:-1], N)


# BASIWAN-prefixed aliases (Phase 2 of the rename, user directive 2026-06-06).
# These are the preferred new-code names; the Marlin-prefixed names above
# remain as deprecated aliases for back-compat with existing call sites.
BasiwanRuntimeBuffers = BasiwanRuntimeBuffers
PackedBasiwanWeight = PackedBasiwanWeight
basiwan_q4k_linear = basiwan_q4k_linear
pack_basiwan_weight = pack_basiwan_weight


__all__ = [
    # BASIWAN-prefixed names (preferred, user directive 2026-06-06)
    "BasiwanRuntimeBuffers",
    "PackedBasiwanWeight",
    "QuantType",
    "basiwan_q4k_linear",
    "pack_basiwan_weight",
    # Legacy Marlin-prefixed names (deprecated aliases — keep for back-compat)
    "BasiwanRuntimeBuffers",
    "PackedBasiwanWeight",
    "basiwan_q4k_linear",
    "pack_basiwan_weight",
]
