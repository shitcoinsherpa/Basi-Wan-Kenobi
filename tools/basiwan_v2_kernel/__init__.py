"""BASIWAN v2 — from-scratch Q4_K + Q6_K CUDA kernel for Wan2.2.

See docs/DESIGN.md for the architecture rationale.

This module is multi-session work; the kernel is implemented incrementally
across phases A through D. The Python surface here is the BUILD/IMPORT layer
that pytorch's cpp_extension uses.

Phase D has shipped (2026-06-06). Callers route to this module when
`BASIWAN_V2=1` is set (the production ship default; see _env.sh).
The `tools/gguf_vendor/q4_marlin` v1 kernel remains as fallback when
BASIWAN_V2 is unset or on non-sm_89 GPUs.
"""
from __future__ import annotations

import os
import sys
import torch
from pathlib import Path

_HERE = Path(__file__).parent
_EXT = None


def _load_extension():
    """Build (if needed) and import the marlin_v2 CUDA extension.

    The kernel is in `basiwan_v2.cu`. Compile flags target sm_89 (Ada) for now;
    sm_90+ tuning happens after sm_89 ships.
    """
    global _EXT
    if _EXT is not None:
        return _EXT
    if not torch.cuda.is_available():
        raise RuntimeError(
            "marlin_v2 requires CUDA — no CPU fallback (would be slower than v1 anyway)"
        )
    from torch.utils.cpp_extension import load

    maxrreg = int(os.environ.get("BASIWAN_V2_MAXRREG", "128"))
    skip_qh = os.environ.get("BASIWAN_V2_Q6K_SKIP_QH", "0") == "1"
    name_suffix = "_skipqh" if skip_qh else ""
    name = f"wan_basiwan_v2_ext_r{maxrreg}{name_suffix}"
    cflags = [
        "-O3",
        "-std=c++17",
        "--use_fast_math",
        "-gencode=arch=compute_89,code=sm_89",
        f"--maxrregcount={maxrreg}",
        # During Phase A development, keep -Xptxas=-v on so we can spot register
        # pressure regressions immediately. Remove for Phase D ship build.
        "-Xptxas=-v",
    ]
    if skip_qh:
        # DIAGNOSTIC ONLY: breaks Q6_K correctness, isolates qh-load cost.
        cflags.append("-DBASIWAN_V2_Q6K_SKIP_QH")
    if os.name == "nt":
        cflags.append("-allow-unsupported-compiler")
        # 2026-06-08: direct-load cached .pyd to skip torch's build-chain
        # re-validation (which requires vcvarsall'd shell). Mirrors
        # basiwan_q4_kernel Windows-port fix.
        _torch_ext_root = Path(
            os.environ.get(
                "TORCH_EXTENSIONS_DIR",
                Path(os.environ.get("LOCALAPPDATA", "")) / "torch_extensions" / "torch_extensions" / "Cache",
            )
        )
        _cuda_ver = (torch.version.cuda or "").replace(".", "")
        _pyd_path = _torch_ext_root / f"py{sys.version_info.major}{sys.version_info.minor}_cu{_cuda_ver}" / name / f"{name}.pyd"
        if _pyd_path.exists():
            import importlib.util as _ilu
            _spec = _ilu.spec_from_file_location(name, str(_pyd_path))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _EXT = _mod
            return _EXT
        # First-build path: augment PATH for ninja + CUDA 13.1.
        _scripts = str(Path(sys.executable).resolve().parent)
        if _scripts not in os.environ.get("PATH", ""):
            os.environ["PATH"] = _scripts + os.pathsep + os.environ.get("PATH", "")
        _cuda13 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
        if Path(_cuda13).exists() and _cuda13 not in os.environ.get("PATH", ""):
            os.environ["CUDA_HOME"] = _cuda13
            os.environ["CUDA_PATH"] = _cuda13
            os.environ["PATH"] = _cuda13 + r"\bin" + os.pathsep + os.environ.get("PATH", "")
    _EXT = load(
        name=name,
        sources=[str(_HERE / "basiwan_v2.cu")],
        extra_include_paths=[str(_HERE)],
        extra_cuda_cflags=cflags,
        verbose=True,
    )
    return _EXT


def stub_status() -> str:
    """Phase A check: confirm the extension builds and the stub entry is callable."""
    ext = _load_extension()
    return ext.stub_status()


__all__ = ["_load_extension", "stub_status"]
