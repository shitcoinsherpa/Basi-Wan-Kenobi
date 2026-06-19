"""[#395] Pre-build the BASIWAN CUDA kernels at INSTALL time.

The v2 (Q4_K/Q6_K) and v1 (q4) kernels JIT-compile on first generate via
_load_extension() (~50s nvcc compile). Doing it here means: (1) the first user
generate doesn't stall on a silent compile, and (2) a missing toolchain
(nvcc / MSVC) surfaces in the install terminal, not mid-generate.

TOLERANT BY DESIGN: any failure (no CUDA, no nvcc/MSVC, CUDA at a non-default
path) is a WARNING, not an error — the runtime still lazy-builds on first use
exactly as before. So this never blocks a working install. ASCII only."""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))


def _try(label, importer):
    t0 = time.time()
    try:
        load = importer()
    except Exception as e:  # loader module itself didn't import
        print(f"[build-kernel] {label}: loader import failed "
              f"({type(e).__name__}: {e}) — runtime will lazy-build", flush=True)
        return False
    try:
        ext = load()
        ok = ext is not None
        print(f"[build-kernel] {label}: {'built/loaded OK' if ok else 'returned None'} "
              f"in {time.time() - t0:.0f}s", flush=True)
        return ok
    except Exception as e:  # compile/load failed — fine, lazy path remains
        print(f"[build-kernel] {label}: build skipped "
              f"({type(e).__name__}: {e}) — runtime will lazy-build on first generate",
              flush=True)
        return False


def _check_toolchain() -> list:
    """Return the list of MISSING build tools. The kernel compiles via torch
    cpp_extension, which needs all three: ninja (pip), the MSVC C++ compiler
    (cl.exe, from VS Build Tools), and the CUDA Toolkit (nvcc). pip can only
    supply ninja — the other two are system installs."""
    import shutil
    missing = []
    if shutil.which("ninja") is None:
        try:
            import ninja  # noqa: F401  (the wheel ships ninja.exe in Scripts)
        except Exception:
            missing.append("ninja  (pip install ninja — usually auto from requirements.txt)")
    if os.name == "nt" and shutil.which("cl") is None:
        missing.append("cl.exe (MSVC C++ — install 'Visual Studio Build Tools' + the "
                       "'Desktop development with C++' workload, then launch from an "
                       "'x64 Native Tools Command Prompt' OR add it to PATH)")
    if shutil.which("nvcc") is None:
        missing.append("nvcc   (CUDA Toolkit — install the NVIDIA CUDA Toolkit matching "
                       "your torch CUDA version; add its bin/ to PATH)")
    return missing


def main() -> int:
    try:
        import torch
    except Exception as e:
        print(f"[build-kernel] torch import failed ({e}) — skipping pre-build", flush=True)
        return 0
    if not torch.cuda.is_available():
        print("[build-kernel] CUDA not available (CPU / AMD / Apple / driver absent) "
              "— skipping; the runtime handles the non-CUDA path", flush=True)
        return 0
    missing = _check_toolchain()
    if missing:
        print("[build-kernel] build toolchain INCOMPLETE — the BASIWAN CUDA kernel "
              "cannot be compiled until these are installed:", flush=True)
        for m in missing:
            print(f"[build-kernel]   - MISSING: {m}", flush=True)
        print("[build-kernel] The kernel will be (re)built automatically on first "
              "generate once the toolchain is present. See README 'Build "
              "prerequisites'. Continuing install (non-fatal).", flush=True)
        return 0  # never fail the install on a missing toolchain

    def _v2():
        from basiwan_v2_kernel import _load_extension
        return _load_extension

    def _q4():
        from gguf_vendor.basiwan_q4_kernel import _load_extension
        return _load_extension

    n = int(_try("v2 (Q4_K/Q6_K fast path)", _v2)) + int(_try("v1 (q4 fallback)", _q4))
    print(f"[build-kernel] done — {n}/2 kernel extension(s) ready at install time",
          flush=True)
    return 0  # always success: a skipped pre-build must never fail the install


if __name__ == "__main__":
    raise SystemExit(main())
