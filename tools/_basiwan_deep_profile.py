"""BASIWAN deep-profile — single module per 2026-06-08 research fan-out.

Two modes:
- LIGHT (BASIWAN_PROFILE=1): per-step wall + alloc + cudaPeekAtLastError
  at phase boundaries. <2% overhead. Always-safe on production runs.
- DEEP (BASIWAN_DEEP_PROFILE=1): per-block CUDA events, sync-degrade
  detection on every non_blocking=True .to(), pinned-byte scan via
  /proc/self/maps + cudaHostGetFlags, memory._snapshot dump on OOM.

Wired into tools/run_one_video_gguf.py at three sites:
1. main() expert boundaries via assert_cuda_clean / step_start / step_end
2. _basiwan_swap_forward at H2D prefetch (detect_sync_degradation wrap)
3. _prepare_then_reapply_swap at MoE boundary residency burst

Built from the four-agent research fan-out 2026-06-08. The single most
load-bearing signal: assert_cuda_clean() before each expert-boundary
.to(cuda) call. WSL2 dxgkrnl translates at least two distinct failures
(resource table exhaustion, pinned-page cap) into cudaErrorMemoryAllocation
with zero diagnostic context. The peek surfaces which one, and where.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
from typing import Optional

import torch


_LIGHT = os.environ.get("BASIWAN_PROFILE") == "1"
_DEEP = os.environ.get("BASIWAN_DEEP_PROFILE") == "1"

# Touching cudart for cudaPeekAtLastError (CPU register read, sub-µs).
# We import lazily because not all hosts have libcudart at the expected
# path. Fall back to torch.cuda.synchronize() as a non-clearing probe
# if cudart isn't reachable.
_CUDART = None
try:
    for _path in (
        os.environ.get("BASIWAN_CUDART_PATH"),
        "libcudart.so.13",
        "libcudart.so.12",
        "/usr/local/cuda-13/lib64/libcudart.so.13",
        "/usr/local/cuda-12.8/lib64/libcudart.so.12",
    ):
        if not _path:
            continue
        try:
            _CUDART = ctypes.CDLL(_path)
            _CUDART.cudaPeekAtLastError.restype = ctypes.c_int
            break
        except OSError:
            continue
except Exception:
    _CUDART = None


def assert_cuda_clean(label: str) -> None:
    """Peek at the CUDA error register and raise if non-zero.

    Per WSL2 dxgkrnl behavior (microsoft/WSL #8447 #10269 #11050), a
    failure in the kernel-side resource table or pinned-page route
    manifests as cudaErrorMemoryAllocation on the NEXT user CUDA call.
    Calling this before each phase boundary surfaces the error at its
    actual origin instead of at a stale downstream callsite.

    Cost: one CPU register read. Non-blocking. Safe to leave always-on.
    """
    if not _LIGHT:
        return
    if _CUDART is None:
        return
    rc = _CUDART.cudaPeekAtLastError()
    if rc != 0:
        # Don't clear; let downstream see it too. Just report.
        print(f"[basiwan-prof] CUDA ERROR PENDING at {label!r}: code={rc}",
              flush=True)


_STEP_T0: dict[str, float] = {}
_STEP_EV: dict[str, torch.cuda.Event] = {}


def step_start(label: str) -> None:
    if not _LIGHT:
        return
    if torch.cuda.is_available():
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        _STEP_EV[label] = ev
    _STEP_T0[label] = time.perf_counter()


def step_end(label: str) -> None:
    if not _LIGHT:
        return
    t0 = _STEP_T0.pop(label, None)
    if t0 is None:
        return
    cpu_ms = (time.perf_counter() - t0) * 1e3
    gpu_ms = -1.0
    ev_s = _STEP_EV.pop(label, None)
    if ev_s is not None and torch.cuda.is_available():
        ev_e = torch.cuda.Event(enable_timing=True)
        ev_e.record()
        ev_e.synchronize()
        try:
            gpu_ms = ev_s.elapsed_time(ev_e)
        except Exception:
            gpu_ms = -1.0
    alloc_gb = -1.0
    resv_gb = -1.0
    retries = -1
    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated() / (1 << 30)
        resv_gb = torch.cuda.memory_reserved() / (1 << 30)
        try:
            retries = torch.cuda.memory_stats().get('num_alloc_retries', -1)
        except Exception:
            pass
    print(f"[basiwan-prof] {label} cpu={cpu_ms:.0f}ms gpu={gpu_ms:.0f}ms "
          f"alloc={alloc_gb:.2f}GB resv={resv_gb:.2f}GB retries={retries}",
          flush=True)
    assert_cuda_clean(f"after {label}")


# --- DEEP only ---

def detect_sync_degradation(label: str, copy_callable, threshold_ms: float = 1.0):
    """Wrap a .to(..., non_blocking=True) call. Warn if it blocked the
    host longer than expected for a true async DMA enqueue.

    Returns whatever copy_callable returned.
    """
    if not _DEEP:
        return copy_callable()
    t0 = time.perf_counter()
    out = copy_callable()
    dt_ms = (time.perf_counter() - t0) * 1e3
    if dt_ms > threshold_ms:
        print(f"[basiwan-prof] SYNC-DEGRADE {label}: host wall={dt_ms:.1f}ms "
              f"on non_blocking=True (source likely unpinned)", flush=True)
    return out


def scan_pinned_bytes() -> int:
    """Walk /proc/self/maps, identify regions whose START address
    successfully returns flags via cudaHostGetFlags (== pinned), sum bytes.

    Uses libcudart directly via ctypes — torch._C._cudart does NOT expose
    cudaHostGetFlags (verified empirically 2026-06-08). The _CUDART handle
    is initialized at module import.
    """
    if not _DEEP or _CUDART is None:
        return -1
    # Set up cudaHostGetFlags(unsigned int *pFlags, void *pHost) -> int
    try:
        _CUDART.cudaHostGetFlags.argtypes = [ctypes.POINTER(ctypes.c_uint),
                                              ctypes.c_void_p]
        _CUDART.cudaHostGetFlags.restype = ctypes.c_int
    except Exception:
        return -1
    flags = ctypes.c_uint(0)
    total = 0
    try:
        with open(f"/proc/{os.getpid()}/maps") as f:
            for line in f:
                # On WSL2 cudaHostRegister'd pages don't always show as
                # /dev/zero — we test ANY rw segment that's anonymous.
                if " r" not in line[:6]:
                    continue
                # Skip large library-mapped ranges to keep walk cheap.
                if ".so" in line:
                    continue
                try:
                    rng = line.split()[0]
                    a_str, b_str = rng.split("-")
                    start = int(a_str, 16)
                    end = int(b_str, 16)
                    rc = _CUDART.cudaHostGetFlags(ctypes.byref(flags),
                                                  ctypes.c_void_p(start))
                    if rc == 0:
                        total += (end - start)
                except Exception:
                    pass
    except Exception:
        pass
    return total


def report_pinned(label: str) -> None:
    if not _DEEP:
        return
    nb = scan_pinned_bytes()
    if nb < 0:
        return
    print(f"[basiwan-prof] pinned[{label}] = {nb / (1<<30):.3f} GB",
          flush=True)


def dump_snapshot_on_oom(label: str, path: str = "/tmp/basiwan_oom_snap.pickle") -> None:
    """Call from an except OOM block to capture the allocator state."""
    try:
        from torch.cuda import memory as _mem
        snap = _mem._snapshot()
        import pickle
        with open(path, "wb") as f:
            pickle.dump(snap, f)
        print(f"[basiwan-prof] OOM snapshot @ {label} → {path}", flush=True)
    except Exception as e:
        print(f"[basiwan-prof] snapshot dump failed @ {label}: {e}", flush=True)


def enabled() -> bool:
    return _LIGHT or _DEEP


def deep_enabled() -> bool:
    return _DEEP
