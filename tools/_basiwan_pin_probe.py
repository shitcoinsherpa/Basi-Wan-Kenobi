"""Runtime probe for pinned-host-memory budget.

Determines the maximum K (number of pinned ring-buffer slots) we can hold
without tripping the WSL2 dxgkrnl pinned-page cap or breaking the next
CUDA op. Strategy: try-register-then-immediately-unregister N times,
verifying with a tiny CUDA op between each, so peak pinned during probe
is exactly one slot — never K × slot_bytes.

Platform detection is via `/proc/version` (WSL signal in lowercased
contents). The probe is safe to call before model load; it consumes a
few MB of host RAM transiently and one slot_bytes-sized region per
trial step. All transient allocations are released before return.
"""
import os
import sys

# `resource` is Unix-only (Linux/macOS). On Windows there is no RLIMIT_MEMLOCK
# — pinned memory is bounded only by physical RAM via cudaHostRegister, no
# RLIMIT_MEMLOCK soft cap. Gate the import so the module loads on Windows.
try:
    import resource  # type: ignore[import-not-found]
    _HAVE_RESOURCE = True
except ImportError:
    resource = None  # type: ignore[assignment]
    _HAVE_RESOURCE = False

import torch


def _detect_platform():
    plat = {"is_wsl": False, "is_linux": False, "is_macos": False, "is_win": False}
    if sys.platform == "darwin":
        plat["is_macos"] = True
        return plat
    if os.name == "nt":
        plat["is_win"] = True
        return plat
    try:
        with open("/proc/version") as f:
            v = f.read().lower()
        plat["is_linux"] = True
        plat["is_wsl"] = ("microsoft" in v) or ("wsl" in v)
    except FileNotFoundError:
        pass
    return plat


def _memlock_hard_bytes():
    # Windows has no RLIMIT_MEMLOCK; pinned memory is bounded by physical RAM
    # via cudaHostRegister. Report a very large value so downstream sizing
    # picks the desired K_MAX rather than artificially capping at 64 MB.
    if not _HAVE_RESOURCE:
        return 1 << 62
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if hard == resource.RLIM_INFINITY:
            return 1 << 62
        return hard
    except Exception:
        return 64 << 20


def _shm_free_bytes():
    try:
        st = os.statvfs("/dev/shm")
        return st.f_bavail * st.f_frsize
    except Exception:
        return 0


def _host_avail_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def probe_pin_budget(slot_bytes, max_slots=8, headroom_bytes=64 << 20):
    """Probe how many pinned slots of `slot_bytes` each we can safely hold.

    Returns:
        int: K in [0, max_slots]. 0 means async pinning is not viable on
        this platform / configuration; caller should fall back to sync.

    Args:
        slot_bytes: Size of each ring-buffer slot (use the largest single
            block's byte count to ensure any block fits).
        max_slots: Upper bound on K — don't probe beyond this even if
            the cap allows it.
        headroom_bytes: Bytes to leave free below the cap for cublas /
            T5 load / other CUDA initial allocations.
    """
    plat = _detect_platform()
    if plat["is_macos"]:
        return 0
    if not torch.cuda.is_available():
        return 0

    memlock = _memlock_hard_bytes()
    if plat["is_wsl"]:
        # WSL2: memlock cap is NOT enforced for pinned pages — they route
        # through /dev/shm tmpfs. Use shm free as the ceiling.
        shm = _shm_free_bytes()
        ceiling = shm // 2 if shm > 0 else memlock
    else:
        ceiling = memlock

    # Always honor the per-process available host RAM.
    avail = _host_avail_bytes()
    if avail > 0:
        ceiling = min(ceiling, avail - (1 << 30))  # leave 1 GB for python heap

    if ceiling <= headroom_bytes:
        return 0

    cudart = torch.cuda.cudart()
    k_ok = 0
    held = []  # list of (buffer, ptr, nbytes) we registered and not yet unregistered

    def _release_all():
        for _buf, _ptr, _nb in held:
            try:
                cudart.cudaHostUnregister(_ptr)
            except Exception:
                pass
        held.clear()

    try:
        for k in range(1, max_slots + 1):
            need = k * slot_bytes
            if need > ceiling - headroom_bytes:
                break
            buf = torch.empty(slot_bytes, dtype=torch.uint8)
            storage = buf.untyped_storage()
            ptr = storage.data_ptr()
            nbytes = storage.nbytes()
            try:
                rc = cudart.cudaHostRegister(ptr, nbytes, 0)
            except Exception:
                break
            if int(rc) != 0:
                break

            # Smoke: tiny CUDA op must succeed. If WSL2 dxgkrnl translates a
            # pinned-cap violation to cudaErrorMemoryAllocation, this fires.
            try:
                tmp = torch.zeros(4, device="cuda:0")
                tmp.add_(1)
                torch.cuda.synchronize()
                del tmp
            except Exception:
                try:
                    cudart.cudaHostUnregister(ptr)
                except Exception:
                    pass
                break
            held.append((buf, ptr, nbytes))
            k_ok = k
    finally:
        _release_all()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    return k_ok
