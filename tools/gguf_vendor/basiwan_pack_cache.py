"""Persistent BASIWAN pre-pack cache.

Pre-packing 400 Q4_K weights per expert takes ~280s of pure CPU work. Across
N benches this cost is paid N times. This module caches the result keyed by
(GGUF path + mtime + size) so subsequent runs load the packed tensors from
disk in seconds.

Cache layout:
  $BASIWAN_PACK_CACHE_DIR/<safe-gguf-basename>__<size>_<mtime>.pt

The cache file holds a dict {module_dotted_name: serialized PackedBasiwanWeight}.
On load we walk the module tree and reassign each module's `_basiwan_packed`.

Disabled when `BASIWAN_NO_PACK_CACHE=1` or when the GGUF file path
can't be resolved.
"""
from __future__ import annotations
import hashlib
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# [2026-06-11 #380] Registry of in-flight mmap pre-warm threads so the first
# generation can BLOCK on them instead of racing a cold 19 GB mapping. The
# pre-warm was previously fire-and-forget: `ready` was emitted before it
# joined, so the first I2V forward pass page-faulted the pack at disk speed
# (~150 s alone, amplified to 1000 s+ by the #378 duplicate worker eating
# page cache). join_prewarm() converts that hidden random-fault storm into
# one visible, sequential wait. See memory/tier1_release_buildplans.
_PREWARM_THREADS: list = []


def join_prewarm(progress_cb=None, poll_s: float = 3.0) -> float:
    """Block until all registered mmap pre-warm threads finish. Returns the
    wall seconds waited. progress_cb(elapsed_s) is called every poll_s so the
    caller can surface a 'paging in model weights' phase to the UI. Safe to
    call when no threads are pending (returns ~0)."""
    pending = [t for t in _PREWARM_THREADS if t.is_alive()]
    if not pending:
        _PREWARM_THREADS.clear()
        return 0.0
    t0 = time.time()
    for t in pending:
        while t.is_alive():
            t.join(timeout=poll_s)
            if t.is_alive() and progress_cb is not None:
                try:
                    progress_cb(time.time() - t0)
                except Exception:
                    pass  # a UI callback must never break the join
    _PREWARM_THREADS.clear()
    return time.time() - t0


def _cache_dir() -> Path:
    # Default to ext4 (~/.cache/marlin_packs). The earlier /mnt/d default was
    # the WSL 9p mount that corrupts tensor state under torch.save load —
    # see audit_marlin_pack_cache_ROOT_CAUSED_2026-06-05. Sourcing _env.sh
    # always overrides via BASIWAN_PACK_CACHE_DIR, but unsourced callers
    # (tests, CI, external repro) now also get the safe default.
    # .expanduser() is REQUIRED: pathlib does NOT expand "~" on its own.
    # Without it, BASIWAN_PACK_CACHE_DIR="~/.cache/marlin_packs" (start.js)
    # created a LITERAL "~" directory under the worker cwd on Windows —
    # cache never hit, every cold start re-packed both 14B experts (~8 min)
    # and re-wrote ~19 GB of cache. Root-caused 2026-06-09 live on the
    # user's Pinokio install (D:\Pinokio\api\basiwan.git\~\.cache\...).
    p = Path(os.environ.get(
        "BASIWAN_PACK_CACHE_DIR", str(Path.home() / ".cache/marlin_packs"))).expanduser()
    if str(p).startswith("/mnt/"):
        import warnings
        warnings.warn(
            f"BASIWAN_PACK_CACHE_DIR={p} is on a /mnt mount; WSL 9p path "
            "corrupts torch-saved tensor state. Use ext4 (~/.cache).",
            RuntimeWarning, stacklevel=2)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_key(gguf_path: Path) -> str:
    st = gguf_path.stat()
    safe = gguf_path.name.replace(".", "_")
    return f"{safe}__sz{st.st_size}_mt{int(st.st_mtime)}"


def _cache_path(gguf_path: Path) -> Path:
    return _cache_dir() / f"{_cache_key(gguf_path)}.pt"


def _pack_to_dict(p) -> dict:
    """PackedBasiwanWeight → dict (only tensor + meta fields, no class identity)."""
    return {
        "quant_type": p.quant_type,
        "weight": p.weight,
        "weight_hi": p.weight_hi,
        "scales": p.scales,
        "mins": p.mins,
        "n": p.n,
        "k": p.k,
        "group_size": p.group_size,
    }


def _dict_to_pack(d: dict, materialize: bool = True):
    """Reconstruct PackedBasiwanWeight from a cache dict.

    materialize=True (default): clone each tensor so the result lives in anon RSS
    instead of an mmap'd file. /mnt/d is a 9p mount to Windows NTFS — leaving
    tensors mmap-backed there makes every diffusion-time access page-fault
    through the slow 9p layer (measured 2.25× wall regression).
    """
    from .basiwan_q4_kernel import PackedBasiwanWeight as PackedBasiwanWeight
    if materialize:
        return PackedBasiwanWeight(
            quant_type=d["quant_type"],
            weight=d["weight"].clone(),
            weight_hi=d["weight_hi"].clone() if d["weight_hi"] is not None else None,
            scales=d["scales"].clone(),
            mins=d["mins"].clone() if d["mins"] is not None else None,
            n=d["n"],
            k=d["k"],
            group_size=d["group_size"],
        )
    return PackedBasiwanWeight(
        quant_type=d["quant_type"],
        weight=d["weight"],
        weight_hi=d["weight_hi"],
        scales=d["scales"],
        mins=d["mins"],
        n=d["n"],
        k=d["k"],
        group_size=d["group_size"],
    )


def cached_prepack_basiwan_weights(
    module: nn.Module,
    gguf_path: Optional[Path] = None,
) -> tuple[int, bool]:
    """Pre-pack with on-disk cache.

    Returns (n_packed, from_cache).

    When the cache hits, packed tensors are loaded from disk and assigned to
    each BareGGMLLinear's `_basiwan_packed`; `prepack_basiwan_weights` is NOT
    called and the original Q4 blobs stay in place until the first forward.

    When the cache misses, falls back to the original prepack and writes the
    cache atomically (tmp + rename).
    """
    from .bare_gguf import BareGGMLLinear, prepack_basiwan_weights, is_quantized

    if (gguf_path is None
            or os.environ.get("BASIWAN_NO_PACK_CACHE") == "1"):
        n = prepack_basiwan_weights(module)
        return n, False

    gguf_path = Path(gguf_path)
    if not gguf_path.exists():
        n = prepack_basiwan_weights(module)
        return n, False

    cache_path = _cache_path(gguf_path)
    if cache_path.exists():
        try:
            t0 = time.time()
            # mmap=True keeps tensors as mmap-views (host VMA, no anon RSS).
            # Without it, torch.load fully materializes the ~9.4 GB dict at
            # once, which combined with the OTHER expert's resident packs +
            # the LOoRA Q4 blobs still in memory peaks ~26 GB and OOMs the
            # WSL 32 GB-capped VM. With mmap, peak stays under 14 GB.
            packed_state = torch.load(str(cache_path), weights_only=False,
                                      map_location="cpu", mmap=True)
            # [2026-06-09] mmap pre-warm: background thread that touches every
            # 4 KB page of the mmap region forces sequential disk reads, which
            # the OS coalesces into large I/Os. Without this, the first forward
            # pass touches ~400 tensors in irregular order and trips a 100-160s
            # random-access page-fault penalty on a cold OS page cache (Windows
            # native, subprocess-per-click pattern). See
            # memory/cold_cache_penalty_root_caused_2026-06-09.md.
            # Gated on BASIWAN_MMAP_PREWARM!="0" so we can A/B.
            if os.environ.get("BASIWAN_MMAP_PREWARM") != "0":
                import threading as _th, ctypes as _ct
                _ps_ref = packed_state  # capture reference for the closure

                def _warm():
                    try:
                        import numpy as _np
                        # Unique storages, dedup by data_ptr.
                        _seen = set()
                        _storages = []
                        for _t in _ps_ref.values():
                            if isinstance(_t, dict):
                                _vals = _t.values()
                            else:
                                _vals = [_t]
                            for _v in _vals:
                                if not hasattr(_v, "untyped_storage"):
                                    continue
                                _s = _v.untyped_storage()
                                _ptr = _s.data_ptr()
                                if _ptr in _seen or _s.nbytes() == 0:
                                    continue
                                _seen.add(_ptr)
                                _storages.append((_ptr, _s.nbytes()))
                        _t0 = time.time()
                        _total = 0
                        for _ptr, _sz in _storages:
                            # numpy view of the raw bytes; releases GIL during scan.
                            _arr = _np.ctypeslib.as_array(
                                (_ct.c_uint8 * _sz).from_address(_ptr))
                            # Touch one byte every 4096 to force page-in.
                            _ = int(_arr[::4096].sum())
                            _total += _sz
                        _dt = time.time() - _t0
                        _gb = _total / (1024 ** 3)
                        print(f"[basiwan-cache] mmap-prewarm {_gb:.1f}GB in "
                              f"{_dt:.1f}s ({_gb/_dt:.2f} GB/s)", flush=True)
                    except Exception as _e:
                        print(f"[basiwan-cache] mmap-prewarm skipped: "
                              f"{type(_e).__name__}: {_e}", flush=True)

                _pw = _th.Thread(target=_warm, name="mmap-prewarm", daemon=True)
                _pw.start()
                _PREWARM_THREADS.append(_pw)  # [#380] joinable before first gen
            n_loaded = 0
            for name, sub in module.named_modules():
                if not isinstance(sub, BareGGMLLinear):
                    continue
                if name not in packed_state:
                    continue
                d = packed_state[name]
                sub._basiwan_packed = _dict_to_pack(d)
                # Free the original GGUF blob — BASIWAN owns dequant from here.
                # Matches prepack_basiwan_weights:333 behavior.
                if sub.weight is not None and is_quantized(sub.weight):
                    sub.weight = None
                n_loaded += 1
            # Drop the dict reference. The PackedBasiwanWeight assignments above
            # hold the underlying tensors directly; the dict is just a key index.
            packed_state = None
            print(f"[basiwan-cache] loaded {n_loaded} packs from "
                  f"{cache_path.name} in {time.time()-t0:.1f}s (mmap)", flush=True)
            return n_loaded, True
        except Exception as e:
            print(f"[basiwan-cache] load failed ({e}); falling back to fresh pack",
                  flush=True)

    # Cache miss or load failure → fresh pack
    t0 = time.time()
    n = prepack_basiwan_weights(module)
    pack_s = time.time() - t0

    # [2026-06-05 audit-AA diagnostic] BASIWAN_CACHE_NO_WRITE=1 skips the save.
    # Tests whether the torch.save block is mutating tensor state (vs. just slow I/O).
    if os.environ.get("BASIWAN_CACHE_NO_WRITE") == "1":
        print(f"[basiwan-cache] SKIPPING write (BASIWAN_CACHE_NO_WRITE=1)", flush=True)
        return n, False

    # Serialize and save atomically (write to .tmp, fsync, rename)
    try:
        t0 = time.time()
        state = {}
        for name, sub in module.named_modules():
            if not isinstance(sub, BareGGMLLinear):
                continue
            if sub._basiwan_packed is None:
                continue
            state[name] = _pack_to_dict(sub._basiwan_packed)

        tmp_path = cache_path.with_suffix(".pt.tmp")
        torch.save(state, str(tmp_path))
        # fsync for crash safety. Windows raises OSError(EBADF) on read-only
        # fd; torch.save already flushed buffers and os.replace gives atomicity,
        # so swallow the fsync error rather than abort the rename and leak .tmp.
        try:
            with tmp_path.open("rb") as f:
                os.fsync(f.fileno())
        except OSError:
            pass
        os.replace(str(tmp_path), str(cache_path))
        save_s = time.time() - t0
        print(f"[basiwan-cache] wrote {len(state)} packs to {cache_path.name} "
              f"in {save_s:.1f}s (next launch saves ~{pack_s:.0f}s)", flush=True)
    except Exception as e:
        print(f"[basiwan-cache] save failed ({e}); continuing without cache",
              flush=True)
    return n, False
