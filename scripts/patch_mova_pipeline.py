"""BASIWAN — make the vendored OpenMOSS/MOVA package import on Windows/macOS.

MOVA's inference pipeline runs on plain torch + SDPA, but three modules import the
Linux-only `yunchang` (multi-GPU sequence/context-parallel attention) UNCONDITIONALLY at
module top level, so the whole package fails to import on a box without yunchang — even
though single-GPU inference (cp_mesh=None) never uses it (AttnType is only referenced as a
default arg in replace_attention(), the multi-GPU path).

This patch guards those imports with try/except + a complete AttnType fallback, exactly
mirroring the repo's own (incomplete) guard in wan_video_dit.py. It is IDEMPOTENT and
TOLERANT (missing file / already-patched -> no-op, exit 0), so install.js can run it after
cloning OpenMOSS/MOVA into ext/mova. Verified 2026-06-19: with these three guards the
pipeline imports + runs single-GPU inference with yunchang absent (SDPA attention).

Run from the BASIWAN app root.
"""
from pathlib import Path
import sys

ROOT = Path("ext/mova")

# A complete AttnType fallback — must carry every member referenced as a default arg at
# import/class-build time (USPAttention.__init__ and replace_attention default to AttnType.FA).
_FALLBACK = (
    "    class AttnType:  # type: ignore  [basiwan] yunchang-absent fallback\n"
    "        FA = None\n"
    "        FA3 = None\n"
    "        TORCH = None\n"
)

# (relative path, the exact upstream import line, the guarded replacement)
_GUARD_IMPORT = [
    "mova/diffusion/pipelines/pipeline_mova.py",
    "mova/diffusion/pipelines/mova_train.py",
]


def _patch_bases() -> list:
    """Every on-disk `mova` package to patch: the vendored source (ext/mova) AND the INSTALLED
    copy in env_mova's site-packages (pip install ./ext/mova COPIES the source, so patching only
    the source leaves the imported module unpatched on an already-installed env). Patching both
    makes the fix robust to install order and to re-running this patcher on a live env."""
    bases = []
    if ROOT.exists():
        bases.append(ROOT)
    try:
        import importlib.util as _u
        spec = _u.find_spec("mova")
        if spec and spec.origin:
            inst = Path(spec.origin).parent.parent          # site-packages (parent of the mova/ pkg)
            if inst.exists() and inst.resolve() not in {b.resolve() for b in bases}:
                bases.append(inst)
    except Exception:
        pass
    return bases


def _guard_yunchang_import(base, rel: str) -> str:
    p = base / rel
    if not p.exists():
        return f"[basiwan-mova] skip (not found): {rel}"
    t = p.read_text(encoding="utf-8")
    if "[basiwan] yunchang-absent fallback" in t:
        return f"[basiwan-mova] already patched: {rel}"
    needle = "from yunchang.kernels import AttnType"
    if needle not in t:
        return f"[basiwan-mova] no yunchang import to guard in {rel} (upstream changed?)"
    guarded = (
        "try:\n"
        "    from yunchang.kernels import AttnType\n"
        "except Exception:\n"
        "    # [basiwan] yunchang is Linux-only (multi-GPU CP attention) and unused by single-GPU\n"
        "    # inference; guard so the package imports on Windows/macOS (plain torch + SDPA).\n"
        + _FALLBACK
    )
    t = t.replace(needle, guarded, 1)
    p.write_text(t, encoding="utf-8")
    return f"[basiwan-mova] guarded yunchang import: {rel}"


def _fix_wan_dit_fallback(base) -> str:
    """The repo's OWN fallback in wan_video_dit.py defines only AttnType.FA3, but
    USPAttention.__init__ defaults to AttnType.FA -> AttributeError at import when yunchang
    is absent. Complete the fallback."""
    p = base / "mova/diffusion/models/wan_video_dit.py"
    if not p.exists():
        return "[basiwan-mova] skip (not found): wan_video_dit.py"
    t = p.read_text(encoding="utf-8")
    if "[basiwan] yunchang-absent fallback" in t:
        return "[basiwan-mova] already patched: wan_video_dit.py"
    needle = "    class AttnType:  # type: ignore\n        FA3 = None\n"
    if needle not in t:
        return "[basiwan-mova] wan_video_dit fallback shape changed (upstream changed?) — review manually"
    t = t.replace(needle, _FALLBACK, 1)
    p.write_text(t, encoding="utf-8")
    return "[basiwan-mova] completed AttnType fallback (added FA/TORCH): wan_video_dit.py"


# ── PyAV-backed torchcodec replacement ─────────────────────────────────────────────────
# MOVA's training dataset (mova/datasets/video_audio_dataset.py) decodes video+audio with
# torchcodec, whose CUDA-extension wheels track the newest torch (cu13x) and need FFmpeg shared
# libs — neither is present in the shipped env_mova (torch 2.7/cu128, low driver bar). BASIWAN
# already decodes video+audio with PyAV everywhere else (basi/mova_data.py), and PyAV ships a
# self-contained FFmpeg + works on env_mova as-is. This module provides VideoDecoder/AudioDecoder
# drop-ins with the exact surface video_audio_dataset.py uses, so MOVA TRAINING runs on the
# shipped env_mova with no torchcodec/cu130 dependency. The import is GUARDED (torchcodec first,
# this fallback otherwise), so a cu130+torchcodec env is used unchanged when present.
_PYAV_DECODE_SRC = '''"""[basiwan] PyAV-backed drop-in for torchcodec VideoDecoder/AudioDecoder.

Matches the minimal surface mova/datasets/video_audio_dataset.py uses:
  VideoDecoder(path, device=...).metadata.num_frames
  VideoDecoder(...).get_frames_at(indices=[...]).data   -> uint8 tensor [T, C, H, W] (RGB)
  AudioDecoder(path, sample_rate=sr, num_channels=1)
    .get_samples_played_in_range(start_seconds, stop_seconds).data -> float [C, N], .pts_seconds
Runs on plain torch + PyAV (self-contained FFmpeg); no torchcodec / no cu130 / no FFmpeg sys libs.
"""
import av as _av
import numpy as _np
import torch as _torch


class _Meta:
    def __init__(self, num_frames):
        self.num_frames = int(num_frames)


class _FrameBatch:
    def __init__(self, data):
        self.data = data


class VideoDecoder:
    def __init__(self, video_path, device="cpu", **kw):
        self._path = str(video_path)
        with _av.open(self._path) as c:
            s = c.streams.video[0]
            n = int(s.frames or 0)
            if n <= 0:
                n = sum(1 for _ in c.decode(s))
        self.metadata = _Meta(n)

    def get_frames_at(self, indices):
        want = sorted({int(i) for i in indices})
        maxi = want[-1] if want else -1
        got = {}
        with _av.open(self._path) as c:
            s = c.streams.video[0]
            i = 0
            for f in c.decode(s):
                if i in want:
                    got[i] = f.to_ndarray(format="rgb24")  # [H, W, C] uint8
                if i >= maxi:
                    break
                i += 1
        if not got:
            raise RuntimeError(f"[basiwan] PyAV decoded no frames from {self._path}")
        arr = _np.stack([got[i] for i in sorted(got)], axis=0)        # [T, H, W, C]
        t = _torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()   # [T, C, H, W] uint8
        return _FrameBatch(t)


class _AudioSamples:
    def __init__(self, data, pts_seconds):
        self.data = data
        self.pts_seconds = pts_seconds


class AudioDecoder:
    def __init__(self, video_path, sample_rate=48000, num_channels=1, **kw):
        self._path = str(video_path)
        self._sr = int(sample_rate)
        self._ch = int(num_channels)

    def get_samples_played_in_range(self, start_seconds=0.0, stop_seconds=None):
        layout = "mono" if self._ch == 1 else "stereo"
        chunks = []
        with _av.open(self._path) as c:
            if not c.streams.audio:
                dur = float(stop_seconds or 0.0) - float(start_seconds or 0.0)
                n = max(0, int(round(dur * self._sr)))
                return _AudioSamples(_torch.zeros((self._ch, n), dtype=_torch.float32), 0.0)
            s = c.streams.audio[0]
            resampler = _av.AudioResampler(format="fltp", layout=layout, rate=self._sr)
            def _emit(frame):
                res = resampler.resample(frame)
                if res is None:
                    return
                if not isinstance(res, list):
                    res = [res]
                for rf in res:
                    chunks.append(rf.to_ndarray())   # [C, n] float32 (planar)
            for f in c.decode(s):
                _emit(f)
            _emit(None)  # flush
        if not chunks:
            return _AudioSamples(_torch.zeros((self._ch, 0), dtype=_torch.float32), 0.0)
        arr = _np.concatenate(chunks, axis=1).astype("float32")       # [C, total]
        return _AudioSamples(_torch.from_numpy(arr), 0.0)
'''

_TC_NEEDLE = "from torchcodec.decoders import AudioDecoder, VideoDecoder"
_TC_GUARD = (
    "try:\n"
    "    from torchcodec.decoders import AudioDecoder, VideoDecoder\n"
    "except Exception:\n"
    "    # [basiwan] torchcodec needs cu13x + FFmpeg sys libs (absent on env_mova torch 2.7);\n"
    "    # fall back to the PyAV-backed decoders so MOVA training runs on the shipped env.\n"
    "    from mova.datasets._basiwan_pyav_decode import AudioDecoder, VideoDecoder\n"
)


def _add_pyav_decode_fallback(base) -> str:
    """Write the PyAV decoder module + guard video_audio_dataset.py's torchcodec import."""
    ds_dir = base / "mova" / "datasets"
    if not ds_dir.exists():
        return "[basiwan-mova] skip (not found): mova/datasets"
    helper = ds_dir / "_basiwan_pyav_decode.py"
    helper.write_text(_PYAV_DECODE_SRC, encoding="utf-8")          # idempotent overwrite
    out = ["[basiwan-mova] wrote PyAV decoder fallback: mova/datasets/_basiwan_pyav_decode.py"]
    ds = ds_dir / "video_audio_dataset.py"
    if not ds.exists():
        return "\n".join(out + ["[basiwan-mova] skip (not found): video_audio_dataset.py"])
    t = ds.read_text(encoding="utf-8")
    if "_basiwan_pyav_decode" in t:
        out.append("[basiwan-mova] already guarded: video_audio_dataset.py torchcodec import")
    elif _TC_NEEDLE in t:
        t = t.replace(_TC_NEEDLE, _TC_GUARD, 1)
        ds.write_text(t, encoding="utf-8")
        out.append("[basiwan-mova] guarded torchcodec import -> PyAV fallback: video_audio_dataset.py")
    else:
        out.append("[basiwan-mova] no torchcodec import to guard in video_audio_dataset.py (upstream changed?)")
    return "\n".join(out)


def main():
    bases = _patch_bases()
    if not bases:
        print("[basiwan-mova] no mova package found (ext/mova absent + not installed) — nothing to patch.")
        return
    for base in bases:
        print(f"[basiwan-mova] === patching {base} ===")
        for rel in _GUARD_IMPORT:
            print(_guard_yunchang_import(base, rel))
        print(_fix_wan_dit_fallback(base))
        print(_add_pyav_decode_fallback(base))


if __name__ == "__main__":
    main()
    sys.exit(0)
