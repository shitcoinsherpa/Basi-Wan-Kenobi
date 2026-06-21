"""Auto-split long videos into 2-5s single-shot training clips.

The Wan2.2 LoRA dataset recipe wants 20-50 clips of ~2-5s, each ONE
continuous shot — a cut inside a clip
teaches the model to cut. This module turns "a few big videos" into that
shape: ffmpeg scene-change detection finds shot boundaries, long shots are
subsampled into evenly-spaced chunks (varied moments, not consecutive
frames), too-short shots are dropped with a reason, and every produced
clip gets a sharpness FLAG (never an auto-delete — curation stays human).

Engine: the ffmpeg binary bundled by imageio-ffmpeg. PySceneDetect was
REJECTED: its ContentDetector imports cv2 unconditionally — even with the
PyAV backend — so it drags in opencv-python (~200 MB) for marginal gain
over ffmpeg's scene score on hard cuts. ffmpeg's known weakness
(gradual fades/dissolves under the
threshold) is acceptable for dataset prep; fades make poor training clips
anyway.

Scene scores: `select=gt(scene,T)` marks the FIRST frame of each new
shot; `metadata=print:file=` writes a clean two-line record per hit
(frame header with pts_time, then lavfi.scene_score=...) — far easier to
parse than showinfo's stderr lines. Splitting: input-side -ss/-to with
re-encode is frame-accurate (-accurate_seek default since ffmpeg 2.1).
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .dataset import probe_video

VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v"}

# Clip-shape constants from the training brief. Exposed as function args
# for the UI but these defaults ARE the recipe.
# 2.2 not 2.0: musubi's 24→16fps resampling turns a 2.000s clip into 32
# frames — one short of a 33-frame window — and it silently caches ZERO
# items for that clip (exact boundary 2.042s at 24fps source). 2.2 adds
# safety margin.
MIN_CLIP_S = 2.2
MAX_CLIP_S = 5.0       # above this, subsample chunks instead
TARGET_CHUNK_S = 3.5   # chunk length cut from long shots
MAX_CHUNKS_PER_SHOT = 4  # one static long shot must not dominate the set
DEFAULT_MAX_TOTAL = 60   # >80 clips is diminishing returns; leave headroom
DEFAULT_THRESHOLD = 0.4  # ffmpeg scene-score de facto default for hard cuts


def _ffmpeg_exe() -> str:
    # Prefer the imageio-ffmpeg-bundled binary (what the gym ships), but fall back to a system
    # ffmpeg on PATH so the dataset pipeline works in any env that has ffmpeg but not the wheel.
    # Errors clearly if neither exists.
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        raise RuntimeError(
            "ffmpeg not found: install `imageio-ffmpeg` (pip) or a system ffmpeg on PATH")


@dataclass
class SplitPlanItem:
    t1: float
    t2: float
    shot_idx: int
    note: str = ""


@dataclass
class SplitReport:
    source: str
    duration_s: float
    fps: float
    n_shots: int
    clips: list[str] = field(default_factory=list)
    flagged: dict[str, str] = field(default_factory=dict)   # path -> reason
    skipped_shots: list[str] = field(default_factory=list)  # human lines
    dropped_for_cap: int = 0


def detect_scene_cuts(src: Path, threshold: float = DEFAULT_THRESHOLD) -> list[float]:
    """Return pts seconds of frames that START a new shot (ascending).

    Uses metadata=print:file= (research-verified format, f_metadata.c):
        frame:0    pts:123    pts_time:4.105
        lavfi.scene_score=0.523156
    The filtergraph comma needs NO escaping because we pass list-args
    (no shell) — subprocess hands the string to ffmpeg verbatim.
    """
    with tempfile.TemporaryDirectory() as td:
        # The file= option lives INSIDE the filtergraph, where ':' is the
        # option separator — a Windows absolute path (C:\...) silently
        # truncates the option and detection returns nothing. Run with
        # cwd=tempdir and a RELATIVE filename instead of fighting two levels
        # of escaping. The backslash before the comma is FILTERGRAPH-level
        # escaping (the graph parser splits filters on ',' without respecting
        # parentheses) — required even with list-args/no shell; the unescaped
        # form fails with "No such filter: '0.4)'".
        cmd = [
            _ffmpeg_exe(), "-hide_banner", "-nostdin",
            "-i", str(src),
            "-vf", f"select=gt(scene\\,{threshold}),"
                   f"metadata=print:file=scores.txt",
            "-f", "null", "-",
        ]
        r = subprocess.run(cmd, cwd=td, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE, timeout=1800)
        if r.returncode != 0:
            tail = r.stderr.decode(errors="replace")[-400:]
            raise RuntimeError(f"ffmpeg scene detection failed: {tail}")
        cuts: list[float] = []
        scores_file = Path(td) / "scores.txt"
        if scores_file.exists():
            for line in scores_file.read_text(errors="replace").splitlines():
                m = re.search(r"pts_time:([0-9.]+)", line)
                if m:
                    cuts.append(float(m.group(1)))
    return sorted(set(cuts))


def plan_clips(duration_s: float, cuts: list[float], fps: float,
               min_clip_s: float = MIN_CLIP_S,
               max_clip_s: float = MAX_CLIP_S,
               max_total: int = DEFAULT_MAX_TOTAL,
               ) -> tuple[list[SplitPlanItem], list[str], int]:
    """Turn shot boundaries into a clip plan.

    - shots shorter than min_clip_s are skipped (recorded with reason)
    - shots within [min, max] become one whole clip
    - longer shots yield up to MAX_CHUNKS_PER_SHOT chunks of
      TARGET_CHUNK_S, evenly SPACED across the shot (varied moments —
      consecutive chunks of one static shot add nothing per the brief)
    - if the plan exceeds max_total, it is evenly subsampled so coverage
      stays spread across the whole source (no silent truncation: the
      dropped count is returned)
    """
    frame = 1.0 / max(fps, 1.0)
    bounds = [0.0] + [c for c in cuts if 0.0 < c < duration_s] + [duration_s]
    plan: list[SplitPlanItem] = []
    skipped: list[str] = []
    for k in range(len(bounds) - 1):
        # Trim 2 frames off the shot end: keeps the next shot's lead-in
        # and any flash/transition frame out of the clip.
        a, b = bounds[k], bounds[k + 1] - 2 * frame
        length = b - a
        if length < min_clip_s:
            skipped.append(f"shot {k} ({a:.1f}-{b:.1f}s): {length:.1f}s < "
                           f"{min_clip_s:.0f}s minimum")
            continue
        if length <= max_clip_s:
            plan.append(SplitPlanItem(a, b, k))
            continue
        n = min(MAX_CHUNKS_PER_SHOT, max(1, int(length / TARGET_CHUNK_S)))
        # Even spacing: chunk i starts at a + i * stride, where stride
        # spreads n chunks across the full shot.
        stride = (length - TARGET_CHUNK_S) / max(n - 1, 1)
        for i in range(n):
            t1 = a + i * stride
            plan.append(SplitPlanItem(
                t1, min(t1 + TARGET_CHUNK_S, b), k,
                note=f"chunk {i + 1}/{n} of long shot"))
    dropped = 0
    if len(plan) > max_total:
        dropped = len(plan) - max_total
        step = len(plan) / max_total
        plan = [plan[int(i * step)] for i in range(max_total)]
    return plan, skipped, dropped


def extract_clip(src: Path, t1: float, t2: float, dst: Path) -> None:
    """Frame-accurate re-encoded extraction of [t1, t2).

    Input-side -ss/-to + re-encode = frame-accurate (accurate_seek is
    default-on). crf 18 is visually transparent for training data and the
    musubi cache step re-encodes to latents anyway. Even-dims scale guard:
    libx264+yuv420p rejects odd dimensions (crops/rotations produce them).
    """
    cmd = [
        _ffmpeg_exe(), "-hide_banner", "-nostdin", "-y",
        "-ss", f"{t1:.3f}", "-to", f"{t2:.3f}",
        "-i", str(src),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p", "-an",
        str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=600)
    if r.returncode != 0:
        tail = r.stderr.decode(errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg extract failed for {dst.name}: {tail}")


def _middle_frame_sharpness(path: Path) -> float | None:
    """Laplacian variance of the clip's middle frame (pure numpy + PyAV).
    Used to FLAG soft clips — never to delete them."""
    try:
        import av
        import numpy as np
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            total = stream.frames or 0
            target = max(total // 2, 0)
            for i, f in enumerate(container.decode(stream)):
                if i >= target:
                    g = f.to_ndarray(format="gray").astype("float32")
                    lap = (g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1]
                           + g[:-2, 1:-1] - 4 * g[1:-1, 1:-1])
                    return float(np.var(lap))
    except Exception:
        return None
    return None


def autosplit_video(src: str | Path, out_dir: str | Path,
                    threshold: float = DEFAULT_THRESHOLD,
                    max_total: int = DEFAULT_MAX_TOTAL,
                    progress_cb=None) -> SplitReport:
    """Split one long video into training clips inside out_dir.

    Clip names: <source-stem>_s<shot>_<i>.mp4 — stable + traceable back
    to the source shot. progress_cb(frac, desc) is optional (Gradio)."""
    src = Path(src)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info = probe_video(src)
    duration, fps = info.duration_s, info.fps
    if progress_cb:
        progress_cb(0.05, f"detecting cuts in {src.name} "
                          f"({duration:.0f}s — one decode pass)")
    cuts = detect_scene_cuts(src, threshold)
    plan, skipped, dropped = plan_clips(duration, cuts, fps,
                                        max_total=max_total)
    report = SplitReport(source=str(src), duration_s=duration, fps=fps,
                         n_shots=len(cuts) + 1, skipped_shots=skipped,
                         dropped_for_cap=dropped)
    # Low-sharpness flag threshold is RELATIVE to this batch's median —
    # absolute Laplacian variance varies wildly with content/resolution,
    # so a fixed cutoff would misfire. Half the median is "notably softer
    # than this video's own norm".
    sharpness: dict[str, float] = {}
    for j, item in enumerate(plan):
        stem = re.sub(r"[^A-Za-z0-9_-]", "_", src.stem)[:48]
        dst = out_dir / f"{stem}_s{item.shot_idx}_{j:02d}.mp4"
        if progress_cb:
            progress_cb(0.1 + 0.85 * (j / max(len(plan), 1)),
                        f"extracting clip {j + 1}/{len(plan)} "
                        f"[{item.t1:.1f}-{item.t2:.1f}s]")
        extract_clip(src, item.t1, item.t2, dst)
        report.clips.append(str(dst))
        s = _middle_frame_sharpness(dst)
        if s is not None:
            sharpness[str(dst)] = s
    if sharpness:
        med = sorted(sharpness.values())[len(sharpness) // 2]
        for p, s in sharpness.items():
            if s < 0.5 * med:
                report.flagged[p] = (f"soft/blurry (sharpness {s:.0f} vs "
                                     f"batch median {med:.0f}) — review")
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: autosplit.py <video> <out_dir> [threshold]")
        sys.exit(1)
    th = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_THRESHOLD
    rep = autosplit_video(sys.argv[1], sys.argv[2], threshold=th)
    print(f"{rep.n_shots} shots -> {len(rep.clips)} clips "
          f"({len(rep.skipped_shots)} shots skipped, "
          f"{rep.dropped_for_cap} clips dropped for cap)")
    for p in rep.clips:
        flag = f"  ⚠ {rep.flagged[p]}" if p in rep.flagged else ""
        print(f"  {Path(p).name}{flag}")
    for s in rep.skipped_shots:
        print(f"  skipped: {s}")
