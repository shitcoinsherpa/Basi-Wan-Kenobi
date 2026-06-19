"""Video dataset validator + bucketing for Wan2.2 LoRA training."""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

# Wan VAE temporal stride = 4 → frame counts must be 4n+1
VALID_FRAME_COUNTS = (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81)

# Standard Wan2.2 resolution buckets (from wan/configs/__init__.py SUPPORTED_SIZES)
SUPPORTED_RESOLUTIONS = (
    (480, 832), (832, 480),       # 480p
    (704, 1024), (1024, 704),     # 720p~
    (704, 1280), (1280, 704),     # 720p alt
    (720, 1280), (1280, 720),     # 720p
)


@dataclass
class ClipInfo:
    path: str
    width: int
    height: int
    frame_count: int  # raw source frame count
    fps: float
    duration_s: float
    caption_path: str | None = None
    caption_text: str | None = None
    valid: bool = True
    issues: list[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []


def _probe_video_pyav(path: Path) -> ClipInfo:
    """PyAV-based probe — no external binary required."""
    import av
    container = av.open(str(path))
    try:
        if not container.streams.video:
            raise RuntimeError("no video stream")
        stream = container.streams.video[0]
        w = stream.codec_context.width
        h = stream.codec_context.height
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        if container.duration:
            dur = container.duration / 1_000_000.0
        elif stream.duration and stream.time_base:
            dur = float(stream.duration * stream.time_base)
        else:
            dur = 0.0
        n_frames = stream.frames or int(round(fps * dur))
        return ClipInfo(str(path), w, h, n_frames, fps, dur)
    finally:
        container.close()


def probe_video(path: Path) -> ClipInfo:
    """Extract video metadata. Prefers PyAV (no external binary needed) and
    falls back to ffprobe binary on PATH.

    [2026-06-09] Windows Pinokio installs don't ship ffprobe; PyAV (the `av`
    package, already a hard dep) covers the same metadata fields.
    """
    try:
        return _probe_video_pyav(path)
    except Exception as pyav_err:
        if not shutil.which("ffprobe"):
            return ClipInfo(str(path), 0, 0, 0, 0, 0, valid=False,
                            issues=[f"PyAV probe failed: {pyav_err}; "
                                    f"ffprobe not on PATH"])
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
        "-of", "json", str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.PIPE, timeout=10)
        info = json.loads(out)
        stream = info["streams"][0]
        w, h = stream["width"], stream["height"]
        num, den = (int(x) for x in stream["r_frame_rate"].split("/"))
        fps = num / den if den else 0
        dur = float(stream.get("duration", 0))
        try:
            n_frames = int(stream["nb_frames"])
        except (KeyError, ValueError):
            n_frames = int(round(fps * dur))
        return ClipInfo(str(path), w, h, n_frames, fps, dur)
    except Exception as e:
        return ClipInfo(str(path), 0, 0, 0, 0, 0, valid=False, issues=[f"probe failed: {e}"])


def nearest_valid_frames(n: int, target_count: int = 81) -> int:
    """Snap to closest valid 4n+1 not exceeding source."""
    if n <= 0:
        return 1
    # Largest valid count ≤ both n and target
    upper = min(n, target_count)
    valid = [c for c in VALID_FRAME_COUNTS if c <= upper]
    return max(valid) if valid else 1


def nearest_supported_resolution(w: int, h: int) -> tuple[int, int]:
    """Find closest supported (W, H) bucket by aspect ratio + area."""
    if w == 0 or h == 0:
        return (832, 480)
    src_ar = w / h
    src_area = w * h
    best = None
    best_score = float("inf")
    for bw, bh in SUPPORTED_RESOLUTIONS:
        ar = bw / bh
        area = bw * bh
        # Combined penalty: aspect ratio mismatch + area ratio
        ar_pen = abs(ar - src_ar) / max(ar, src_ar)
        area_pen = abs(area - src_area) / max(area, src_area)
        score = ar_pen + 0.3 * area_pen
        if score < best_score:
            best_score = score
            best = (bw, bh)
    return best


def validate_clip(clip: ClipInfo, target_frames: int = 81) -> ClipInfo:
    """Annotate clip with validation issues."""
    if not clip.valid:
        return clip

    snapped_frames = nearest_valid_frames(clip.frame_count, target_frames)
    target_w, target_h = nearest_supported_resolution(clip.width, clip.height)

    if clip.frame_count < 17:
        clip.issues.append(f"only {clip.frame_count} frames; min recommended 17 (~1s @16fps)")
    if clip.fps < 8:
        clip.issues.append(f"low fps {clip.fps:.1f}; will produce stuttering video")
    if snapped_frames < clip.frame_count // 2:
        clip.issues.append(f"frame snap would drop {clip.frame_count - snapped_frames} frames")
    if abs((clip.width / max(clip.height, 1)) - (target_w / target_h)) > 0.2:
        clip.issues.append(f"aspect ratio {clip.width}x{clip.height} far from bucket {target_w}x{target_h}")

    return clip


def find_captions(clip: ClipInfo) -> ClipInfo:
    """Look for matching .txt caption next to the video file."""
    p = Path(clip.path)
    txt = p.with_suffix(".txt")
    if txt.exists():
        clip.caption_path = str(txt)
        clip.caption_text = txt.read_text(encoding="utf-8").strip()
    return clip


def extract_thumbnail(clip_path: Path, dst: Path, time_s: float = 0.0) -> Path | None:
    """T4.F: extract a single frame as a JPEG thumbnail via ffmpeg.

    Returns the dst path on success, or None on failure (caller logs).
    Used by the gym's dataset table to display per-clip posters.
    """
    import subprocess
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        # -frames:v 1 = one frame; -q:v 4 = JPEG quality knob (1=best, 31=worst)
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(time_s), "-i", str(clip_path),
             "-frames:v", "1", "-q:v", "4", str(dst)],
            check=True, capture_output=True, timeout=10,
        )
        return dst if dst.exists() else None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


def trigger_word_coverage(clips: Iterable[ClipInfo], trigger: str) -> tuple[int, int]:
    """T4.F: count how many captions contain the trigger word.

    Returns (n_covered, n_total). Caller decides threshold (e.g. <80% = warn).
    """
    if not trigger:
        return 0, 0
    trigger = trigger.strip().lower()
    if not trigger:
        return 0, 0
    n_total = 0
    n_covered = 0
    for c in clips:
        if not c.caption_text:
            continue
        n_total += 1
        if trigger in c.caption_text.lower():
            n_covered += 1
    return n_covered, n_total


def trigger_word_coverage_fast(dataset_dir: str | Path, trigger: str) -> tuple[int, int]:
    """T4.F fast path — read .txt sidecars directly without ffprobe.

    Used by the gym's trigger_word.change() handler where we don't need full
    clip metadata, just caption text. ffprobe per-keystroke would be O(N) and
    block the UI for big datasets.
    """
    if not trigger:
        return 0, 0
    trigger = trigger.strip().lower()
    if not trigger:
        return 0, 0
    ds = Path(dataset_dir)
    if not ds.exists():
        return 0, 0
    n_total = 0
    n_covered = 0
    for txt in ds.glob("*.txt"):
        # only count if a sibling video exists (skip orphan .txt)
        if not any(txt.with_suffix(ext).exists()
                   for ext in (".mp4", ".mov", ".webm", ".mkv", ".avi")):
            continue
        n_total += 1
        if trigger in txt.read_text(encoding="utf-8", errors="ignore").lower():
            n_covered += 1
    return n_covered, n_total


def bucket_distribution(clips: Iterable[ClipInfo]) -> dict[tuple, int]:
    """Count clips per (resolution_bucket, frame_count_snapped) bucket."""
    buckets: dict[tuple, int] = {}
    for c in clips:
        if not c.valid:
            continue
        res = nearest_supported_resolution(c.width, c.height)
        frames = nearest_valid_frames(c.frame_count)
        key = (res, frames)
        buckets[key] = buckets.get(key, 0) + 1
    return buckets


def scan_dataset(dataset_dir: str | Path, target_frames: int = 81) -> list[ClipInfo]:
    """Walk dataset_dir, probe each video, validate, attach captions.

    [2026-06-09] Removed hard-fail-if-no-ffprobe precheck. probe_video()
    now uses PyAV first (no external binary needed), falls back to ffprobe
    binary only if PyAV can't parse the file.
    """
    dataset_dir = Path(dataset_dir)
    exts = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
    clips = []
    for p in sorted(dataset_dir.iterdir()):
        if p.suffix.lower() not in exts:
            continue
        clip = probe_video(p)
        clip = validate_clip(clip, target_frames=target_frames)
        clip = find_captions(clip)
        clips.append(clip)
    return clips


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: dataset.py <dataset_dir>")
        sys.exit(1)
    clips = scan_dataset(sys.argv[1])
    print(f"Scanned {len(clips)} clips:")
    for c in clips:
        cap = f"[+caption]" if c.caption_text else "[NO CAPTION]"
        issues = f"  ISSUES: {c.issues}" if c.issues else ""
        print(f"  {c.path}  {c.width}x{c.height} @{c.fps:.1f}fps × {c.frame_count}f  {cap}{issues}")
    print(f"\nBucket distribution:")
    for k, v in bucket_distribution(clips).items():
        print(f"  {k}: {v} clips")
