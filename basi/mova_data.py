"""MOVA (joint audio-video) dataset construction for the training Gym.

The MOVA/A-V side of the gym, parallel to the Wan2.2 video-only path (basi/autosplit.py +
basi/dataset.py). It DIFFERS because MOVA learns audio<->video correspondence, and MOVA's
VideoAudioDataset ingests A/V in a way that forces the dataset to pre-bake sync/fps/loudness:

  audio_len = num_frames / video_fps   # loader TRUSTS the declared fps blindly (wrong fps ->
                                       # progressive A/V drift + a silent zero-pad tail the
                                       # model learns as motion->silence)
  mono = channels.mean()               # NAIVE downmix (phase-cancellation risk)
  (no loudness normalization; no min-clip-length guard)

So clips MUST be authored at a fixed CFR fps with embedded, synced, 48kHz mono, loudness-
normalized audio. Output is MOVA's native format: metadata.json + videos/*.mp4 (audio embedded).

Defaults are research-backed (MOVA loader source + MMAudio / HunyuanVideo-Foley curation +
A/V-sync literature). Shot-boundary detection + the clip plan are REUSED from
basi/autosplit.py (one detector for both paths).

Engine: the ffmpeg binary bundled by imageio-ffmpeg (same as autosplit).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from basi.autosplit import _ffmpeg_exe, detect_scene_cuts, plan_clips
# Canonical MOVA A/V caption prompt lives in basi/caption.py (the captioning home); aliased here
# for callers of the dataset module.
from basi.caption import MOVA_AV_PROMPT_TEMPLATE as MOVA_STYLE_CAPTION_PROMPT  # noqa: F401

# ---- MOVA A/V defaults -------------------------------------
DEFAULT_FPS = 24            # match the MOVA checkpoint; loader default video_fps=24
DEFAULT_SR = 48000          # MOVA DAC sample rate
DEFAULT_LUFS = -18.0        # EBU R128 integrated-loudness target (consistent dataset levels)
# Stop-motion / 4:3 TV source -> a 4:3 MOVA target avoids the heavy crop a 16:9 target would
# inflict. Both dims % 16 == 0 (MOVA height/width_division_factor = vae_spatial(8)*2 = 16).
MOVA_RES_4_3 = (240, 320)   # (height, width) 4:3 landscape
MOVA_RES_16_9 = (240, 416)  # (height, width) 16:9 alt
# Clips a touch longer than the video-only 2-5s norm so a foley event AND its on-screen cause
# co-occur in-window for the bridge to bind. min 3.5s >= 81f@24fps (the comfortable train len).
MOVA_MIN_CLIP_S = 3.5
MOVA_MAX_CLIP_S = 6.0

def extract_clip_av(src: Path, t1: float, t2: float, dst: Path,
                    fps: int = DEFAULT_FPS, sr: int = DEFAULT_SR,
                    height: int = MOVA_RES_4_3[0], width: int = MOVA_RES_4_3[1],
                    lufs: float = DEFAULT_LUFS) -> None:
    """Frame-accurate A/V clip extraction for MOVA: CFR fps + embedded synced audio.

    Unlike autosplit.extract_clip (which uses -an), this KEEPS audio and bakes in everything
    MOVA's loader assumes: constant fps (so num_frames/fps == wall-clock audio -> no drift),
    48kHz MONO (proper fold via aresample, not the loader's naive average), loudness-normalized
    (loader does none), and scale+center-crop to the MOVA target (preserves content for any
    source aspect). Input-side -ss/-to + re-encode = frame-accurate (accurate_seek default-on);
    -ss seeks BOTH streams so the clip's audio starts at t1 too.
    """
    vf = (f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},"
          "setpts=PTS-STARTPTS")   # rebase video to pts 0
    dur = max(0.0, t2 - t1)
    cmd = [
        _ffmpeg_exe(), "-hide_banner", "-nostdin", "-y",
        # input-side -ss (fast, accurate_seek default) + -t DURATION. NOT -to: with input-side
        # -ss, -to mis-times the cut (observed: a 3.5s request -> 4.958s video). -t is exact.
        "-ss", f"{t1:.3f}", "-i", str(src), "-t", f"{dur:.3f}",
        "-vf", vf,
        "-r", str(fps), "-vsync", "cfr",
        "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
        # audio: loudness-normalize, then fold+resample to mono 48kHz (aresample does a proper
        # downmix, not a naive channel average -> dodges phase cancellation). first_pts=0 forces
        # the audio to start at pts 0 (source clips can have a >0 audio start_time -> MOVA's
        # loader would otherwise left-pad silence and DESYNC audio from video).
        "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11,aresample={sr}:first_pts=0,asetpts=PTS-STARTPTS",
        "-ac", "1", "-ar", str(sr), "-c:a", "aac", "-b:a", "128k",
        # rebase BOTH streams so the clip starts at t=0 with audio<->video aligned.
        "-avoid_negative_ts", "make_zero",
        str(dst),
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=600)
    if r.returncode != 0:
        tail = r.stderr.decode(errors="replace")[-400:]
        raise RuntimeError(f"ffmpeg A/V extract failed for {dst.name}: {tail}")


def audio_content_flags(clip_path: Path, sr: int = DEFAULT_SR,
                        silence_db: float = -35.0) -> dict:
    """FLAG a clip's audio for human curation (never auto-delete -- mirrors autosplit's
    sharpness-flag philosophy). Returns {duration, mean_db, silence_ratio, flags[]}.

    Curation policy (the A/V divergence): DROP near-silent clips (loader zero-pads -> teaches
    motion->silence) and music-dominated clips (audio tower would memorize the show's soundtrack
    as 'style'). This function reliably detects SILENCE/loudness via ffmpeg; music-vs-dialogue
    is left to human review (a robust music classifier is a heavier dep -- TODO). Flags:
      'silent'    : >80% of the clip is below silence_db -> DROP
      'quiet'     : low mean loudness -> review
      'no_audio'  : clip has no audio stream -> DROP (MOVA needs audio)
    """
    flags: list[str] = []
    cmd = [
        _ffmpeg_exe(), "-hide_banner", "-nostdin",
        "-i", str(clip_path),
        "-af", f"silencedetect=noise={silence_db}dB:d=0.2,volumedetect",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=300)
    err = r.stderr.decode(errors="replace")
    # total duration
    dur = 0.0
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", err)
    if m:
        dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    # mean volume (dB); absent if no audio stream
    mean_db = None
    mv = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", err)
    if mv:
        mean_db = float(mv.group(1))
    else:
        flags.append("no_audio")
    # silence ratio = sum of silence_duration / total
    sil = sum(float(x) for x in re.findall(r"silence_duration:\s*([\d.]+)", err))
    silence_ratio = (sil / dur) if dur > 0 else 0.0
    if silence_ratio > 0.80:
        flags.append("silent")
    if mean_db is not None and mean_db < -30.0:
        flags.append("quiet")
    return {"duration": round(dur, 3), "mean_db": mean_db,
            "silence_ratio": round(silence_ratio, 3), "flags": flags}


# ---- Curation: audio class / motion / dedup / training schedule ----------------------------
# Research-backed: dependency-light, cross-platform, FLAG-not-delete. Every numeric threshold
# tagged "(calibrate)" is a starting default (GUESS until measured on a real corpus) --
# conservative so it FLAGS rather than silently drops. All deps are imported defensively: a
# missing dep degrades to "unavailable" and never crashes the dataset build (mirrors autosplit's
# flag philosophy).

def classify_audio_content(clip_path: Path, sr: int = 16000) -> dict:
    """Heuristic speech/music/SFX label for a clip's audio (librosa spectral features, no model
    download, no GPU). Returns {label, music_score, speech_score, hzcrr, harm_ratio, flatness,
    available}. Discriminators:
      - HZCRR (high zero-crossing-rate ratio): speech alternates voiced/unvoiced -> HZCRR ~0.15;
        music is steady -> ~0.05. The ZCR *dynamics*, not the mean, separate speech from music.
      - HPSS harmonic ratio (librosa.effects.hpss): music = sustained harmonic content (high);
        SFX = percussive/noisy (low).
      - spectral flatness: tonal (speech/music) low; noisy (SFX) high.
    Thresholds are calibration starting points; this FLAGS music-dominated clips for review (the
    audio tower can otherwise memorize a soundtrack as 'style'). A YAMNet-ONNX upgrade (onnxruntime
    + 16MB model) is the accurate path if heuristics misfire on a given corpus. available=False
    (librosa absent / too-short / no audio) -> caller simply skips this flag."""
    try:
        import librosa
        import numpy as np
    except Exception:
        return {"label": "unknown", "music_score": None, "available": False}
    try:
        y, _ = librosa.load(str(clip_path), sr=sr, mono=True)
    except Exception:
        return {"label": "unknown", "music_score": None, "available": False}
    import numpy as np
    if y.size < sr // 2:                       # < 0.5s of audio -> not classifiable
        return {"label": "unknown", "music_score": None, "available": False}
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    hzcrr = float(np.mean(zcr > 1.5 * np.mean(zcr))) if zcr.size else 0.0
    h, p = librosa.effects.hpss(y)
    he, pe = float(np.sum(h ** 2)), float(np.sum(p ** 2))
    harm_ratio = he / (he + pe + 1e-9)
    flat = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    # Soft scores in [0,1]: music = harmonic AND steady (low HZCRR); speech = HZCRR-driven, tonal.
    music_score = harm_ratio * (1.0 - min(hzcrr / 0.15, 1.0))
    speech_score = min(hzcrr / 0.15, 1.0) * (1.0 - flat)
    if music_score >= 0.6 and hzcrr < 0.08:    # (calibrate)
        label = "music"
    elif hzcrr >= 0.12:                         # (calibrate)
        label = "speech"
    elif flat > 0.3 and harm_ratio < 0.5:      # (calibrate)
        label = "sfx"
    else:
        label = "mixed"
    return {"label": label, "music_score": round(music_score, 3),
            "speech_score": round(speech_score, 3), "hzcrr": round(hzcrr, 3),
            "harm_ratio": round(harm_ratio, 3), "flatness": round(flat, 3), "available": True}


def clip_motion_score(clip_path: Path, max_frames: int = 48, downscale: int = 64) -> float | None:
    """Mean inter-frame luma difference (0-255 scale) over the clip via PyAV+numpy (NO opencv).
    Low value = near-static (too little motion to teach). 'static' flag default: < ~2.0 gray-
    levels (calibrate). Returns None if PyAV/numpy absent or decode fails (caller skips flag)."""
    try:
        import av
        import numpy as np
    except Exception:
        return None
    try:
        prev = None
        diffs: list[float] = []
        n = 0
        with av.open(str(clip_path)) as c:
            s = c.streams.video[0]
            for f in c.decode(s):
                g = f.to_ndarray(format="gray")
                step_r = max(1, g.shape[0] // downscale)
                step_c = max(1, g.shape[1] // downscale)
                g = g[::step_r, ::step_c].astype("float32")     # cheap stride-downscale
                if prev is not None and prev.shape == g.shape:
                    diffs.append(float(np.mean(np.abs(g - prev))))
                prev = g
                n += 1
                if n >= max_frames:
                    break
        return round(sum(diffs) / len(diffs), 3) if diffs else None
    except Exception:
        return None


def pick_style_frame(clip_paths: list, out_png, max_clips: int = 40,
                     frames_per_clip: int = 3) -> str | None:
    """Pick the most TEXTURE-RICH frame across the training clips and save it as the IP-Adapter
    STYLE source (the Gym writes this as <workspace>/style.png at train time).

    WHY: for SDXL+IP-Adapter style transfer, a close, high-detail frame transfers a niche style
    (claymation) far better than a wide establishing shot. Texture/detail
    (clay grain, facial modeling) correlates with sharpness, so we score each sampled frame by the
    VARIANCE OF ITS LAPLACIAN (classic focus/detail measure) gated by enough overall contrast (skip
    flat title/fade frames). The highest-scoring frame is typically a close character shot — the
    ideal style anchor. Distinct from the I2V ref.png (which optimizes first-frame anchoring).

    PyAV + numpy (same stack as clip_motion_score; no opencv). Returns the saved path or None."""
    try:
        import av
        import numpy as np
        from PIL import Image
    except Exception:
        return None
    best_score = -1.0
    best_rgb = None
    clips = [Path(c) for c in clip_paths][:max_clips]
    for cp in clips:
        try:
            with av.open(str(cp)) as c:
                s = c.streams.video[0]
                total = s.frames or 0
                # sample frames spread across the clip, skipping the very first (often a title/fade)
                want = set()
                if total > 0:
                    for k in range(1, frames_per_clip + 1):
                        want.add(min(total - 1, int(total * k / (frames_per_clip + 1))))
                idx = 0
                grabbed = 0
                for f in c.decode(s):
                    take = (idx in want) if total > 0 else (idx % 10 == 5)
                    if take:
                        rgb = f.to_ndarray(format="rgb24").astype("float32")
                        g = rgb.mean(axis=2)
                        # Reject near-flat (title/fade) AND text-heavy/credits frames. Credits are
                        # near-MONOCHROME (bright text on dark) -> low color saturation; real clay
                        # scenes are colorful. Sharpness (Laplacian var) alone is fooled by letter
                        # edges, so GATE on color saturation = max-min across RGB per pixel, mean.
                        mx = rgb.max(axis=2); mn = rgb.min(axis=2)
                        sat = float((mx - mn).mean())
                        if float(g.std()) < 18.0 or sat < 28.0:   # flat OR low-color (text/credits)
                            pass
                        else:
                            # Laplacian = 4*center - 4-neighbors (numpy slicing); var = detail/sharpness
                            lap = (4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1]
                                   - g[1:-1, :-2] - g[1:-1, 2:])
                            # weight texture by colorfulness so a colorful clay close-up beats a
                            # sharp-but-drab graphic; keeps the metric a STYLE proxy, not pure edges.
                            score = float(lap.var()) * (sat / 64.0)
                            if score > best_score:
                                best_score, best_rgb = score, rgb.astype("uint8")
                        grabbed += 1
                        if grabbed >= frames_per_clip:
                            break
                    idx += 1
        except Exception:
            continue
    if best_rgb is None:
        return None
    Image.fromarray(best_rgb).save(str(out_png))
    return str(out_png)


def frame_phashes(clip_path: Path, n: int = 5) -> list:
    """n evenly-spaced frame perceptual hashes (pHash, 64-bit via imagehash) for dedup. Returns []
    if imagehash/PyAV/PIL absent or decode fails. pHash chosen: its Hamming distances are normally
    distributed -> a single threshold behaves predictably."""
    try:
        import av
        import imagehash
        from PIL import Image
    except Exception:
        return []
    try:
        with av.open(str(clip_path)) as c:
            frames = list(c.decode(c.streams.video[0]))
        if not frames:
            return []
        if len(frames) >= n:
            idxs = [int(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(n)]
        else:
            idxs = list(range(len(frames)))
        return [imagehash.phash(Image.fromarray(frames[i].to_ndarray(format="rgb24")))
                for i in idxs]
    except Exception:
        return []


def find_near_duplicates(clip_hashes: dict, max_hamming: int = 5) -> list[list[str]]:
    """Group clips whose sampled-frame pHashes match within max_hamming (64-bit: <=5 near-identical,
    ~10 similar -- pHash.org/imagededup, VERIFIED). clip_hashes = {path: [phash,...]}. Two clips are
    near-dup if the MEDIAN per-position frame distance <= max_hamming. Returns dup-groups (each a
    list of paths, >=2); caller keeps one per group. O(n^2) -- fine for the <=few-hundred-clip sets
    the gym produces."""
    import statistics
    paths = [p for p, h in clip_hashes.items() if h]
    seen: set[str] = set()
    groups: list[list[str]] = []
    for i, a in enumerate(paths):
        if a in seen:
            continue
        grp = [a]
        for b in paths[i + 1:]:
            if b in seen:
                continue
            ha, hb = clip_hashes[a], clip_hashes[b]
            m = min(len(ha), len(hb))
            if m == 0:
                continue
            dist = statistics.median(ha[k] - hb[k] for k in range(m))   # imagehash __sub__ = Hamming
            if dist <= max_hamming:
                grp.append(b)
                seen.add(b)
        if len(grp) > 1:
            seen.add(a)
            groups.append(grp)
    return groups


def plan_training_schedule(n_clips: int, target_steps: int = 2400, batch_size: int = 1,
                           min_epochs: int = 4, max_epochs: int = 20,
                           target_per_epoch: int = 150) -> dict:
    """Pick (repeats, epochs) so total steps land near target_steps for ANY dataset size -- the
    'correct for all users' size fix. total_steps = n_clips * repeats * epochs / batch (the
    kohya/musubi/diffusion-pipe formula, VERIFIED). Target ~2400 for video LoRA;
    small sets need higher repeats, large sets repeats=1. Returns
    {repeats, epochs, est_steps, per_epoch, note}."""
    n = max(1, int(n_clips))
    repeats = max(1, round(target_per_epoch / n)) if n < target_per_epoch else 1
    per_epoch = n * repeats / max(1, batch_size)
    epochs = max(min_epochs, min(max_epochs, int(round(target_steps / max(per_epoch, 1)))))
    est = int(per_epoch * epochs)
    return {"repeats": repeats, "epochs": epochs, "est_steps": est, "per_epoch": int(per_epoch),
            "note": (f"{n} clips x{repeats} repeats x{epochs} epochs = ~{est} steps "
                     f"(target ~{target_steps})")}


def write_mova_metadata(items: list[tuple[str, str]], dataset_dir: Path) -> Path:
    """Emit MOVA's metadata.json. items = [(video_path_relative_to_dataset, caption), ...].
    Layout: dataset_dir/metadata.json + dataset_dir/videos/*.mp4 (caller places the mp4s)."""
    dataset_dir = Path(dataset_dir)
    meta = [{"video_path": vp, "caption": cap or ""} for vp, cap in items]
    mp = dataset_dir / "metadata.json"
    mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return mp


def build_mova_clips(src: Path, out_videos_dir: Path, *, threshold: float = 0.4,
                     fps: int = DEFAULT_FPS, sr: int = DEFAULT_SR,
                     height: int = MOVA_RES_4_3[0], width: int = MOVA_RES_4_3[1],
                     min_clip_s: float = MOVA_MIN_CLIP_S, max_clip_s: float = MOVA_MAX_CLIP_S,
                     max_total: int = 60, lufs: float = DEFAULT_LUFS) -> list[dict]:
    """One episode -> A/V clips (mp4 with synced audio) + per-clip audio flags. Reuses
    autosplit's shot-boundary detector + clip plan; extracts with extract_clip_av. Returns a
    list of {path, t1, t2, audio:{...flags}} for the caller to curate + caption + assemble
    into a dataset via write_mova_metadata. Does NOT delete flagged clips (human curates)."""
    from basi.dataset import probe_video  # local import: avoids a hard cycle at module load
    src = Path(src)
    out_videos_dir = Path(out_videos_dir); out_videos_dir.mkdir(parents=True, exist_ok=True)
    info = probe_video(src)
    cuts = detect_scene_cuts(src, threshold=threshold)
    plan, _skipped, _dropped = plan_clips(info.duration_s, cuts, info.fps,
                                          min_clip_s=min_clip_s, max_clip_s=max_clip_s,
                                          max_total=max_total)
    results: list[dict] = []
    stem = src.stem.replace(" ", "_")
    for j, item in enumerate(plan):
        dst = out_videos_dir / f"{stem}_{j:03d}.mp4"
        extract_clip_av(src, item.t1, item.t2, dst, fps=fps, sr=sr,
                        height=height, width=width, lufs=lufs)
        audio = audio_content_flags(dst, sr=sr)
        aclass = classify_audio_content(dst)
        motion = clip_motion_score(dst)
        # Consolidate per-clip curation flags (FLAG, never auto-delete here). 'music' and 'static'
        # join the audio module's 'silent'/'no_audio'/'quiet'. drop_recommended marks the
        # unambiguous cases the optional auto-curate (curate_clip_set) will remove.
        flags = list(audio.get("flags", []))
        if aclass.get("available") and aclass.get("music_score") is not None \
                and aclass["music_score"] >= 0.70:                       # (calibrate)
            flags.append("music")
        if motion is not None and motion < 2.0:                          # (calibrate)
            flags.append("static")
        drop = any(f in flags for f in ("silent", "no_audio"))
        results.append({"path": str(dst), "t1": item.t1, "t2": item.t2,
                        "audio": audio, "audio_class": aclass, "motion": motion,
                        "flags": flags, "drop_recommended": drop})
    return results


def _episode_edges(clips: list[dict]) -> set[str]:
    """Paths of the FIRST and LAST clip of each source episode -- the positions where intro/title
    cards and end-credits live (true for ~any show). Episodes are grouped by the clip filename with
    its trailing _<index> stripped (build_mova_clips names clips '<source>_<NNN>.mp4'). Index order
    is taken from that numeric suffix so it's chronological regardless of dict order. An opening
    title shot (a title-card sign + theme music) can leak past the static/music/dedup filters --
    positional edge detection catches what content filters miss."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for c in clips:
        p = c["path"]
        stem = Path(p).stem
        head, _, tail = stem.rpartition("_")
        key, idx = (head, int(tail)) if (head and tail.isdigit()) else (stem, 0)
        groups.setdefault(key, []).append((idx, p))
    edges: set[str] = set()
    for items in groups.values():
        items.sort()
        edges.add(items[0][1])           # first clip = intro/title
        edges.add(items[-1][1])          # last clip = outro/credits
    return edges


def curate_clip_set(clips: list[dict], *, auto_curate: bool = False,
                    drop_music: bool = False, drop_static: bool = False,
                    drop_edges: bool = False,
                    dedup: bool = True, max_hamming: int = 5) -> dict:
    """Cross-clip curation over an assembled clip set (the 'correct for all users' pass). Operates
    on build_mova_clips() output. FLAG-not-delete by default: with auto_curate=False it only
    reports what WOULD be dropped (human decides); with auto_curate=True it returns the kept subset.
    Decisions:
      - always drop: clips flagged silent/no_audio (drop_recommended) -- MOVA needs real audio.
      - dedup: among near-duplicate pHash groups (Hamming<=max_hamming), keep the first, drop rest.
      - opt-in: drop_music (music>=0.70) / drop_static (motion<2.0) / drop_edges (first+last clip of
        each episode -- intro/title cards + end credits; positional, catches title shots that escape
        the content filters). All edges are ALWAYS reported (under 'edges') for audit even when not
        dropped.
    Returns {keep, drop:[{path,reason}], dedup_groups, edges:[...], summary}. Pure analysis when
    auto_curate=False -- nothing on disk is touched (caller deletes if it wants)."""
    drop: list[dict] = []
    dropped_paths: set[str] = set()
    edges = _episode_edges(clips)

    def _mark(path, reason):
        if path not in dropped_paths:
            dropped_paths.add(path)
            drop.append({"path": path, "reason": reason})

    for c in clips:
        p = c["path"]
        if c.get("drop_recommended"):
            _mark(p, "no usable audio (silent/no_audio)")
        elif drop_music and "music" in c.get("flags", []):
            _mark(p, f"music-dominated (score {c.get('audio_class', {}).get('music_score')})")
        elif drop_static and "static" in c.get("flags", []):
            _mark(p, f"near-static (motion {c.get('motion')})")
        elif drop_edges and p in edges:
            _mark(p, "intro/outro edge clip (first/last of episode -- likely title/credits)")
    # Dedup across the survivors only.
    n_groups = 0
    if dedup:
        survivors = [c for c in clips if c["path"] not in dropped_paths]
        hashes = {c["path"]: frame_phashes(Path(c["path"])) for c in survivors}
        for grp in find_near_duplicates(hashes, max_hamming=max_hamming):
            n_groups += 1
            for dup in grp[1:]:                       # keep grp[0], drop the rest
                _mark(dup, f"near-duplicate of {Path(grp[0]).name}")
    keep = [c for c in clips if c["path"] not in dropped_paths]
    summary = (f"{len(clips)} clips -> keep {len(keep)}, drop {len(drop)} "
               f"({n_groups} near-dup groups, {len(edges)} intro/outro edge clips"
               f"{' dropped' if drop_edges else ' flagged-review'}). auto_curate={auto_curate}.")
    return {"keep": keep if auto_curate else clips, "drop": drop,
            "dedup_groups": n_groups, "edges": sorted(edges), "summary": summary}
