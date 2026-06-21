"""FinalFrame-style continuation helpers for MOVA A/V inference. Self-contained (numpy + PIL +
pyloudnorm, all present in env_mova) so mova_sample.py can chain clips IN-PROCESS (model stays
resident across clips). Video: carry the sharpest last frame of clip N as clip N+1's I2V ref,
LAB-color-anchored to clip-0 to fight drift. Audio: per-clip LUFS normalize + linear crossfade
overlap-add at each seam to remove clicks.

Research basis: MOVA is strict first-frame I2V, so last-frame
-> next-ref chaining is the continuation path; it generates audio jointly (no prefix conditioning),
so audio continuity is a post-hoc waveform stitch (LUFS + crossfade) -> non-jarring seams, not
seamless (prosody resets per clip; true seamlessness needs audio-prefix retraining). LAB math
mirrors the proven basi/sliding.py match_and_blend_colors.
"""
import numpy as np

# ---- video: sharpest tail frame (avoid motion-blur/blink at the cut) ----------------------------

def _laplacian_var(gray):
    lap = (4.0 * gray[1:-1, 1:-1] - gray[:-2, 1:-1] - gray[2:, 1:-1]
           - gray[1:-1, :-2] - gray[1:-1, 2:])
    return float(lap.var())

def sharpest_tail_frame(frames_pil, tail=5):
    """Return (index, PIL.Image) of the sharpest of the last `tail` frames (Laplacian variance).
    The last decoded frame is often mid-blink/motion-blurred; the sharpest tail frame is a cleaner
    I2V anchor. `frames_pil` is the list of PIL frames MOVA returns for a clip."""
    n = len(frames_pil)
    cand = list(range(max(0, n - tail), n))
    best, best_i = -1.0, cand[-1]
    for i in cand:
        g = np.asarray(frames_pil[i].convert("RGB"), dtype=np.float32).mean(axis=2)
        v = _laplacian_var(g)
        if v > best:
            best, best_i = v, i
    return best_i, frames_pil[best_i]

# ---- video: LAB Reinhard color anchor (numpy; mirrors basi/sliding.match_and_blend_colors) ------

_RGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                     [0.2126, 0.7152, 0.0722],
                     [0.0193, 0.1192, 0.9505]], dtype=np.float64)
_XYZ2RGB = np.linalg.inv(_RGB2XYZ)
_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

def _srgb_to_lin(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
def _lin_to_srgb(c): return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.clip(c, 0, None) ** (1 / 2.4) - 0.055)
def _f(t): d = 6.0 / 29.0; return np.where(t > d ** 3, np.cbrt(t), t / (3 * d * d) + 4.0 / 29.0)
def _finv(t): d = 6.0 / 29.0; return np.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))

def _rgb_to_lab(rgb01):                 # rgb01: (...,3) in [0,1] -> Lab
    lin = _srgb_to_lin(rgb01)
    xyz = lin @ _RGB2XYZ.T / _WHITE
    fx, fy, fz = _f(xyz[..., 0]), _f(xyz[..., 1]), _f(xyz[..., 2])
    return np.stack([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)], axis=-1)

def _lab_to_rgb(lab):                    # lab: (...,3) -> rgb01 in [0,1]
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    xyz = np.stack([_finv(fx), _finv(fy), _finv(fz)], axis=-1) * _WHITE
    return np.clip(_lin_to_srgb(xyz @ _XYZ2RGB.T), 0.0, 1.0)

def color_match(frame_rgb_u8, ref_lab_stats, strength=0.5):
    """Reinhard-match a frame's per-channel LAB mean/std toward a reference (ref_lab_stats =
    (mean[3], std[3])), blended by `strength`. frame_rgb_u8: HxWx3 uint8. Returns HxWx3 uint8."""
    lab = _rgb_to_lab(frame_rgb_u8.astype(np.float64) / 255.0)
    m, s = lab.reshape(-1, 3).mean(0), lab.reshape(-1, 3).std(0) + 1e-6
    rm, rs = ref_lab_stats
    matched = (lab - m) / s * rs + rm
    out = lab + strength * (matched - lab)
    return (_lab_to_rgb(out) * 255.0 + 0.5).astype(np.uint8)

def lab_stats(frame_rgb_u8):
    """(mean[3], std[3]) of a frame in LAB — the anchor stats for color_match."""
    lab = _rgb_to_lab(frame_rgb_u8.astype(np.float64) / 255.0).reshape(-1, 3)
    return lab.mean(0), lab.std(0) + 1e-6

# ---- audio: LUFS normalize + linear crossfade overlap-add ---------------------------------------

def lufs_normalize(wave, sr, target_lufs=-18.0):
    """Normalize a mono float waveform to `target_lufs` (ITU-R BS.1770 via pyloudnorm). Silence /
    pathological -> unchanged. Returns float32 clipped to [-1,1]."""
    try:
        import pyloudnorm as pyln
        w = np.asarray(wave, dtype=np.float64)
        loud = pyln.Meter(sr).integrated_loudness(w)
        if not np.isfinite(loud) or loud < -70:
            return np.asarray(wave, dtype=np.float32)
        gain = 10.0 ** ((target_lufs - loud) / 20.0)
        return np.clip(w * gain, -1.0, 1.0).astype(np.float32)
    except Exception:
        return np.asarray(wave, dtype=np.float32)

def addnoise_pil(pil, sigma=0.02, seed=0):
    """Add mild Gaussian noise (~sigma in [0,1], research ~0.02) to a carried I2V ref frame to
    shrink the clean-vs-generated distribution gap, softening the continuation seam. Deterministic."""
    from PIL import Image
    a = np.asarray(pil.convert("RGB"), dtype=np.float32) / 255.0
    a = np.clip(a + np.random.RandomState(seed).randn(*a.shape).astype(np.float32) * sigma, 0.0, 1.0)
    return Image.fromarray((a * 255.0 + 0.5).astype(np.uint8))


def crossfade_video_seam(accum, new_frames, xf):
    """Linear PIXEL crossfade at a clip join to smooth the motion seam of single-frame I2V
    continuation: blend accum's last `xf` frames with new_frames' first `xf` (index-aligned ramp),
    then append the rest. Lists of PIL in, list of PIL out. Net: removes `xf` frames per join (so the
    audio crossfade must be `xf` frames to stay A/V-synced). xf>=1."""
    from PIL import Image
    if not accum:
        return list(new_frames)
    xf = max(1, min(xf, len(accum), len(new_frames)))
    out = list(accum[:-xf])
    for i in range(xf):
        w = (i + 1) / (xf + 1)                                   # ramp in (0,1), endpoints excluded
        a = np.asarray(accum[-xf + i].convert("RGB"), dtype=np.float32)
        b = np.asarray(new_frames[i].convert("RGB"), dtype=np.float32)
        out.append(Image.fromarray(((1.0 - w) * a + w * b + 0.5).astype(np.uint8)))
    out.extend(new_frames[xf:])
    return out


def crossfade_concat(waves, sr, crossfade_ms=100.0):
    """Concatenate mono waveforms with a linear crossfade overlap-add at each seam (kills the click
    from independent-chunk sample discontinuity). Each `waves[i]` is a 1-D float array. Returns the
    stitched 1-D float32. Length = sum(len) - (n-1)*W."""
    waves = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves if len(w)]
    if not waves:
        return np.zeros(0, dtype=np.float32)
    out = waves[0]
    for nxt in waves[1:]:
        W = int(min(crossfade_ms / 1000.0 * sr, len(out), len(nxt)))
        if W <= 0:
            out = np.concatenate([out, nxt]); continue
        ramp = np.linspace(0.0, 1.0, W, dtype=np.float32)
        seam = out[-W:] * (1.0 - ramp) + nxt[:W] * ramp
        out = np.concatenate([out[:-W], seam, nxt[W:]])
    return out.astype(np.float32)


def equal_power_crossfade_concat(waves, sr, crossfade_ms=80.0):
    """Concatenate mono waveforms with an EQUAL-POWER crossfade at each seam. Gains sqrt(1-t)/sqrt(t)
    keep instantaneous power ~constant across the blend (g_out^2 + g_in^2 = 1), so there's no
    mid-seam loudness dip — the failure mode of a linear fade between two uncorrelated speech
    segments. Window kept short (~80ms) so consonants aren't smeared. Returns 1-D float32;
    length = sum(len) - (n-1)*W. Preferred over crossfade_concat for speech."""
    waves = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves if len(w)]
    if not waves:
        return np.zeros(0, dtype=np.float32)
    out = waves[0]
    for nxt in waves[1:]:
        W = int(min(crossfade_ms / 1000.0 * sr, len(out), len(nxt)))
        if W <= 0:
            out = np.concatenate([out, nxt]); continue
        t = np.linspace(0.0, 1.0, W, dtype=np.float32)
        seam = out[-W:] * np.sqrt(1.0 - t) + nxt[:W] * np.sqrt(t)
        out = np.concatenate([out[:-W], seam, nxt[W:]])
    return out.astype(np.float32)


# ---- speech-aware anchor selection (sharpness + audio-quiet proxy for mouth-rest) ----------------

def _frame_audio_rms(wave, sr, fps, frame_idx, half_win_ms=60.0):
    """RMS of the audio in a window centered on frame_idx's timestamp. Low RMS => the speaker is at
    a pause there => the face is near a neutral/closed-mouth rest, which restarts I2V cleanly and
    won't fight the next audio segment's onset. Returns 0.0 when no audio is available."""
    if wave is None or sr is None:
        return 0.0
    w = np.asarray(wave, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return 0.0
    c = int((frame_idx + 0.5) / float(fps) * sr)
    h = max(1, int(half_win_ms / 1000.0 * sr))
    seg = w[max(0, c - h): c + h]
    return float(np.sqrt((seg ** 2).mean())) if seg.size else 0.0


def best_tail_frame(frames_pil, wave=None, sr=None, fps=24, tail=8, sharp_w=0.6, pause_w=0.4):
    """Speech-aware continuation anchor over the last `tail` frames: combine sharpness (reject motion
    blur — variance of Laplacian) with audio QUIETNESS at each frame's timestamp (prefer a pause, so
    the mouth is at rest and won't freeze mid-vowel at the seam). Returns (index, PIL.Image). With no
    audio it degrades to pure sharpness (== sharpest_tail_frame). Fine eye/mouth judgment is left to
    the optional Qwen3-VL pass; this is the dependency-light numeric default."""
    n = len(frames_pil)
    cand = list(range(max(0, n - tail), n))
    sharp = np.array([_laplacian_var(np.asarray(frames_pil[i].convert("RGB"),
                                                dtype=np.float32).mean(axis=2)) for i in cand])
    rms = np.array([_frame_audio_rms(wave, sr, fps, i) for i in cand])

    def _norm(x):
        rng = float(x.max() - x.min())
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    score = sharp_w * _norm(sharp) + pause_w * (1.0 - _norm(rms))   # sharp high good, audio low good
    best = int(np.argmax(score))
    return cand[best], frames_pil[cand[best]]


def rank_tail_frames(frames_pil, wave=None, sr=None, fps=24, tail=8, k=5, sharp_w=0.6, pause_w=0.4):
    """Return up to k (index, PIL.Image) tail candidates, BEST-FIRST, by the same speech-aware score
    as best_tail_frame (sharpness + audio-quiet). For surfacing a 'pick the continuation frame'
    gallery so the user can override the auto pick. With no audio it ranks by sharpness alone."""
    n = len(frames_pil)
    cand = list(range(max(0, n - tail), n))
    sharp = np.array([_laplacian_var(np.asarray(frames_pil[i].convert("RGB"),
                                                dtype=np.float32).mean(axis=2)) for i in cand])
    rms = np.array([_frame_audio_rms(wave, sr, fps, i) for i in cand])

    def _norm(x):
        rng = float(x.max() - x.min())
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    score = sharp_w * _norm(sharp) + pause_w * (1.0 - _norm(rms))
    order = list(np.argsort(score)[::-1][:k])
    return [(cand[j], frames_pil[cand[j]]) for j in order]
