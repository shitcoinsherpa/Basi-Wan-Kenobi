"""#387 sliding-window restyle — the two pure-logic, CPU-verifiable pieces:
a window PLAN (frame arithmetic) and an LAB color-match (pixel post-process).

The hard core — per-step overlap injection in LATENT space inside the denoise
loop — lives in wan/text2video.py and needs the GPU; it is built on top of the
plan this module computes. Keeping the arithmetic and the color transfer here,
unit-tested, means the GPU integration only has to wire proven parts.

Scheme (Wan2GP sequential, per v2v_implementation_specs_2026-06-11):
  window=81 pixel frames, overlap=9, discard=4.
  Wan VAE is 4x temporal-compressed: pixel f -> latent (f-1)//4 + 1.
    81 -> 21 latent frames; overlap 9 -> 3 latent frames (hence latents[:, :, :3]).
  Each window after the first re-renders the `overlap` tail of the previous
  window so motion is continuous; at stitch we DROP `discard` of those
  re-rendered overlap frames (they drift worst) and keep the rest.
"""
from __future__ import annotations

from dataclasses import dataclass


def latent_frames(pixel_frames: int) -> int:
    """Wan VAE temporal compression: 4x with a +1 keyframe. 81 -> 21, 9 -> 3."""
    return (pixel_frames - 1) // 4 + 1


@dataclass
class Window:
    index: int           # 0-based window number
    src_start: int       # first source pixel-frame fed to this window (inclusive)
    src_end: int         # last source pixel-frame + 1 (exclusive); <= src_start+window
    n_overlap: int       # leading pixel frames seeded from the prev window's tail
    n_overlap_lat: int   # those frames in latent space (what the denoise injects)
    emit_start: int      # first RENDER-LOCAL frame to keep at stitch
    emit_end: int        # one past the last RENDER-LOCAL frame to keep at stitch
    emit_src_start: int  # global source index emit_start maps to
    is_final: bool       # last window: keep its tail (no discard), run to true end


def plan_windows(total_frames: int, window: int = 81, overlap: int = 9,
                 discard: int = 4) -> list[Window]:
    """Tile `total_frames` into overlapping windows (Wan2GP sequential scheme).
    Pure arithmetic — no tensors. Returns the per-window source slice + the
    exact render-local emit range to keep at stitch.

    The scheme (derived + test-locked):
      * stride = window - overlap - discard. Each non-final window renders
        `window` frames but its tail `discard` frames degrade, so they're
        dropped and the NEXT window re-renders that region with continuity.
      * window k>0 SEEDS its first `overlap` frames from the previous window's
        kept tail (latent injection, in text2video.py); those seed frames are
        dropped at stitch (we keep the previous window's copy), so emit_start
        = overlap. Window 0 has no seed: emit_start = 0.
      * non-final emit_end = window - discard; the FINAL window keeps its whole
        tail (emit_end = render length) so the clip reaches `total_frames`.
      * concatenating every window's [emit_start:emit_end] covers [0, total)
        exactly once — no gap, no double-count (asserted in tools/_smoke_sliding).
    """
    if total_frames <= 0:
        return []
    if overlap >= window:
        raise ValueError(f"overlap {overlap} must be < window {window}")
    if discard < 0 or overlap < 0:
        raise ValueError("overlap and discard must be >= 0")
    if window - overlap - discard <= 0:
        raise ValueError(
            f"stride window-overlap-discard = {window-overlap-discard} must be > 0")
    stride = window - overlap - discard
    wins: list[Window] = []
    src_start = 0
    idx = 0
    while True:
        src_end = min(src_start + window, total_frames)
        render_len = src_end - src_start
        is_final = src_start + window >= total_frames
        emit_start = 0 if idx == 0 else overlap
        emit_end = render_len if is_final else (window - discard)
        win = Window(
            index=idx,
            src_start=src_start,
            src_end=src_end,
            n_overlap=(0 if idx == 0 else overlap),
            n_overlap_lat=(0 if idx == 0 else latent_frames(overlap)),
            emit_start=emit_start,
            emit_end=emit_end,
            emit_src_start=src_start + emit_start,
            is_final=is_final,
        )
        wins.append(win)
        if is_final:
            break
        src_start += stride
        idx += 1
    return wins


def stitch_ranges(wins: list[Window], total_frames: int) -> list[tuple[int, int, int]]:
    """Given the plan, return (window_index, render_local_start, n_frames) tuples
    to concatenate, in order, to cover [0, total_frames) once. Worker-side
    assembly of the final clip."""
    out = []
    for w in wins:
        n = w.emit_end - w.emit_start
        if n > 0:
            out.append((w.index, w.emit_start, n))
    return out


def orchestrate(source, plan, gen_window, *, overlap: int = 9,
                color_match: float = 1.0, emit_cb=None):
    """Pure sliding-window restyle orchestration — the GPU-free control flow, so
    it's unit-testable with a MOCK gen_window before any model runs.

    source:     (3, total_F, H, W) in [-1,1] — the full uploaded clip.
    plan:       plan_windows(total_F) output.
    gen_window: callable(src_slice, overlap_pixels, window) -> styled
                (3, render_len, H, W). overlap_pixels is None for window 0,
                else the previous window's styled tail (3, overlap, H, W) — the
                denoise loop re-anchors its leading frames to it for continuity.
    Returns the stitched (3, total_F, H, W), covering [0, total_F) exactly once
    (the plan guarantees no gap/overlap; asserted here).

    Per window: generate → LAB color-match to the previous seam (kills residual
    hue/exposure drift) → keep [emit_start, emit_end). The next window's
    overlap_pixels = the last `overlap` frames of THIS window's kept region (the
    frames it re-renders), so the seam is continuous in both motion and colour.
    """
    import torch
    total = source.shape[1]
    kept, prev_tail, prev_ref = [], None, None
    for w in plan:
        styled = gen_window(source[:, w.src_start:w.src_end], prev_tail, w)
        if prev_ref is not None and color_match > 0:
            styled = match_and_blend_colors(styled, prev_ref, color_match)
        emit = styled[:, w.emit_start:w.emit_end]
        kept.append(emit)
        if emit_cb is not None:
            emit_cb(w, int(emit.shape[1]))
        prev_tail = None if w.is_final else emit[:, -overlap:]
        prev_ref = emit[:, -1]
    out = torch.cat(kept, dim=1)
    assert out.shape[1] == total, f"stitched {out.shape[1]} != total {total}"
    return out


def match_and_blend_colors(frames, ref_frame, strength: float = 1.0):
    """LAB Reinhard color transfer, blended by `strength`. Re-aligns `frames`'
    per-channel colour statistics to `ref_frame` (the last RENDERED frame of the
    previous window) so windows don't drift in hue/exposure at the seam.

    frames:    (3, F, H, W) float in [-1, 1]  (the model's pixel convention).
    ref_frame: (3, H, W)    float in [-1, 1].
    strength:  0 = passthrough, 1 = full match. Blended in LAB then converted back.

    Pure torch (no kornia/cv2 dependency) — RGB<->LAB via the standard sRGB->XYZ
    ->CIELAB pipeline so it runs anywhere the model already runs. Reinhard et al.
    2001 'Color Transfer between Images' is the per-channel mean/std method.
    """
    import torch
    if strength <= 0.0:
        return frames
    f01 = (frames.clamp(-1, 1) + 1.0) * 0.5          # (3,F,H,W) in [0,1]
    r01 = (ref_frame.clamp(-1, 1) + 1.0) * 0.5       # (3,H,W)
    lab_f = _rgb_to_lab(f01)                          # (3,F,H,W)
    lab_r = _rgb_to_lab(r01.unsqueeze(1))            # (3,1,H,W)
    # per-channel stats over spatial (+temporal for source) dims
    m_f = lab_f.mean(dim=(1, 2, 3), keepdim=True)
    s_f = lab_f.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    m_r = lab_r.mean(dim=(1, 2, 3), keepdim=True)
    s_r = lab_r.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-5)
    matched = (lab_f - m_f) / s_f * s_r + m_r
    lab_out = lab_f + strength * (matched - lab_f)    # blend in LAB
    out01 = _lab_to_rgb(lab_out).clamp(0, 1)
    return out01 * 2.0 - 1.0                          # back to [-1,1]


# --- sRGB <-> CIELAB (D65), pure torch, operating on a leading-3-channel tensor ---
def _rgb_to_lab(rgb):
    import torch
    # rgb: (3, ...) in [0,1]; vectorized over all trailing dims.
    def _lin(c):
        return torch.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    r, g, b = _lin(rgb[0]), _lin(rgb[1]), _lin(rgb[2])
    x = r * 0.4124 + g * 0.3576 + b * 0.1805
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = r * 0.0193 + g * 0.1192 + b * 0.9505
    # normalize by D65 white
    x = x / 0.95047; z = z / 1.08883
    def _f(t):
        d = 6.0 / 29.0
        return torch.where(t > d ** 3, t ** (1.0 / 3.0), t / (3 * d * d) + 4.0 / 29.0)
    fx, fy, fz = _f(x), _f(y), _f(z)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.stack([L, a, bb], dim=0)


def _lab_to_rgb(lab):
    import torch
    L, a, bb = lab[0], lab[1], lab[2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - bb / 200.0
    def _fi(t):
        d = 6.0 / 29.0
        return torch.where(t > d, t ** 3, 3 * d * d * (t - 4.0 / 29.0))
    x = _fi(fx) * 0.95047
    y = _fi(fy)
    z = _fi(fz) * 1.08883
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570
    def _gam(c):
        c = c.clamp(0, 1)
        return torch.where(c > 0.0031308, 1.055 * (c ** (1.0 / 2.4)) - 0.055, 12.92 * c)
    return torch.stack([_gam(r), _gam(g), _gam(b)], dim=0)
