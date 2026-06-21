"""VACE conditioning — build the 96-channel control latent that feeds
vace_patch_embedding (Conv3d 96->5120). Source-verified against ali-vilab/VACE
wan_vace.py (vace_encode_frames / vace_encode_masks / vace_latent). The
forward-pass integration lives in wan/modules/model.py (forward_vace + hint
injection); this module owns only the pure-tensor conditioning construction so
it can be unit-tested without the VAE or the model.

96-channel layout (EXACT, verified):
    ch  0-15 : inactive = VAE.encode(frames * (1 - mask))   16ch latent
    ch 16-31 : reactive = VAE.encode(frames * mask)         16ch latent
    ch 32-95 : mask pixel-unshuffled 8x8 -> 64ch, nearest-exact resized to F_lat

Wan2.x VAE stride = (4, 8, 8): temporal /4 (+1 keyframe), spatial /8.

For a DEPTH-control restyle (no inpainting): mask = all-ones, so
inactive = VAE.encode(zeros) and reactive = VAE.encode(depth_video); the 64
mask channels are all-ones. (ali-vilab prepare_source sets src_mask=ones when
none given.)
"""
from __future__ import annotations

VAE_STRIDE = (4, 8, 8)


def latent_t(pixel_frames: int) -> int:
    """Wan VAE temporal compression: (F + 3) // 4. 81 -> 21, 1 -> 1."""
    return (pixel_frames + 3) // VAE_STRIDE[0]


# VACE depth-control QUALITY-TIER recipe — single source of truth; the
# worker/app VACE path MUST pull these, never inherit the Lightning/SDEdit fast
# defaults. At these settings the output geometry tracks the control depth at
# 0.973 correlation (vs 0.512 with control OFF). The VACE hint is a RESIDUAL
# added onto the denoising stream, so it needs real CFG + full steps to bite; at
# the Lightning-4 / guide-1.5 fast regime the depth-lock COLLAPSES to ~0 and the
# output looks uncontrolled. Use SDEdit (s8_d6_L0.7) for FAST restyle; VACE for
# the depth-locked QUALITY tier.
VACE_DEPTH_RECIPE = {
    "sampling_steps": 50,
    "guide_scale": 5.0,
    "shift": 3.0,
    "vace_context_scale": 1.0,
}
VACE_MIN_STEPS = 40      # below this, depth-lock degrades sharply
VACE_MIN_GUIDE = 3.0     # below this, the hint can't bite the under-conditioned stream


def check_vace_regime(sampling_steps, guide_scale):
    """Return a loud warning string if a VACE depth generate is running BELOW
    the validated regime (which makes the control look broken), else None.
    Called from text2video.generate() whenever vace_video is active so a weak
    regime fails LOUD instead of silently producing uncontrolled output."""
    # guide_scale may be a per-expert tuple (high,low) or a scalar; check the
    # most conservative (smallest) value so a weak expert still trips the warn.
    if isinstance(guide_scale, (tuple, list)):
        g = min(float(x) for x in guide_scale)
    else:
        g = float(guide_scale)
    if sampling_steps < VACE_MIN_STEPS or g < VACE_MIN_GUIDE:
        return (f"VACE depth-control at steps={sampling_steps} guide={guide_scale} is "
                f"BELOW the validated regime (need steps>={VACE_MIN_STEPS}, "
                f"guide>={VACE_MIN_GUIDE}; canonical {VACE_DEPTH_RECIPE}). Depth-lock "
                f"degrades sharply here (corr ~0) and output will look uncontrolled. "
                f"Use SDEdit for fast restyle; VACE for the quality tier.")
    return None


def split_inactive_reactive(frames, mask):
    """frames: (3, F, H, W) in [-1,1]; mask: (F, H, W) or (1,F,H,W) in {0,1}.
    Returns (inactive_px, reactive_px), each (3, F, H, W), for VAE encoding.
    inactive keeps pixels OUTSIDE the mask (background to preserve); reactive
    keeps pixels INSIDE the mask (region to regenerate). Verified channel sense:
    inactive = i*(1-m), reactive = i*m."""
    import torch
    if mask.dim() == 4:
        mask = mask[0]
    m = (mask > 0.5).to(frames.dtype)          # binarize, (F,H,W)
    m = m.unsqueeze(0)                          # (1,F,H,W) broadcast over channels
    inactive = frames * (1 - m)
    reactive = frames * m
    return inactive, reactive


def build_mask_channels(mask, latent_frames: int, latent_h: int, latent_w: int,
                        temporal_reduce: str = "nearest"):
    """Pixel-unshuffle a single-channel pixel-resolution mask into 64 latent
    channels, then resample temporally to latent_frames.

    mask: (F, H, W) or (1,F,H,W) in {0,1}, with H = latent_h*8, W = latent_w*8.
    Returns (64, latent_frames, latent_h, latent_w).

    Each latent cell covers an 8x8 pixel block; the 64 channels encode that
    block's sub-pixel mask pattern (vae_stride[1]*vae_stride[2] = 8*8 = 64).
    Spatial reshape verified against vace_encode_masks (view -> permute(2,4,0,1,3)
    -> reshape).

    temporal_reduce:
      "nearest" — interpolate(nearest-exact), EXACTLY the ali-vilab reference.
        Correct for DENSE masks (depth = all-ones; spatial inpaint spanning many
        frames). KEEP for the depth path (byte-identical).
      "vae_min" — causal VAE-group MIN-pool: latent 0 <- pixel 0; latent k <- min
        over pixels [4(k-1)+1 .. 4k]. Needed for SPARSE keyframe anchors:
        nearest-exact samples only pixels ~{0,4,8,...} and silently DROPS a
        single-frame anchor sitting between them (it does upstream too — VACE was
        never trained on sparse RGB anchors). min-pool guarantees an anchor in any
        group's 4 pixels marks that latent frame as keep (mask 0=keep -> min wins).
    """
    import torch
    import torch.nn.functional as F
    if mask.dim() == 4:
        mask = mask[0]
    Fd, H, W = mask.shape
    sh, sw = VAE_STRIDE[1], VAE_STRIDE[2]
    assert H == latent_h * sh and W == latent_w * sw, (
        f"mask spatial {(H, W)} != latent {(latent_h, latent_w)} * stride {(sh, sw)}")
    m = mask.view(Fd, latent_h, sh, latent_w, sw)       # (F, Hl, 8, Wl, 8)
    m = m.permute(2, 4, 0, 1, 3)                         # (8, 8, F, Hl, Wl)
    m = m.reshape(sh * sw, Fd, latent_h, latent_w).float()   # (64, F, Hl, Wl)
    if temporal_reduce == "vae_min":
        out = m.new_empty(sh * sw, latent_frames, latent_h, latent_w)
        out[:, 0] = m[:, 0]                              # causal keyframe latent
        for k in range(1, latent_frames):
            lo, hi = 4 * (k - 1) + 1, min(4 * k + 1, Fd)
            out[:, k] = m[:, lo:hi].amin(dim=1)          # keep (0) wins in a group
        return out
    return F.interpolate(m.unsqueeze(0),
                         size=(latent_frames, latent_h, latent_w),
                         mode="nearest-exact").squeeze(0)    # (64, Fl, Hl, Wl)


def assemble_vace_latent(inactive_lat, reactive_lat, mask_64):
    """Concat the verified 96-ch order: [inactive(16) | reactive(16) | mask(64)].
    inactive_lat, reactive_lat: (16, Fl, Hl, Wl); mask_64: (64, Fl, Hl, Wl).
    Returns (96, Fl, Hl, Wl)."""
    import torch
    assert inactive_lat.shape[0] == 16 and reactive_lat.shape[0] == 16, (
        inactive_lat.shape, reactive_lat.shape)
    assert mask_64.shape[0] == 64, mask_64.shape
    mask_64 = mask_64.to(inactive_lat.dtype).to(inactive_lat.device)
    return torch.cat([inactive_lat, reactive_lat, mask_64], dim=0)


def build_keyframe_guide_and_mask(anchors, positions, total_frames, height, width):
    """Keyframe-anchored editing (Ray Modify parity). Build the VACE
    (guide, mask) so the edited anchor frames are KEPT and the rest is generated
    to match — the edit propagates. anchors: list of (3,H,W) edited frames in
    [-1,1]; positions: their frame indices in [0,total_frames). Returns:
      guide: (3, total_frames, H, W) — grey (0.0 in [-1,1], = the VACE generate
             fill) everywhere except each anchor placed at its position.
      mask:  (total_frames, H, W) in {0,1} — 0 at anchor frames (keep real
             frame), 1 elsewhere (generate). Verified vs ali-vilab / Wan2GP
             prepare_video_guide_and_mask ("0=keep, 1=generate, grey fill").
    Feed through the SAME 96-ch path as depth control (split_inactive_reactive →
    VAE encode → assemble_vace_latent): inactive=VAE(anchors), reactive=VAE(grey),
    the 64 mask channels mark which frames to regenerate. Honest ceiling ~4-6
    anchors/clip; anchors + a dense control video is a known Wan2GP bug,
    so this path is anchors-ONLY (no depth)."""
    import torch
    if len(anchors) != len(positions):
        raise ValueError(f"{len(anchors)} anchors but {len(positions)} positions")
    guide = torch.zeros(3, total_frames, height, width)   # grey = 0 in [-1,1]
    mask = torch.ones(total_frames, height, width)
    for a, p in zip(anchors, positions):
        if not (0 <= p < total_frames):
            raise ValueError(f"anchor position {p} out of [0,{total_frames})")
        if a.shape[-2:] != (height, width):
            a = torch.nn.functional.interpolate(
                a.unsqueeze(0), size=(height, width), mode="bicubic",
                align_corners=False).squeeze(0)
        guide[:, p] = a.clamp(-1, 1)
        mask[p] = 0.0
    return guide, mask


def ones_mask(frames):
    """Depth-control / full-regen mask = all-ones, matching ali-vilab
    prepare_source when src_mask is None. frames: (3,F,H,W). Returns (F,H,W)."""
    import torch
    _, Fd, H, W = frames.shape
    return torch.ones((Fd, H, W), dtype=frames.dtype, device=frames.device)
