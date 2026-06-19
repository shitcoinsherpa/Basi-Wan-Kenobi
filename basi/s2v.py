"""Wan2.2-S2V (audio-driven talking character) helpers for the optimized stack.

S2V is AUDIO-DRIVEN: it lip-syncs a reference character to a supplied voice track
(it does NOT generate sound). The DiT engine + 80-frame chunked autoregressive
generate loop live in wan/speech2video.py (WanS2V); the GGUF model load is in
tools/s2v_loader.py. This module owns the validated recipe + a centralized,
CPU-pinned audio-feature extractor so audio handling is testable without the VAE
or the DiT.

Recipe values are the vendored config (wan/configs/wan_s2v_14B.py): steps 40,
guide 4.5, shift 3.0, fps 16, motion_frames 73, drop_first_motion. S2V is NOT a
Lightning-distilled fast path — it needs real CFG + full steps for lip-sync to
bite (same lesson as VACE; do not inherit the T2V Lightning-4 defaults).
"""
from __future__ import annotations

# [S3] Validated S2V regime — single source of truth. The worker/app S2V path
# MUST pull these, never inherit the Lightning/SDEdit fast defaults.
S2V_RECIPE = {
    "sampling_steps": 40,     # wan_s2v_14B.sample_steps
    "guide_scale": 4.5,       # wan_s2v_14B.sample_guide_scale
    "shift": 3.0,             # wan_s2v_14B.sample_shift
    "fps": 16,                # wan_s2v_14B.sample_fps (audio<->frame alignment)
    "infer_frames": 80,       # per-chunk frame count (speech2video default)
    "motion_frames": 73,      # wan_s2v_14B.transformer.motion_frames (carry overlap)
    "drop_first_motion": True,
    # [S8] Cross-chunk anti-drift LAB color anchor. mean@0.5 measured strictly
    # better on real S2V output (drift 1.61->1.03 dLAB, seam 0.59->0.30) and
    # removes 85% of injected drift; auto-engages only at >= min_chunks so short
    # clips (already coherent, drift sub-JND) stay bit-identical. See
    # tools/_s2v_color_anchor_offline.py + _s2v_drift_probe.py.
    "color_anchor_strength": 0.5,
    "color_anchor_min_chunks": 4,
}
S2V_MIN_STEPS = 30            # below this, lip-sync + motion degrade
S2V_MIN_GUIDE = 3.0           # below this, audio conditioning can't bite


def check_s2v_regime(sampling_steps, guide_scale):
    """Return a loud warning if an S2V generate runs BELOW the validated regime
    (which makes lip-sync look broken), else None. Mirrors check_vace_regime."""
    if isinstance(guide_scale, (tuple, list)):
        g = min(float(x) for x in guide_scale)
    else:
        g = float(guide_scale)
    if sampling_steps < S2V_MIN_STEPS or g < S2V_MIN_GUIDE:
        return (f"S2V at steps={sampling_steps} guide={guide_scale} is BELOW the "
                f"validated regime (need steps>={S2V_MIN_STEPS}, guide>="
                f"{S2V_MIN_GUIDE}; canonical {S2V_RECIPE}). Lip-sync degrades "
                f"sharply here. S2V is not a Lightning fast path.")
    return None


def load_audio_encoder(checkpoint_dir, device="cpu"):
    """Build the wav2vec2 AudioEncoder, pinned to CPU by default.

    CPU is deliberate (and the reference WanS2V already defaults to it): once the
    GGUF DiT pipe has set the global CUDA/cuBLASLt state, F.linear inside the
    HF wav2vec2 forward trips the cuBLASLt workspace heuristic bug (the same
    256-GiB phantom alloc that bit Dinov2 depth — see basi/depth.py). Running the
    small audio encoder on CPU sidesteps it entirely and costs little (it runs
    once per generate, off the hot path). `checkpoint_dir` is the dir holding the
    `wav2vec2-large-xlsr-53-english/` subdir.
    """
    import os
    from wan.modules.s2v.audio_encoder import AudioEncoder
    model_id = os.path.join(str(checkpoint_dir), "wav2vec2-large-xlsr-53-english")
    return AudioEncoder(device=device, model_id=model_id)


def extract_audio(audio_encoder, wav_path, fps=16, infer_frames=80, m=0,
                  dtype=None):
    """Extract per-chunk audio embeddings from a wav, standalone (no WanS2V).

    Mirrors WanS2V.encode_audio's core: extract_audio_feat -> bucket to fps ->
    shape to [1, audio_dim, num_frames]. Returns (audio_emb, num_repeat) where
    num_repeat is how many infer_frames-chunks the audio spans. audio_emb stays
    on the encoder's device (CPU); the caller moves it to the DiT device.
    """
    import torch
    z = audio_encoder.extract_audio_feat(
        wav_path, dtype=(dtype if dtype is not None else torch.float32))
    bucket, num_repeat = audio_encoder.get_audio_embed_bucket_fps(
        z, fps=fps, batch_frames=infer_frames, m=m)
    bucket = bucket.unsqueeze(0)
    if bucket.dim() == 3:
        bucket = bucket.permute(0, 2, 1)        # [1, audio_dim, frames]
    elif bucket.dim() == 4:
        bucket = bucket.permute(0, 2, 3, 1)     # [1, audio_dim, layers, frames]
    return bucket, num_repeat
