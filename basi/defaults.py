"""basi/defaults.py -- the SINGLE SOURCE OF TRUTH for BASIWAN's user-facing default
VALUES and the one validated recipe that was previously hardcoded in app.py.

Why this exists: the UI defaults (frame counts, steps, guidance, strengths, epochs)
were scattered as 20+ magic `value=` literals across app.py, free to silently drift
from the recipes we actually measured. This module collects them so a default has ONE
home with its provenance, and the gradio sliders read from here.

It deliberately holds only PLAIN DATA (no torch/heavy imports) so it loads instantly
and is unit-testable on any interpreter. The per-domain ALGORITHM recipes already have
good homes WITH their code -- this module points at them rather than duplicating:
  - S2V_RECIPE            -> basi/s2v.py   (steps 40, guide 4.5, shift 3.0, fps 16, ...)
  - VACE_DEPTH_RECIPE     -> basi/vace.py  (steps 50, guide 5.0, shift 3.0, scale 1.0)
  - LIGHTNING/USER strength -> basi/combo.py (1.0 / 0.8)
ASCII only.
"""

# --- Restyle (SDEdit) quality recipe -------------------------------------------------
# Measured A/B 2026-06-11 (restyle_steps_ab): "s8_d6_L0.7" won -- 8 steps, denoise 0.6,
# Lightning REDUCED to 0.7 -> style_sim 0.812 + tightest structure, vs the old 4-step /
# L1.0 corner-cut at 0.750. Reducing Lightning lets the high-noise expert (global style)
# impose the look without distill dilution. Applies ONLY on the restyle path; plain T2V
# keeps Lightning 1.0. Was hardcoded in app.py (`_light_s = 0.7`, denoise slider 0.6).
RESTYLE_SDEDIT_RECIPE = {
    "steps": 8,
    "denoise": 0.6,
    "lightning_strength": 0.7,
}

# --- Studio (inference) UI defaults --------------------------------------------------
STUDIO = {
    "t2v_steps": 4,              # Lightning 4-step distilled default
    "t2v_guidance": 1.0,         # CFG-free (Lightning); slider 0..? step 0.1
    "user_lora_strength": 0.8,   # = basi.combo.USER_DEFAULT_STRENGTH (stacked w/ Lightning)
    "restyle_denoise": RESTYLE_SDEDIT_RECIPE["denoise"],   # 0.6, slider 0.4..0.9
    "seed": -1,                  # -1 = random per generation (preview/sampling use a fixed 42)
}

# --- Gym (training) UI defaults ------------------------------------------------------
GYM = {
    "frames": 33,               # 24g sweet spot (4n+1); preset.max_target_frames caps it
    "epochs": 16,
    "repeats": 1,
    "warmup_steps": 100,
    "save_every_n_epochs": 1,   # sample + checkpoint EVERY epoch (Flux-Gym parity)
    "resolution": "832*480",    # train here; generalizes up to 1280x720
    "preview_frames": 17,       # 4n+1; 1s @16fps -- safe quick preview
    "preview_steps": 4,         # Lightning
    "preview_seed": 42,
}

__all__ = ["RESTYLE_SDEDIT_RECIPE", "STUDIO", "GYM"]
