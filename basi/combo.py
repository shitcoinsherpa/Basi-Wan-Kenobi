"""Combine a Lightning-4 distill LoRA with a user-trained LoRA into one
rank-concatenated file per expert, for Studio "generate with my LoRA".

Why rank-concat: the runner's forward-LoRA path attaches a single (down, up,
scale) per module. Two stacked LoRAs are the sum of their deltas:

    delta = s1·(a1/r1)·U1@D1  +  s2·(a2/r2)·U2@D2
          = cat(U1·c1, U2·c2) @ cat(D1, D2)           with ci = si·(ai/ri)

so a single attach of the concatenated (down, up) with alpha = (r1+r2) — which
makes the runner's intrinsic (alpha/rank) scale exactly 1.0 — reproduces the
stacked result with zero hot-path change. Keys are written in Lightning's
dotted `diffusion_model.<name>.lora_{down,up}.weight` / `.alpha` style so the
runner's existing `_parse_lightning_lora` consumes them unchanged.

This is the productized form of the offline build_combo_lora.py that produced
the validated showcase (8/8 shapes), bit-identical to it.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

__all__ = ["build_combo", "build_user_only", "combo_cache_path",
           "LIGHTNING_DEFAULT_STRENGTH", "USER_DEFAULT_STRENGTH"]

LIGHTNING_DEFAULT_STRENGTH = 1.0
USER_DEFAULT_STRENGTH = 0.8  # training brief: 0.7-0.85 when stacked w/ Lightning
_EXPERTS = ("high_noise_model", "low_noise_model")


def _musubi_to_dotted(key: str) -> str | None:
    """Map a musubi LoRA key (lora_unet_blocks_N_<attn>_<proj>) to the dotted
    WanModel path the runner walks. Returns None for keys that don't match the
    block pattern (e.g. embedders) — caller skips them rather than crashing,
    so an unexpected musubi layout degrades gracefully instead of aborting the
    whole combo (the offline prototype asserted here — a real product can't)."""
    k = key[len("lora_unet_"):] if key.startswith("lora_unet_") else key
    m = re.match(r"blocks_(\d+)_(self_attn|cross_attn|ffn)_(.+)$", k)
    if not m:
        return None
    return f"blocks.{m.group(1)}.{m.group(2)}.{m.group(3)}"


def _group(sd: dict, key_fn) -> dict:
    """Group flat LoRA tensors into {target_name: {lora_down, lora_up, alpha}}.
    key_fn maps the raw key prefix to the target name (or None to drop)."""
    t: dict = {}
    for k, v in sd.items():
        for suf in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if k.endswith(suf):
                name = key_fn(k[: -len(suf)])
                if name is None:
                    break
                kind = suf.lstrip(".").replace(".weight", "")
                t.setdefault(name, {})[kind] = v
                break
    return t


def _light_keyfn(k: str) -> str:
    return k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k


def _combine_expert(lightning_file: Path, user_file: Path | None,
                    user_strength: float, lightning_strength: float,
                    out_file: Path) -> dict:
    """Write one expert's combo file. user_file=None → Lightning-only passthrough
    for that expert (legacy low-only LoRAs, or a missing high half)."""
    lt = _group(load_file(str(lightning_file)), _light_keyfn)
    ut = _group(load_file(str(user_file)), _musubi_to_dotted) if user_file else {}
    out: dict = {}
    names = sorted(set(lt) | set(ut))
    for name in names:
        downs, ups = [], []
        for parts, s in ((lt.get(name), lightning_strength),
                         (ut.get(name), user_strength)):
            if not parts or "lora_down" not in parts or "lora_up" not in parts:
                continue
            d = parts["lora_down"].float()
            u = parts["lora_up"].float()
            a = parts.get("alpha")
            r = d.shape[0]
            c = (float(a.item()) if a is not None else r) / r * s
            downs.append(d)
            ups.append(u * c)
        if not downs:
            continue
        down = torch.cat(downs, dim=0)
        up = torch.cat(ups, dim=1)
        out[f"diffusion_model.{name}.lora_down.weight"] = down.to(torch.bfloat16)
        out[f"diffusion_model.{name}.lora_up.weight"] = up.to(torch.bfloat16)
        # alpha = rank → runner's (alpha/rank) intrinsic scale is exactly 1.0;
        # all strength is already baked into `up`.
        out[f"diffusion_model.{name}.alpha"] = torch.tensor(float(down.shape[0]))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_file))
    return {"modules": len(out) // 3,
            "light_only": len(set(lt) - set(ut)),
            "user_only": len(set(ut) - set(lt)),
            "both": len(set(lt) & set(ut))}


def _user_only_expert(user_file: Path | None, user_strength: float,
                      out_file: Path) -> dict:
    """Write one expert's USER-ONLY combo file (no Lightning). Converts musubi
    underscore keys to the dotted WanModel paths the runner walks, baking
    `user_strength` into `up` and setting alpha=rank so the runner's intrinsic
    (alpha/rank) scale is 1.0 — identical key/scale convention to _combine_expert
    so the same forward-LoRA attach consumes it unchanged."""
    ut = _group(load_file(str(user_file)), _musubi_to_dotted) if user_file else {}
    out: dict = {}
    for name, parts in ut.items():
        if "lora_down" not in parts or "lora_up" not in parts:
            continue
        d = parts["lora_down"].float()
        u = parts["lora_up"].float()
        a = parts.get("alpha")
        r = d.shape[0]
        c = (float(a.item()) if a is not None else r) / r * user_strength
        out[f"diffusion_model.{name}.lora_down.weight"] = d.to(torch.bfloat16)
        out[f"diffusion_model.{name}.lora_up.weight"] = (u * c).to(torch.bfloat16)
        out[f"diffusion_model.{name}.alpha"] = torch.tensor(float(r))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(out_file))
    return {"modules": len(out) // 3, "user_only": len(ut)}


def build_user_only(user_high: str | Path | None, user_low: str | Path | None,
                    out_dir: str | Path,
                    user_strength: float = 1.0) -> dict:
    """Build {high,low}_noise_model.safetensors from a user LoRA ALONE — no
    Lightning — for the VACE 50-step / full-CFG depth-control path (Lightning is
    a step-distill LoRA and would distort a non-Lightning generate). A user LoRA
    trained on T2V has only blocks.* keys (never vace_blocks.*), so the 8 VACE
    depth-control blocks stay untouched: style rides the main blocks, structure
    stays locked to the depth control. Default user_strength=1.0 (no Lightning to
    share rank with); the worker can still scale at request time via set_lora."""
    out_dir = Path(out_dir)
    user = {"high_noise_model": user_high, "low_noise_model": user_low}
    stats = {}
    for expert in _EXPERTS:
        uf = Path(user[expert]) if user[expert] else None
        stats[expert] = _user_only_expert(
            uf, user_strength, out_dir / f"{expert}.safetensors")
    return stats


def build_combo(lightning_dir: str | Path, user_high: str | Path | None,
                user_low: str | Path | None, out_dir: str | Path,
                user_strength: float = USER_DEFAULT_STRENGTH,
                lightning_strength: float = LIGHTNING_DEFAULT_STRENGTH) -> dict:
    """Build {high,low}_noise_model.safetensors combos into out_dir.

    user_high / user_low: per-expert user LoRA files. Either may be None
    (Lightning-only for that expert). For a single "both"-trained file, pass
    the same path for both. Returns per-expert stats."""
    lightning_dir = Path(lightning_dir)
    out_dir = Path(out_dir)
    user = {"high_noise_model": user_high, "low_noise_model": user_low}

    # Per-expert user strength. user_strength may be a float (both experts), a
    # (high, low) tuple/list, or a dict keyed by expert name. Lets callers run the
    # high-noise expert hot (e.g. 1.3-1.5, sets global style/color) while the
    # low-noise stays ~1.0. Float path is byte-identical to the scalar case.
    def _us(expert):
        if isinstance(user_strength, dict):
            return float(user_strength.get(expert, USER_DEFAULT_STRENGTH))
        if isinstance(user_strength, (tuple, list)):
            return float(user_strength[0] if expert == "high_noise_model"
                         else user_strength[1])
        return float(user_strength)

    stats = {}
    for expert in _EXPERTS:
        uf = Path(user[expert]) if user[expert] else None
        stats[expert] = _combine_expert(
            lightning_dir / f"{expert}.safetensors", uf,
            _us(expert), lightning_strength, out_dir / f"{expert}.safetensors")
    return stats


def combo_cache_path(cache_root: str | Path, lightning_dir: str | Path,
                     user_files: list, user_strength: float,
                     lightning_strength: float = LIGHTNING_DEFAULT_STRENGTH) -> Path:
    """Deterministic cache dir keyed on (lightning dir name, each user file's
    path+mtime+size, user strength, lightning strength). Retraining overwrites
    the user file → new mtime/size → new key, so stale combos never serve.

    lightning_strength is folded in ONLY when it departs from the default 1.0,
    so the plain-T2V combo key (always 1.0) is stable and its cache is not
    invalidated. The Restyle path builds at 0.7 (measured optimum: s8_d6_L0.7 =
    0.812 style-sim vs 4-step 0.750) and must NOT collide with the 1.0 combo over
    the same user files."""
    h = hashlib.sha1()
    h.update(str(Path(lightning_dir).name).encode())
    h.update(f"{user_strength:.4f}".encode())
    if abs(lightning_strength - LIGHTNING_DEFAULT_STRENGTH) > 1e-6:
        h.update(f"L{lightning_strength:.4f}".encode())
    for f in user_files:
        if f and Path(f).exists():
            st = Path(f).stat()
            h.update(f"{Path(f).name}:{st.st_mtime_ns}:{st.st_size}".encode())
        else:
            h.update(b"none")
    return Path(cache_root) / h.hexdigest()[:16]
