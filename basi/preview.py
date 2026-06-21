"""Live preview generator for BASI WAN KENOBI.

Spawns a Faster-Wan2.2 inference subprocess to render a short clip from the most
recent LoRA checkpoint. Uses Wan2.2-Lightning (4 denoise steps, CFG-free) + FP8
weights for the cheapest preview path — ~3-4s steady-state at 832x480x17.

Important constraint: a single 4090 cannot run trainer + preview concurrently
(trainer holds ~22GB resident with blocks_to_swap=10). The preview is therefore
designed to be invoked BETWEEN training phases — either:
  - on-demand from the BASI UI ("Generate preview" button), or
  - in a training-pause hook (future: needs IPC with the trainer subprocess).

For now the in-loop hook is out of scope; the user triggers preview manually
between epochs from the UI. Multi-GPU users can run preview on the second GPU
concurrently by setting CUDA_VISIBLE_DEVICES.

Why not 256x256: Wan's spatial positional embeddings are tied to the standard
buckets (see basi.dataset.SUPPORTED_RESOLUTIONS); 832x480 is the cheapest
supported size for landscape clips.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Defaults aligned with the Lightning recipe (FP8 + 4 steps, no CFG).
# Override via env vars for Pinokio installs or alternate checkpoint layouts.
# Repo-relative paths, not dev-box absolute: preview.py is basi/preview.py → repo
# root is one parent up. checkpoints/ is where install.js downloads weights.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_CKPT_BASE = os.environ.get("BASIWAN_CKPT_DIR", str(_REPO_ROOT / "checkpoints"))
DEFAULT_BASIWAN_ROOT = os.environ.get("BASIWAN_ROOT", str(_REPO_ROOT))
DEFAULT_CKPT_DIR = os.environ.get(
    "BASI_CKPT_DIR", str(Path(_CKPT_BASE) / "Wan2.2-T2V-A14B"))
DEFAULT_LIGHTNING_LORA_DIR = os.environ.get(
    "BASI_LIGHTNING_LORA_DIR",
    str(Path(_CKPT_BASE) / "Wan2.2-Lightning"
        / "Wan2.2-T2V-A14B-4steps-lora-250928"))


@dataclass
class PreviewSpec:
    """One preview render request."""
    lora_path: str                          # the user's trained LoRA .safetensors
    prompt: str                             # text prompt
    output_path: str                        # where to write the .mp4
    size: str = "832*480"                   # Wan-supported W*H
    frame_num: int = 17                     # 4n+1; 17 frames = 1s @16fps
    sample_steps: int = 4                   # Lightning default
    sample_guide_scale: float = 1.0         # CFG-skip (Lightning is CFG-free)
    base_seed: int = 42
    lora_strength: float = 1.0
    lora_expert: str = "low"                # "low" | "high" | "both" — matches BASI 24G preset


def generate_preview(spec: PreviewSpec,
                     faster_wan_root: str = DEFAULT_BASIWAN_ROOT,
                     ckpt_dir: str = DEFAULT_CKPT_DIR,
                     lightning_lora_dir: str = DEFAULT_LIGHTNING_LORA_DIR,
                     venv_python: str | None = None,
                     timeout_sec: int = 120) -> Path:
    """Render a single preview clip via Faster-Wan2.2 subprocess.

    Returns the output path on success. Raises CalledProcessError on failure
    (the trainer caller should catch + log + continue).
    """
    if not Path(spec.lora_path).exists():
        raise FileNotFoundError(f"LoRA checkpoint not found: {spec.lora_path}")
    if not Path(ckpt_dir).exists():
        raise FileNotFoundError(f"Base Wan2.2 ckpt dir not found: {ckpt_dir}")
    # Preview renders through the SAME validated path the Studio tab uses
    # (Lightning+user-LoRA combo -> tools/run_one_video_gguf.py -> pt->mp4).
    _runner = Path(faster_wan_root) / "tools" / "run_one_video_gguf.py"
    if not _runner.exists():
        raise FileNotFoundError(
            f"GGUF runner not found at {_runner}. Set BASIWAN_ROOT to the app root. "
            f"Preview is optional — training will work without it.")

    # Default to the current interpreter (works inside Pinokio's single-venv
    # install). Override via BASIWAN_VENV_PYTHON if the user maintains a
    # separate Faster-Wan2.2 env (dev setup).
    import sys as _sys
    venv_python = venv_python or os.environ.get(
        "BASIWAN_VENV_PYTHON",
        _sys.executable,
    )
    out = Path(spec.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the T2V GGUF pair (env override -> HF-snapshot/flat glob under the
    # checkpoint dir) + VAE + base from the T2V ckpt dir.
    import glob as _glob
    from basi import combo as _combo

    def _resolve_gguf(env_key, which, fname):
        v = os.environ.get(env_key)
        if v and Path(v).exists():
            return v
        gd = Path(_CKPT_BASE) / "gguf"
        for pat in (str(gd / "models--*" / "snapshots" / "*" / which / "*Q4_K_M.gguf"),
                    str(gd / "*" / which / "*Q4_K_M.gguf"),
                    str(gd / which / fname)):
            hits = sorted(_glob.glob(pat))
            if hits:
                return hits[0]
        raise FileNotFoundError(
            f"T2V GGUF {which} not found under {gd}. Download "
            f"QuantStack/Wan2.2-T2V-A14B-GGUF into checkpoints/gguf, or set {env_key}.")

    gguf_high = _resolve_gguf("BASIWAN_GGUF_HIGH", "HighNoise",
                              "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf")
    gguf_low = _resolve_gguf("BASIWAN_GGUF_LOW", "LowNoise",
                             "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf")
    vae = Path(ckpt_dir) / "Wan2.1_VAE.pth"

    # Build the Lightning + user-LoRA combo. The user's single-file LoRA maps to
    # the expert(s) it was trained for; the runner loads the combo dir's
    # {high,low}_noise_model.safetensors via --lora-dir.
    _u = str(spec.lora_path)
    _e = spec.lora_expert
    user_high = _u if _e in ("high", "both") else None
    user_low = _u if _e in ("low", "both") else None
    combo_dir = Path(tempfile.mkdtemp(prefix="basiwan_preview_combo_"))
    _combo.build_combo(lightning_lora_dir, user_high, user_low, combo_dir,
                       user_strength=float(spec.lora_strength), lightning_strength=1.0)

    w, h = (int(s) for s in spec.size.replace("x", "*").split("*"))
    pt_out = out.with_suffix(".pt")
    meta_out = out.with_suffix(".preview.json")

    env = os.environ.copy()
    # The runner reads these ship-recipe perf flags; the combo dir supplies the
    # LoRA, so clear any BASIWAN_USER_LORA*/LORA_DIR that would conflict.
    env.setdefault("BASIWAN_V2", "1")
    env.setdefault("BASIWAN_USE_PACK_CACHE", "1")
    env.setdefault("BASIWAN_NO_POOL", "1")
    env.setdefault("BASIWAN_VAE_BF16", "1")
    env.setdefault("BASIWAN_TAEHV_VAE", "1")   # preview = convergence monitor
    for _k in ("BASIWAN_USER_LORA", "BASIWAN_USER_LORA_EXPERT",
               "BASIWAN_USER_LORA_STRENGTH", "BASIWAN_LORA_DIR"):
        env.pop(_k, None)

    cmd = [
        venv_python, "-u", str(_runner),
        "--model-type", "t2v",
        "--gguf-high", str(gguf_high), "--gguf-low", str(gguf_low),
        "--vae", str(vae), "--base-dir", str(ckpt_dir),
        "--lora-dir", str(combo_dir), "--lora-mode", "forward",
        "--prompt", spec.prompt,
        "--width", str(w), "--height", str(h), "--frames", str(spec.frame_num),
        "--steps", str(spec.sample_steps), "--guide", str(spec.sample_guide_scale),
        "--out", str(pt_out), "--meta", str(meta_out),
    ]
    print(f"[preview] {spec.size} x {spec.frame_num}f, {spec.sample_steps} steps "
          f"(Lightning+user@{spec.lora_strength} on {_e}) -> {out.name}", flush=True)
    try:
        subprocess.run(cmd, env=env, cwd=str(faster_wan_root), check=True,
                       timeout=timeout_sec)
        # .pt -> .mp4 (fps=16, libx264) via the vendored converter.
        subprocess.run([venv_python, str(Path(faster_wan_root) / "tools" / "pt_to_mp4.py"),
                        str(pt_out), str(out)], check=True, timeout=180)
    finally:
        import shutil as _sh
        _sh.rmtree(combo_dir, ignore_errors=True)
    if not out.exists():
        raise RuntimeError(f"preview render produced no .mp4 at {out}")
    return out


def latest_lora_in(workspace_dir: str | Path) -> Path | None:
    """Find the newest .safetensors LoRA in a BASI workspace output dir."""
    candidates = list(Path(workspace_dir).glob("*.safetensors"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser(description="BASI preview render via Faster-Wan2.2")
    ap.add_argument("--lora", required=True, help="path to trained LoRA .safetensors "
                                                  "(or workspace dir — newest is picked)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True, help="output .mp4 path")
    ap.add_argument("--size", default="832*480")
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    lora = Path(args.lora)
    if lora.is_dir():
        lora = latest_lora_in(lora)
        if lora is None:
            raise SystemExit(f"no .safetensors found in {args.lora}")
    spec = PreviewSpec(
        lora_path=str(lora), prompt=args.prompt, output_path=args.out,
        size=args.size, frame_num=args.frames, sample_steps=args.steps,
        base_seed=args.seed,
    )
    t0 = time.time()
    p = generate_preview(spec)
    print(f"[preview] done in {time.time()-t0:.1f}s → {p}")
