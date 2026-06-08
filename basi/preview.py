"""Live preview generator for BASI WAN K3N0B1.

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
from dataclasses import dataclass
from pathlib import Path

# Defaults aligned with Faster-Wan2.2's Lightning recipe (FP8 + 4 steps, no CFG).
# Override via env vars for Pinokio installs or alternate checkpoint layouts.
DEFAULT_BASIWAN_ROOT = os.environ.get(
    "BASIWAN_ROOT", "/mnt/d/Ai/transformers/work/BASIWAN")
DEFAULT_CKPT_DIR = os.environ.get(
    "BASI_CKPT_DIR", "/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B")
DEFAULT_LIGHTNING_LORA_DIR = os.environ.get(
    "BASI_LIGHTNING_LORA_DIR",
    "/mnt/d/Ai/checkpoints/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928")


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
    fw_generate = Path(faster_wan_root) / "generate.py"
    if not fw_generate.exists():
        raise FileNotFoundError(
            f"Faster-Wan2.2 generate.py not found at {fw_generate}. "
            f"Set BASIWAN_ROOT or clone Faster-Wan2.2 to {faster_wan_root}. "
            f"Preview is optional — training will work without it."
        )

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

    env = os.environ.copy()
    # Lightning LoRA auto-applied via P19 env var
    env["BASIWAN_LORA_DIR"] = lightning_lora_dir
    # User's trained LoRA stacked on top via P24
    env["BASIWAN_USER_LORA"] = spec.lora_path
    env["BASIWAN_USER_LORA_EXPERT"] = spec.lora_expert
    env["BASIWAN_USER_LORA_STRENGTH"] = str(spec.lora_strength)
    # P26: TAEHV tiny-VAE for 4-6× faster decode. Preview is convergence-monitoring,
    # not final output — the slight quality cost is the right tradeoff.
    env.setdefault("BASIWAN_TAEHV_VAE", "1")
    # P25 CUDA Graphs at Lightning shape: kernel-launch amortization. Default on
    # for preview since shape is fixed per render. Set BASIWAN_CUDA_GRAPHS=0
    # in the caller env to disable if it conflicts with anything.
    env.setdefault("BASIWAN_CUDA_GRAPHS", "1")
    # P34 lower-VRAM support: opt the preview into the same shape-aware
    # block-swap auto-tune the inference fork ships. The runner picks N at
    # pipe-build time from measured peak_alloc + free_vram. Explicit int
    # overrides still win (so cards that want a specific N can pin it via
    # the caller env). See Faster-Wan2.2/docs/lower_vram_support.md.
    env.setdefault("BASIWAN_BLOCK_SWAP_N", "auto")
    # bf16-autocast norm fix — universal benefit; only matters at high seq
    # but costs nothing at low seq. Lets p720_81f-shaped previews fit even
    # on 24GB cards, and prevents fp32-promoted layer_norm transients on
    # smaller cards. Mirrors the env used by run_one_video.py --stacked.
    env.setdefault("BASIWAN_RMS_BF16", "1")
    env.setdefault("BASIWAN_LN_BF16", "1")
    # Quantization scheme: the inference path picks this from the prequant
    # dir, but we surface the recommendation here so the UI / installer
    # can pre-select the right prequant for the detected GPU (FP8 vs Int8
    # vs GGUF Q4). The runner will warn on a scheme/SM mismatch.
    try:
        from basi.presets import detect_capability, recommend_inference_scheme
        cap = detect_capability()
        env.setdefault("BASI_RECOMMENDED_SCHEME", recommend_inference_scheme(cap))
        env.setdefault("BASI_GPU_NAME", cap.name)
    except Exception as e:  # never block preview on the recommender
        print(f"[preview] capability probe failed (non-fatal): {e}")

    cmd = [
        venv_python,
        str(Path(faster_wan_root) / "generate.py"),
        "--task", "t2v-A14B",
        "--size", spec.size,
        "--frame_num", str(spec.frame_num),
        "--ckpt_dir", ckpt_dir,
        "--prompt", spec.prompt,
        "--save_file", str(out),
        "--sample_steps", str(spec.sample_steps),
        "--sample_guide_scale", str(spec.sample_guide_scale),
        "--base_seed", str(spec.base_seed),
        "--offload_model", "True",
        "--convert_model_dtype",
    ]

    print(f"[preview] {spec.size} × {spec.frame_num}f, {spec.sample_steps} steps → {out.name}")
    subprocess.run(cmd, env=env, cwd=faster_wan_root, check=True, timeout=timeout_sec)
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
