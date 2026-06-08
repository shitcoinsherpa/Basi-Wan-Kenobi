#!/usr/bin/env python3
"""BASI WAN K3N0B1 — smoke test.

Verifies the install without launching the UI. Run after `install.js` finishes:
  python scripts/smoke_test.py

Checks (in order, fails fast):
  1. Python + key library imports (torch, gradio, transformers, qwen_vl_utils, toml)
  2. CUDA / MPS availability + VRAM detection
  3. Wan2.2 base ckpt presence at expected paths
  4. Lightning LoRA dir presence
  5. TAEHV weights (auto-download if missing)
  6. Faster-Wan2.2 importable
  7. BASI modules importable
  8. ffprobe available (dataset validator dependency)

Exit code: 0 = OK, 1 = first failed check (with explanation).
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _fail(msg: str, hint: str = "") -> None:
    print(f"  ✗ {msg}")
    if hint:
        print(f"    → {hint}")
    sys.exit(1)


def check_imports() -> None:
    print("[1/8] Python imports")
    for mod, hint in [
        ("torch", "pip install torch (see install.js for GPU-aware index URL)"),
        ("gradio", "pip install -r requirements.txt"),
        ("transformers", "pip install transformers==4.56.1"),
        ("qwen_vl_utils", "pip install qwen-vl-utils"),
        ("toml", "pip install toml"),
        ("safetensors", "pip install safetensors"),
    ]:
        try:
            __import__(mod)
            _ok(f"{mod}")
        except ImportError as e:
            _fail(f"{mod}: {e}", hint)


def check_cuda() -> None:
    print("[2/8] CUDA / MPS / VRAM")
    import torch
    if torch.cuda.is_available():
        free_gb = torch.cuda.mem_get_info(0)[0] / (1024**3)
        total_gb = torch.cuda.mem_get_info(0)[1] / (1024**3)
        _ok(f"CUDA: {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB total, {free_gb:.1f} free)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _ok("MPS (Apple Silicon) — slow path for Wan2.2; CPU fallback active")
    else:
        _fail("no CUDA / MPS detected", "Wan2.2-A14B is GPU-only; CPU path is impractical")


def check_wan_ckpts() -> None:
    print("[3/8] Wan2.2 base checkpoints")
    ckpt_dir = Path(os.environ.get("BASI_CKPT_DIR",
                                   "/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B"))
    if not ckpt_dir.exists():
        _fail(f"{ckpt_dir} missing",
              "set BASI_CKPT_DIR or download Wan-AI/Wan2.2-T2V-A14B from HuggingFace")
    for required in ["low_noise_model", "high_noise_model", "Wan2.1_VAE.pth",
                     "models_t5_umt5-xxl-enc-bf16.pth"]:
        if not (ckpt_dir / required).exists():
            _fail(f"missing: {ckpt_dir}/{required}",
                  "re-download or fix the path")
    _ok(f"all present under {ckpt_dir}")


def check_lightning_lora() -> None:
    print("[4/8] Wan2.2-Lightning LoRA (for preview)")
    lora_dir = Path("/mnt/d/Ai/checkpoints/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928")
    if not lora_dir.exists():
        _fail(f"{lora_dir} missing — preview will be unusable",
              "download lightx2v/Wan2.2-Lightning from HuggingFace")
    for required in ["high_noise_model.safetensors", "low_noise_model.safetensors"]:
        if not (lora_dir / required).exists():
            _fail(f"missing: {lora_dir}/{required}", "")
    _ok(f"Lightning LoRA at {lora_dir}")


def check_taehv() -> None:
    print("[5/8] TAEHV (tiny-VAE for preview)")
    ckpt = Path(os.environ.get("BASIWAN_TAEHV_CKPT",
                               "/mnt/d/Ai/checkpoints/taehv/taew2_1.pth"))
    if not ckpt.exists():
        try:
            import urllib.request
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
            print(f"  ⤓ downloading {url}")
            urllib.request.urlretrieve(url, ckpt)
        except Exception as e:
            _fail(f"download failed: {e}",
                  f"manually fetch taew2_1.pth → {ckpt}")
    try:
        from taehv import TAEHV  # noqa: F401
    except ImportError:
        _fail("taehv package missing",
              "pip install git+https://github.com/madebyollin/taehv.git")
    _ok(f"{ckpt} ({ckpt.stat().st_size // 1024**2} MB)")


def check_faster_wan() -> None:
    print("[6/8] Faster-Wan2.2 importable")
    fw_root = Path(os.environ.get("BASIWAN_ROOT",
                                  "/mnt/d/Ai/transformers/work/BASIWAN"))
    if not fw_root.exists():
        _fail(f"{fw_root} missing",
              "clone https://github.com/Wan-Video/Wan2.2 or set BASIWAN_ROOT")
    sys.path.insert(0, str(fw_root))
    try:
        from wan.text2video import WanT2V  # noqa: F401
        from wan.modules.cuda_graphs import WanForwardCudaGraph  # noqa: F401
        _ok("wan.text2video + cuda_graphs import OK")
    except ImportError as e:
        _fail(f"Faster-Wan2.2 import: {e}",
              "check that Faster-Wan2.2's deps are installed in this venv")


def check_basi() -> None:
    print("[7/8] BASI modules")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from basi import caption, dataset, preview, presets, train  # noqa: F401
        _ok("basi.{caption, dataset, preview, presets, train} import OK")
    except ImportError as e:
        _fail(f"basi import: {e}", "")


def check_ffprobe() -> None:
    print("[8/8] ffprobe (for dataset validator)")
    if shutil.which("ffprobe") is None:
        _fail("ffprobe not on PATH",
              "apt install ffmpeg / brew install ffmpeg / choco install ffmpeg")
    _ok("ffprobe found")


def main() -> int:
    print("BASI WAN K3N0B1 — smoke test\n")
    check_imports()
    check_cuda()
    check_wan_ckpts()
    check_lightning_lora()
    check_taehv()
    check_faster_wan()
    check_basi()
    check_ffprobe()
    print("\nAll checks passed. Run `python app.py` to launch the UI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
