#!/usr/bin/env python3
"""BASI WAN K3N0B1 — end-to-end real-training smoke test.

Extends scripts/smoke_test_pipeline.py from `bash -n` (syntax check only) to
an actual training step. The point: prove the gym ships a working pipeline,
not just a syntactically-valid one.

What it does:
  1. Ingest the same pose.mp4 sample
  2. Generate cache.sh + train.sh via TrainConfig
  3. Override --max_train_steps=5 (smoke, not real training)
  4. Run cache.sh (T5 + VAE precompute for 1 clip — ~60s)
  5. Run train.sh for 5 steps
  6. Verify {workspace}/{lora_name}.safetensors was produced
  7. Verify {workspace}/sample/*.mp4 exists if sample_every_n_steps was set
  8. Cleanup

Pre-conditions: CUDA device, ≥16GB free VRAM, base Wan2.2 + Lightning LoRA
present. Time budget: ~3-5 min.

Run after smoke_test.py + smoke_test_pipeline.py both pass.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SAMPLE_VIDEO = Path("/mnt/d/Ai/transformers/work/Faster-Wan2.2/examples/pose.mp4")
TIMEOUT_CACHE_S = 300
TIMEOUT_TRAIN_S = 600  # 5 steps × ~30s each ≈ 150s, with margin
MAX_TRAIN_STEPS = 5


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _patch_max_steps(train_script: Path) -> None:
    """Inject --max_train_steps=N into the generated train.sh."""
    src = train_script.read_text()
    if "--max_train_steps" not in src:
        src = re.sub(r"(--max_train_epochs \d+)",
                     rf"\1 --max_train_steps {MAX_TRAIN_STEPS}",
                     src, count=1)
    train_script.write_text(src)


def main() -> int:
    print("BASI end-to-end smoke test (real training, 5 steps)\n")
    if not SAMPLE_VIDEO.exists():
        _fail(f"sample video not found at {SAMPLE_VIDEO}")
    try:
        import torch
        if not torch.cuda.is_available():
            _fail("CUDA not available; e2e smoke requires GPU")
        free, total = torch.cuda.mem_get_info()
        free_gb = free / 1024**3
        if free_gb < 16:
            _fail(f"only {free_gb:.1f} GB GPU free; need ≥16 GB")
        _ok(f"GPU: {torch.cuda.get_device_name(0)} ({free_gb:.1f} GB free)")
    except ImportError:
        _fail("torch not installed")

    from basi.dataset import scan_dataset
    from basi.presets import PRESETS, auto_select, detect_vram_gb
    from basi.train import TrainConfig, prepare_training_run

    workspace = REPO / "outputs" / "_smoke_e2e"
    if workspace.exists():
        shutil.rmtree(workspace)
    ds_dir = workspace / "dataset"
    ds_dir.mkdir(parents=True)
    shutil.copy2(SAMPLE_VIDEO, ds_dir / SAMPLE_VIDEO.name)
    (ds_dir / SAMPLE_VIDEO.with_suffix(".txt").name).write_text(
        "a person dancing, full body shot, studio lighting\n")
    _ok(f"ingested {SAMPLE_VIDEO.name} → {ds_dir}")

    clips = scan_dataset(str(ds_dir))
    if not clips:
        _fail("scan_dataset returned no clips")
    _ok(f"validator: {clips[0].width}x{clips[0].height} × {clips[0].frame_count}f")

    preset_key = "24g" if (detect_vram_gb() or 24) >= 22 else "16g"
    cfg = TrainConfig(
        lora_name="_smoke_e2e",
        dataset_dir=str(ds_dir),
        dit_low_path="/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors",
        dit_high_path=None,
        t5_path="/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth",
        vae_path="/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth",
        preset=PRESETS[preset_key],
        target_resolution=(832, 480),
        target_frames=17,  # smallest valid for fastest smoke
        max_train_epochs=1,
        sample_every_n_steps=0,
    )
    run = prepare_training_run(cfg)
    cache_script = Path(run["cache_script"])
    train_script = Path(run["train_script"])
    _patch_max_steps(train_script)
    _ok(f"scripts ready (max_train_steps={MAX_TRAIN_STEPS})")

    print(f"[3/5] Run cache.sh (timeout {TIMEOUT_CACHE_S}s)")
    t0 = time.time()
    rv = subprocess.run(["bash", str(cache_script)],
                        capture_output=True, text=True, timeout=TIMEOUT_CACHE_S)
    if rv.returncode != 0:
        print(rv.stdout[-2000:])
        print(rv.stderr[-2000:])
        _fail(f"cache.sh failed (rc={rv.returncode})")
    _ok(f"cache.sh OK in {time.time()-t0:.1f}s")

    print(f"[4/5] Run train.sh (timeout {TIMEOUT_TRAIN_S}s, {MAX_TRAIN_STEPS} steps)")
    t0 = time.time()
    rv = subprocess.run(["bash", str(train_script)],
                        capture_output=True, text=True, timeout=TIMEOUT_TRAIN_S)
    if rv.returncode != 0:
        print(rv.stdout[-3000:])
        print(rv.stderr[-3000:])
        _fail(f"train.sh failed (rc={rv.returncode})")
    _ok(f"train.sh OK in {time.time()-t0:.1f}s")

    print("[5/5] Verify outputs")
    safetensors = list(workspace.glob("*.safetensors"))
    if not safetensors:
        _fail(f"no .safetensors LoRA found in {workspace} after training")
    _ok(f"checkpoint: {safetensors[0].name} "
        f"({safetensors[0].stat().st_size / 1024**2:.1f} MB)")

    shutil.rmtree(workspace)
    print("\nEnd-to-end smoke test passed. The gym ships a working pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
