#!/usr/bin/env python3
"""BASI WAN K3N0B1 — pipeline smoke test (no GPU training run).

Exercises the full BASI flow against a single sample clip:
  1. Ingest a video into a throwaway workspace
  2. Run the dataset validator (4n+1, bucketing)
  3. Generate cache.sh + train.sh scripts via TrainConfig
  4. Verify both scripts are valid Bash (`bash -n`)
  5. Verify musubi-tuner entrypoints exist

Run after `smoke_test.py` passes. Cleanup is automatic.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Portable: BASIWAN_CKPT_DIR (default <repo>/checkpoints) + a sample clip from
# BASIWAN_SAMPLE_VIDEO or the bundled examples/ dir. No dev-box paths.
import os as _os
_CKPT = Path(_os.environ.get("BASIWAN_CKPT_DIR", str(REPO / "checkpoints")))
_T2V = _CKPT / "Wan2.2-T2V-A14B"
SAMPLE_VIDEO = Path(_os.environ.get(
    "BASIWAN_SAMPLE_VIDEO", str(REPO / "examples" / "pose.mp4")))


def _fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    sys.exit(1)


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def main() -> int:
    print("BASI pipeline smoke test\n")
    if not SAMPLE_VIDEO.exists():
        _fail(f"sample video not found at {SAMPLE_VIDEO}. Set BASIWAN_SAMPLE_VIDEO to any short .mp4 clip.")

    from basi.dataset import scan_dataset, bucket_distribution
    from basi.presets import PRESETS, auto_select, detect_vram_gb
    from basi.train import TrainConfig, prepare_training_run, MUSUBI_ROOT

    workspace = REPO / "outputs" / "_smoke_pipeline"
    if workspace.exists():
        shutil.rmtree(workspace)
    ds_dir = workspace / "dataset"
    ds_dir.mkdir(parents=True)
    shutil.copy2(SAMPLE_VIDEO, ds_dir / SAMPLE_VIDEO.name)
    (ds_dir / SAMPLE_VIDEO.with_suffix(".txt").name).write_text(
        "a person dancing, full body shot, studio lighting\n")
    _ok(f"ingested {SAMPLE_VIDEO.name} → {ds_dir}")

    print("[2/5] Dataset validator")
    clips = scan_dataset(str(ds_dir))
    if not clips:
        _fail("scan_dataset returned no clips")
    c = clips[0]
    _ok(f"clip: {c.width}x{c.height} @{c.fps:.1f}fps × {c.frame_count}f; caption: {bool(c.caption_text)}")
    buckets = bucket_distribution(clips)
    if not buckets:
        _fail("no buckets produced")
    _ok(f"bucket distribution: {dict(buckets)}")

    print("[3/5] Generate scripts (24G preset)")
    preset = auto_select(detect_vram_gb() or 24)
    cfg = TrainConfig(
        lora_name="_smoke_pipeline",
        dataset_dir=str(ds_dir),
        dit_low_path=str(_T2V / "low_noise_model" / "diffusion_pytorch_model-00001-of-00006.safetensors"),
        dit_high_path=None,
        t5_path=str(_T2V / "models_t5_umt5-xxl-enc-bf16.pth"),
        vae_path=str(_T2V / "Wan2.1_VAE.pth"),
        preset=preset,
        target_resolution=(832, 480), target_frames=81,
        max_train_epochs=1, sample_every_n_steps=0,
    )
    run = prepare_training_run(cfg)
    _ok(f"cache.sh: {run['cache_script']}")
    _ok(f"train.sh: {run['train_script']}")

    print("[4/5] Script syntax check (bash -n)")
    for key in ("cache_script", "train_script"):
        rv = subprocess.run(["bash", "-n", run[key]], capture_output=True, text=True)
        if rv.returncode != 0:
            _fail(f"{key} bash -n failed: {rv.stderr}")
        _ok(f"{Path(run[key]).name} parses")

    print("[5/5] musubi-tuner entrypoints exist")
    for entry in ("wan_train_network.py", "wan_cache_latents.py",
                  "wan_cache_text_encoder_outputs.py"):
        p = MUSUBI_ROOT / "src" / "musubi_tuner" / entry
        if not p.exists():
            _fail(f"missing musubi entrypoint: {p}")
        _ok(f"found: {entry}")

    # Cleanup
    shutil.rmtree(workspace)
    print("\nPipeline smoke test passed. Full training requires a real dataset + Run training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
