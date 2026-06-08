"""Subprocess wrapper for musubi-tuner Wan2.2 LoRA training (Flux-Gym pattern)."""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import toml

from .presets import Preset

BASI_ROOT = Path(__file__).resolve().parent.parent
MUSUBI_ROOT = BASI_ROOT / "ext" / "musubi-tuner"
OUTPUTS_ROOT = BASI_ROOT / "outputs"


@dataclass
class TrainConfig:
    """Everything BASI needs to fire a single LoRA training run."""
    lora_name: str
    dataset_dir: str          # contains .mp4 + .txt pairs
    dit_low_path: str         # path to wan2.2 low_noise model (single .safetensors or NNNNN-of-NNNNN)
    dit_high_path: str | None # optional: high_noise for dual-expert (40GB+ only)
    t5_path: str              # umt5-xxl T5 encoder
    vae_path: str             # Wan2.1 VAE (same for 14B)
    preset: Preset
    target_resolution: tuple[int, int]   # e.g. (832, 480)
    target_frames: int        # e.g. 81 (4n+1)
    max_train_epochs: int = 20
    save_every_n_epochs: int = 2
    sample_every_n_steps: int = 200
    sample_prompts: list[str] | None = None
    trigger_word: str | None = None
    seed: int = 42
    # T4.A: Optional resume from a saved state directory (musubi --resume).
    # If set, must point at a previously-saved training state directory
    # (musubi saves state every N epochs when --save_state is on).
    resume_state: str | None = None
    # T4.G: Per-image repeats — applies to the [[datasets]] block. >1 means
    # each clip is iterated N times per epoch, useful for small datasets.
    num_repeats: int = 1
    # T4.G: Enable musubi auto-bucketing in the [general] block so the trainer
    # groups mixed-resolution clips into per-bucket batches rather than padding
    # to the nominal target_resolution.
    enable_bucket: bool = True
    # T4.H: Advanced optimizer args (None = musubi default).
    lr_scheduler: str | None = None       # constant_with_warmup / cosine / linear / rex
    lr_warmup_steps: int | None = None
    max_grad_norm: float | None = None    # 0 = disable clip_grad_norm
    weight_decay: float | None = None


def gen_dataset_toml(cfg: TrainConfig, output_path: Path) -> Path:
    """Write musubi-tuner dataset config (TOML).

    T4.G: emits [[datasets]] with cfg.num_repeats and toggles
    [general].enable_bucket so musubi auto-groups mixed-resolution clips
    into per-bucket batches (was: padded all to target_resolution).
    """
    res_w, res_h = cfg.target_resolution
    dataset_block = {
        "resolution": [res_w, res_h],
        "target_frames": [cfg.target_frames],
        "frame_extraction": "head",
        "video_directory": cfg.dataset_dir,
        "caption_extension": ".txt",
        "fps": 16,
        "max_frames": cfg.target_frames,
        "batch_size": cfg.preset.batch_size,
        "num_repeats": cfg.num_repeats,
    }
    general_block = {
        "caption_extension": ".txt",
        "batch_size": cfg.preset.batch_size,
    }
    if cfg.enable_bucket:
        general_block["enable_bucket"] = True
    toml_dict = {
        "general": general_block,
        "datasets": [dataset_block],
    }
    output_path.write_text(toml.dumps(toml_dict))
    return output_path


def gen_cache_commands(cfg: TrainConfig, dataset_toml: Path) -> list[list[str]]:
    """Build musubi-tuner's mandatory precompute commands (T5 + VAE caches).

    musubi-tuner's training loop reads from cached .cache files for speed; without
    these the trainer either errors or runs orders of magnitude slower because it
    re-encodes every text + video per step. Run these once after dataset ingest,
    re-run when the dataset changes.
    """
    t5_cmd = [
        "python", str(MUSUBI_ROOT / "src" / "musubi_tuner" / "wan_cache_text_encoder_outputs.py"),
        "--dataset_config", str(dataset_toml),
        "--t5", cfg.t5_path,
        "--batch_size", "16",
    ]
    vae_cmd = [
        "python", str(MUSUBI_ROOT / "src" / "musubi_tuner" / "wan_cache_latents.py"),
        "--dataset_config", str(dataset_toml),
        "--vae", cfg.vae_path,
    ]
    return [t5_cmd, vae_cmd]


def gen_train_command(cfg: TrainConfig, output_dir: Path, dataset_toml: Path) -> list[str]:
    """Build the accelerate launch command for musubi-tuner."""
    p = cfg.preset
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", p.mixed_precision,
        str(MUSUBI_ROOT / "src" / "musubi_tuner" / "wan_train_network.py"),
        # Task
        "--task", "t2v-A14B",
        # Model weights
        "--dit", cfg.dit_low_path,
        "--t5", cfg.t5_path,
        "--vae", cfg.vae_path,
        # Dataset
        "--dataset_config", str(dataset_toml),
        # Attention + precision
        f"--{p.attention}",
        "--mixed_precision", p.mixed_precision,
        # Memory
        f"--blocks_to_swap", str(p.blocks_to_swap),
        # Network
        "--network_module", "networks.lora_wan",
        "--network_dim", str(p.network_dim),
        "--network_alpha", str(p.network_alpha),
        # Optimizer
        "--optimizer_type", p.optimizer,
        "--learning_rate", str(p.learning_rate),
        # Timesteps
        "--timestep_sampling", p.timestep_sampling,
        "--discrete_flow_shift", str(p.discrete_flow_shift),
        # Training duration
        "--max_train_epochs", str(cfg.max_train_epochs),
        "--save_every_n_epochs", str(cfg.save_every_n_epochs),
        "--seed", str(cfg.seed),
        # I/O
        "--output_dir", str(output_dir),
        "--output_name", cfg.lora_name,
        # Data loader
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
    ]
    if p.fp8_base:
        cmd.append("--fp8_base")
    if p.gradient_checkpointing:
        cmd.append("--gradient_checkpointing")
    if p.gradient_checkpointing_cpu_offload:
        cmd.append("--gradient_checkpointing_cpu_offload")
    if cfg.dit_high_path:
        cmd += ["--dit_high_noise", cfg.dit_high_path, "--timestep_boundary", "0.875"]
    # T4.A: resume-from-state
    if cfg.resume_state:
        cmd += ["--resume", cfg.resume_state, "--save_state"]
    # T4.H: advanced optimizer args (only emit if explicitly set)
    if cfg.lr_scheduler:
        cmd += ["--lr_scheduler", cfg.lr_scheduler]
    if cfg.lr_warmup_steps is not None:
        cmd += ["--lr_warmup_steps", str(cfg.lr_warmup_steps)]
    if cfg.max_grad_norm is not None:
        cmd += ["--max_grad_norm", str(cfg.max_grad_norm)]
    if cfg.weight_decay is not None:
        cmd += ["--weight_decay", str(cfg.weight_decay)]
    if cfg.sample_every_n_steps and cfg.sample_prompts:
        sample_file = output_dir / "sample_prompts.txt"
        sample_file.write_text("\n".join(cfg.sample_prompts))
        cmd += [
            "--sample_every_n_steps", str(cfg.sample_every_n_steps),
            "--sample_prompts", str(sample_file),
        ]
    cmd += p.extra_args
    return cmd


def gen_launch_script(cmd: list[str], output_dir: Path, venv_python: str) -> Path:
    """Write a shell script that activates the venv + runs training (Flux-Gym pattern)."""
    venv_dir = Path(venv_python).resolve().parent.parent
    script_path = output_dir / "train.sh"
    # cmd[0] is 'accelerate'; resolve it from the venv
    cmd_str = " \\\n    ".join(
        [str(venv_dir / "bin" / "accelerate")] + cmd[1:]
    )
    script_path.write_text(
        "#!/bin/bash\n"
        f"# BASI WAN K3N0B1 generated training script for: {output_dir.name}\n"
        "set -e\n"
        f"cd {MUSUBI_ROOT}\n"
        f"source {venv_dir}/bin/activate\n"
        "exec " + cmd_str + "\n"
    )
    script_path.chmod(0o755)
    return script_path


def prepare_training_run(cfg: TrainConfig, venv_python: str | None = None) -> dict:
    """One-shot: create output dir, write TOML + cache scripts + train script."""
    venv_python = venv_python or os.environ.get("BASI_VENV_PYTHON",
                                                "/home/ryot/venvs/basi/bin/python")
    venv_dir = Path(venv_python).resolve().parent.parent
    output_dir = OUTPUTS_ROOT / cfg.lora_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_toml = gen_dataset_toml(cfg, output_dir / "dataset.toml")
    cmd = gen_train_command(cfg, output_dir, dataset_toml)
    script_path = gen_launch_script(cmd, output_dir, venv_python)

    # Cache precompute (T5 + VAE) — must run BEFORE training. Wrapped into one
    # shell script for one-click prep from the UI.
    cache_cmds = gen_cache_commands(cfg, dataset_toml)
    cache_script_path = output_dir / "cache.sh"
    cache_lines = ["#!/bin/bash", f"# BASI WAN K3N0B1 cache precompute for: {cfg.lora_name}",
                   "set -e", f"cd {MUSUBI_ROOT}", f"source {venv_dir}/bin/activate"]
    for c in cache_cmds:
        cache_lines.append(" \\\n    ".join([str(venv_dir / "bin" / "python")] + c[1:]))
    cache_script_path.write_text("\n".join(cache_lines) + "\n")
    cache_script_path.chmod(0o755)

    return {
        "output_dir": str(output_dir),
        "dataset_toml": str(dataset_toml),
        "cache_script": str(cache_script_path),
        "train_script": str(script_path),
        "command": cmd,
    }


def spawn_training(script_path: str) -> subprocess.Popen:
    """Spawn the training subprocess and return Popen handle for streaming."""
    return subprocess.Popen(
        ["bash", str(script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
    )
