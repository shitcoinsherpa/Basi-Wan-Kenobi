"""Subprocess wrapper for musubi-tuner Wan2.2 LoRA training (Flux-Gym pattern)."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
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
    # 16 epochs balances convergence with checkpoint diversity: 8 evaluable
    # checkpoints at save_every=2. Overfit symptoms show ~ep 10-12 on small
    # datasets; 16 is the sweet spot for first-attempt LoRAs.
    max_train_epochs: int = 16
    save_every_n_epochs: int = 2
    # Per-EPOCH sampling is the primary monitoring signal — one render per
    # epoch matches the save_every=2 checkpoint cadence so every checkpoint
    # has visual evidence. Step-based sampling (below) is the optional
    # extra-frequency knob; 0 disables it.
    sample_every_n_epochs: int | None = 1
    sample_every_n_steps: int = 0
    sample_prompts: list[str] | None = None
    trigger_word: str | None = None
    seed: int = 42
    # Optional resume from a saved state directory (musubi --resume; musubi
    # saves state every N epochs when --save_state is on).
    resume_state: str | None = None
    # Per-image repeats — applies to the [[datasets]] block. >1 means each
    # clip is iterated N times per epoch, useful for small datasets.
    num_repeats: int = 1
    # Enable musubi auto-bucketing in the [general] block so the trainer
    # groups mixed-resolution clips into per-bucket batches rather than padding
    # to the nominal target_resolution.
    enable_bucket: bool = True
    # cosine_with_restarts prevents late-training overfit (lr decays then
    # spikes back); warmup 100 steps avoids early LR shock on small batches.
    # Override via UI for advanced tuning.
    lr_scheduler: str | None = "cosine_with_restarts"
    lr_warmup_steps: int | None = 100
    max_grad_norm: float | None = None    # 0 = disable clip_grad_norm
    weight_decay: float | None = None
    # Source video fps for musubi's 16fps resampling. None = treat frames
    # as-is (correct for 16fps sources); set to the clips' actual fps (e.g.
    # 24.0 for TV rips) so a 33-frame window spans the right real-time span
    # (2s instead of 1.4s). Schema key is `source_fps`, not `fps` (musubi's
    # validator rejects `fps` with "extra keys not allowed @ datasets[0].fps").
    source_fps: float | None = None
    # Timestep windowing per expert (musubi docs/wan.md): the low expert
    # serves t<875 at inference, high serves t>=875. Training a single expert
    # across the full range wastes gradient on timesteps it never serves —
    # worse with shift>1, which skews sampling HIGH. preserve_distribution_shape
    # keeps the shift distribution's shape inside the window instead of
    # renormalizing it flat.
    min_timestep: int | None = None
    max_timestep: int | None = None
    preserve_distribution_shape: bool = False
    # "uniform" + frame_sample=2 trains on 2 evenly-spaced windows per clip.
    # The "head" mode trains on ONLY the first target_frames window — for 2-5s
    # clips with 33-frame windows that silently discards ~35-45% of curated
    # footage (musubi image_video_dataset.py: head = one fixed window at cache
    # time).
    frame_extraction: str = "uniform"
    frame_sample: int = 2
    # Optional stills arm (official Wan recipe is 30-50 stills + 10-20 clips):
    # images carry style/detail cheaply, videos carry motion. Emitted as a
    # second [[datasets]] block.
    image_dataset_dir: str | None = None
    image_num_repeats: int = 1
    # Per-run overrides of preset values (None = use preset). Lets a style
    # run use dim=alpha=16 (community trend for Wan2.2 style; smaller
    # overfit surface on small datasets) without mutating shared presets.
    network_dim_override: int | None = None
    network_alpha_override: int | None = None
    discrete_flow_shift_override: float | None = None


def gen_dataset_toml(cfg: TrainConfig, output_path: Path) -> Path:
    """Write musubi-tuner dataset config (TOML).

    Emits [[datasets]] with cfg.num_repeats and toggles [general].enable_bucket
    so musubi auto-groups mixed-resolution clips into per-bucket batches rather
    than padding all to target_resolution.
    """
    res_w, res_h = cfg.target_resolution
    dataset_block = {
        "resolution": [res_w, res_h],
        "target_frames": [cfg.target_frames],
        "frame_extraction": cfg.frame_extraction,
        "video_directory": cfg.dataset_dir,
        "caption_extension": ".txt",
        "max_frames": cfg.target_frames,
        "batch_size": cfg.preset.batch_size,
        "num_repeats": cfg.num_repeats,
    }
    if cfg.frame_extraction == "uniform":
        dataset_block["frame_sample"] = cfg.frame_sample
    if cfg.source_fps:
        dataset_block["source_fps"] = float(cfg.source_fps)
    datasets = [dataset_block]
    if cfg.image_dataset_dir:
        datasets.append({
            "resolution": [res_w, res_h],
            "image_directory": cfg.image_dataset_dir,
            "caption_extension": ".txt",
            "batch_size": cfg.preset.batch_size,
            "num_repeats": cfg.image_num_repeats,
        })
    general_block = {
        "caption_extension": ".txt",
        "batch_size": cfg.preset.batch_size,
    }
    if cfg.enable_bucket:
        general_block["enable_bucket"] = True
    toml_dict = {
        "general": general_block,
        "datasets": datasets,
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
        # Incremental re-cache. VAE-encoding video is the expensive half of
        # caching; a re-cache after ADDING clips would otherwise re-encode the
        # whole set. --skip_existing keys on the latent cache filename, which
        # encodes resolution+frames (...-033_0720x0544_wan.safetensors) and
        # depends ONLY on the clip pixels -- not captions -- so skipping an
        # existing latent is safe for every common edit (add/remove clips, edit
        # captions, change res -> new filename -> re-encoded). The one edge it
        # does NOT catch is an in-place pixel swap that keeps the same filename
        # AND dims (rare; the autosplit pipeline emits unique names) -- delete
        # that clip's cache to force a redo. Deliberately NOT applied to the T5
        # text cache above, which keys on clip name not caption content and
        # would serve a STALE embedding after a caption edit (captions are what
        # users iterate; T5 over short captions at batch 16 is cheap anyway).
        "--skip_existing",
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
        "--network_dim", str(cfg.network_dim_override or p.network_dim),
        "--network_alpha", str(cfg.network_alpha_override or p.network_alpha),
        # Optimizer
        "--optimizer_type", p.optimizer,
        "--learning_rate", str(p.learning_rate),
        # Timesteps
        "--timestep_sampling", p.timestep_sampling,
        "--discrete_flow_shift",
        str(cfg.discrete_flow_shift_override or p.discrete_flow_shift),
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
    # Timestep windowing for single-expert runs (musubi docs/wan.md:
    # low expert 0-875, high expert 875-1000).
    if cfg.min_timestep is not None:
        cmd += ["--min_timestep", str(cfg.min_timestep)]
    if cfg.max_timestep is not None:
        cmd += ["--max_timestep", str(cfg.max_timestep)]
    if cfg.preserve_distribution_shape:
        cmd.append("--preserve_distribution_shape")
    # resume-from-state
    if cfg.resume_state:
        cmd += ["--resume", cfg.resume_state, "--save_state"]
    # advanced optimizer args (only emit if explicitly set)
    if cfg.lr_scheduler:
        cmd += ["--lr_scheduler", cfg.lr_scheduler]
    if cfg.lr_warmup_steps is not None:
        cmd += ["--lr_warmup_steps", str(cfg.lr_warmup_steps)]
    if cfg.max_grad_norm is not None:
        cmd += ["--max_grad_norm", str(cfg.max_grad_norm)]
    if cfg.weight_decay is not None:
        cmd += ["--weight_decay", str(cfg.weight_decay)]
    if cfg.sample_prompts and (cfg.sample_every_n_epochs or cfg.sample_every_n_steps):
        sample_file = output_dir / "sample_prompts.txt"
        sample_file.write_text("\n".join(cfg.sample_prompts))
        # --sample_at_first renders before step 1: validates the sampling
        # path immediately and gives the no-LoRA baseline to compare
        # epoch renders against. Both flags verified in musubi's
        # training/parser_common.py.
        cmd += ["--sample_prompts", str(sample_file), "--sample_at_first"]
        if cfg.sample_every_n_epochs:
            cmd += ["--sample_every_n_epochs", str(cfg.sample_every_n_epochs)]
        if cfg.sample_every_n_steps:
            cmd += ["--sample_every_n_steps", str(cfg.sample_every_n_steps)]
    cmd += p.extra_args
    return cmd


def _venv_bin_dir(venv_python: str) -> Path:
    """Cross-platform venv binary directory: Scripts/ on Windows, bin/ on Unix."""
    p = Path(venv_python).resolve()
    # Whichever parent exists is the right one
    if (p.parent.name.lower() == "scripts") or sys.platform == "win32":
        return p.parent  # already Scripts/ on Windows or bin/ on Unix
    return p.parent


def gen_launch_script(cmd: list[str], output_dir: Path, venv_python: str) -> Path:
    """Write a generated launcher that runs accelerate on the train script.

    Cross-platform:
      - Windows: writes `train.bat` that invokes accelerate.exe directly with
        full paths (no activate.bat needed — avoids cmd nesting quirks).
      - Unix:    writes `train.sh` (original Flux-Gym pattern, kept verbatim
        for Linux/macOS users).
    """
    bin_dir = _venv_bin_dir(venv_python)
    is_win = sys.platform == "win32"
    if is_win:
        accelerate = bin_dir / "accelerate.exe"
        script_path = output_dir / "train.bat"
        # cmd format on Windows: ^ continuation, no `exec`, no activate.
        # Escape any path with spaces by quoting.
        parts = [f'"{accelerate}"'] + [str(x) for x in cmd[1:]]
        cmd_str = " ^\n    ".join(parts)
        # PYTHONUTF8=1: musubi prints bilingual (Japanese) log messages;
        # the Windows console default cp1252 can't encode them and the
        # trainer DIES on a print() at optimizer setup
        # (UnicodeEncodeError in accelerate.state.print).
        script_path.write_text(
            "@echo off\r\n"
            f"REM BASI WAN KENOBI generated training script for: {output_dir.name}\r\n"
            "set PYTHONUTF8=1\r\n"
            "set PYTHONIOENCODING=utf-8\r\n"
            f"cd /d \"{MUSUBI_ROOT}\"\r\n"
            f"{cmd_str}\r\n",
            encoding="utf-8",
        )
    else:
        accelerate = bin_dir / "accelerate"
        venv_dir = bin_dir.parent
        script_path = output_dir / "train.sh"
        cmd_str = " \\\n    ".join([str(accelerate)] + cmd[1:])
        script_path.write_text(
            "#!/bin/bash\n"
            f"# BASI WAN KENOBI generated training script for: {output_dir.name}\n"
            "set -e\n"
            f"cd {MUSUBI_ROOT}\n"
            f"source {venv_dir}/bin/activate\n"
            "exec " + cmd_str + "\n"
        )
        script_path.chmod(0o755)
    return script_path


def prepare_training_run(cfg: TrainConfig, venv_python: str | None = None) -> dict:
    """One-shot: create output dir, write TOML + cache scripts + train script.

    Cross-platform: emits .bat on Windows, .sh on Unix.
    Default venv_python is platform-aware (Scripts/ on Windows, bin/ on Unix)
    and uses sys.executable as the fallback so the BASI_VENV_PYTHON env var
    can override on dev machines.
    """
    if venv_python is None:
        venv_python = os.environ.get("BASI_VENV_PYTHON", sys.executable)
    is_win = sys.platform == "win32"
    bin_dir = _venv_bin_dir(venv_python)
    output_dir = OUTPUTS_ROOT / cfg.lora_name
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_toml = gen_dataset_toml(cfg, output_dir / "dataset.toml")
    cmd = gen_train_command(cfg, output_dir, dataset_toml)
    script_path = gen_launch_script(cmd, output_dir, venv_python)

    # Cache precompute (T5 + VAE) — must run BEFORE training. Wrapped into one
    # shell script for one-click prep from the UI.
    cache_cmds = gen_cache_commands(cfg, dataset_toml)
    if is_win:
        py_exe = bin_dir / "python.exe"
        cache_script_path = output_dir / "cache.bat"
        lines = ["@echo off",
                 f"REM BASI WAN KENOBI cache precompute for: {cfg.lora_name}",
                 "set PYTHONUTF8=1",          # musubi's bilingual prints vs cp1252
                 "set PYTHONIOENCODING=utf-8",
                 f"cd /d \"{MUSUBI_ROOT}\""]
        for c in cache_cmds:
            parts = [f'"{py_exe}"'] + [str(x) for x in c[1:]]
            lines.append(" ^\n    ".join(parts))
            # Abort-on-first-failure: cmd has no `set -e`; check ERRORLEVEL.
            lines.append("if errorlevel 1 exit /b 1")
        cache_script_path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")
    else:
        venv_dir = bin_dir.parent
        cache_script_path = output_dir / "cache.sh"
        lines = ["#!/bin/bash",
                 f"# BASI WAN KENOBI cache precompute for: {cfg.lora_name}",
                 "set -e", f"cd {MUSUBI_ROOT}",
                 f"source {venv_dir}/bin/activate"]
        for c in cache_cmds:
            lines.append(" \\\n    ".join([str(bin_dir / "python")] + c[1:]))
        cache_script_path.write_text("\n".join(lines) + "\n")
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
