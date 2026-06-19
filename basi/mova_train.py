"""Subprocess wrapper for MOVA joint-A/V LoRA training — the MOVA side of the gym.

Mirrors basi/train.py (the Wan path) but targets MOVA's OWN trainer instead of musubi:
  - dataset convention is the SAME as the Wan gym: a dir of .mp4 + .txt caption pairs
    (autosplit/mova_data writes the .mp4; the gym captioner writes the .txt). The one
    format step is assembling MOVA's metadata.json FROM those pairs (gen_mova_metadata),
    since MOVA's VideoAudioDataset reads metadata.json (not .txt).
  - the trainer is tools/mova_train_runner.py (the validated NF4-resident trainer = MOVA's
    equivalent of musubi's wan_train_network.py), launched via a generated mova_train.sh that
    sets the CUDA-13 LD_LIBRARY_PATH bitsandbytes/torchcodec need (see mova-m1-setup-state memo).
  - presets come from basi.presets.MovaPreset (240p / NF4 / rank by VRAM tier).

Flow (mirror of prepare_training_run): output dir -> metadata.json -> launch script -> spawn.
There is NO separate cache-precompute script here: the runner caches text per-caption + evicts
the encoders internally (M2). ASCII.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .presets import MovaPreset, auto_select_mova
from .eta import human as _human
from .mova_data import plan_training_schedule

BASI_ROOT = Path(__file__).resolve().parent.parent
MOVA_RUNNER = BASI_ROOT / "tools" / "mova_train_runner.py"
OUTPUTS_ROOT = BASI_ROOT / "outputs"

# --- MOVA training pre-flight estimate -------------------------------------------------
# MEASURED this session on an RTX 4090 @ 240p NF4 rank16 (mova_24g): steady-state
# ~10.5 s/step @49f and ~13 s/step @81f -> the step is compute-bound and ~linear in frame
# count (s/step ~= 6.7 + 0.078*frames). VRAM peak ~16GB @81f (NF4 base ~10GB resident +
# activations). These give a ROUGH pre-flight so the user knows feasibility + time BEFORE
# committing; a real run's first steps are the truth. Numbers are 4090-derived; slower
# cards run proportionally slower (we only have 4090 data -> labelled in the UI).
_MOVA_SPS_INTERCEPT = 6.7       # s/step extrapolated to 0 frames (linear fit)
_MOVA_SPS_PER_FRAME = 0.078     # s/step added per frame
_MOVA_VRAM_BASE_GB = 10.0       # resident NF4 14B video tower + audio_dit + bridge
_MOVA_VRAM_ACT_AT_81 = 6.0      # activation VRAM at 81f (16GB measured peak - 10 base)


def estimate_mova_training(n_clips: int, frames: int, epochs: int,
                           repeats: int = 1, vram_gb: int = 24) -> dict:
    """Rough MOVA training cost (pure; 4090 @240p-NF4 basis). n_clips<=0 means the
    dataset isn't known yet -> total/converge are None (s/step + VRAM still valid)."""
    frames = max(17, int(frames)); epochs = max(1, int(epochs)); repeats = max(1, int(repeats))
    s_per_step = _MOVA_SPS_INTERCEPT + _MOVA_SPS_PER_FRAME * frames
    vram_peak = _MOVA_VRAM_BASE_GB + _MOVA_VRAM_ACT_AT_81 * (frames / 81.0)
    out = {"s_per_step": s_per_step, "vram_peak_gb": vram_peak,
           "desktop_free_gb": max(0.0, float(vram_gb) - vram_peak),
           "steps_per_epoch": None, "total_steps": None, "total_s": None,
           "converge_lo_s": None, "converge_hi_s": None}
    if n_clips and n_clips > 0:
        spe = int(n_clips) * repeats
        epoch_s = spe * s_per_step
        out.update(steps_per_epoch=spe, total_steps=spe * epochs,
                   total_s=spe * epochs * s_per_step,
                   converge_lo_s=4 * epoch_s, converge_hi_s=min(8, epochs) * epoch_s)
    return out


def format_mova_estimate_md(n_clips: int, frames: int, epochs: int, repeats: int,
                            preset_name: str, vram_gb: int = 24) -> str:
    """One-line markdown pre-flight for the Gym MOVA panel."""
    e = estimate_mova_training(n_clips, frames, epochs, repeats, vram_gb)
    head = (f"**MOVA estimate** — ~{e['s_per_step']:.0f} s/step at {frames}f · "
            f"~{e['vram_peak_gb']:.0f} GB VRAM (~{e['desktop_free_gb']:.0f} GB free for your desktop)")
    if e["total_steps"] is None:
        body = " · _set a dataset to see total time_"
    else:
        body = (f" · {n_clips} clips x{repeats} x {epochs} ep = {e['total_steps']:,} steps "
                f"-> ~{_human(e['total_s'])} full, ~{_human(e['converge_lo_s'])}-"
                f"{_human(e['converge_hi_s'])} to a usable checkpoint (A/V LoRAs usually "
                f"converge by epoch 4-8)")
    return (head + body + f" · {preset_name}.\n\n_Rough — measured on an RTX 4090 @240p NF4; "
            "slower cards take longer. The first few steps of a real run are the truth._")
# MOVA runs in its own WSL conda env (torchcodec+bitsandbytes Linux-first); the launcher must
# put the env's CUDA-13 nvidia libs on LD_LIBRARY_PATH. Overridable via env for portability.
DEFAULT_MOVA_PYTHON = os.environ.get(
    "MOVA_VENV_PYTHON", str(Path.home() / "miniforge3" / "envs" / "mova" / "bin" / "python"))


@dataclass
class MovaTrainConfig:
    """Everything BASI needs to fire one MOVA A/V LoRA run (parallels TrainConfig)."""
    lora_name: str
    dataset_dir: str                 # dir of .mp4 + .txt caption pairs (gym convention)
    preset: MovaPreset
    # 4:3 sources (e.g. Moral Orel) -> 240x320; 16:9 -> 240x416. Must match the clip res.
    target_resolution: tuple[int, int] = (240, 320)   # (height, width)
    target_frames: int = 81          # capped to preset.max_target_frames
    max_train_epochs: int = 16
    save_every_n_epochs: int = 1   # sample + checkpoint EVERY epoch (Flux-Gym parity)
    num_repeats: int = 1             # small datasets: iterate each clip N times/epoch
    # When True (default), num_repeats + max_train_epochs are AUTO-DERIVED from the dataset's clip
    # count (plan_training_schedule) so ANY size lands near the ~2400-step target: small sets get
    # more repeats, large sets repeats=1. This is the "correct for all users" fix -- a fixed
    # num_repeats=1 badly undertrains a small set (30 clips x16 ep = 480 steps). Set False to honor
    # the explicit num_repeats/max_train_epochs above (advanced/manual control).
    auto_schedule: bool = True
    learning_rate: float | None = None   # None -> preset.learning_rate
    rank: int | None = None              # None -> preset.rank
    base: str | None = None              # None -> preset.base ("nf4"|"fp8")
    trigger_word: str | None = None
    # Sample prompts rendered (A/V) at BASELINE + every checkpoint (Wan-gym parity:
    # --sample_prompts + --sample_at_first). Include several, a couple OFF-style (no trigger)
    # to see baseline behavior + style bleed. None -> no sampling.
    sample_prompts: list[str] | None = None
    # Inline per-epoch A/V sampling forces the model OFF the GPU and back (to hand the card to the
    # isolated sampler); on a 24GB card with the NF4 81f config that CPU<->GPU round-trip bloats the
    # resumed peak 16.2->~20GB and OOMs at the FIRST epoch boundary (measured 2026-06-16). So default
    # OFF: train checkpoint-only (stable steady-state), then sample the saved checkpoints POST-HOC
    # (tools/mova_sample.py --lora <checkpoint_stepN>) to pick the best epoch. Enable only on a big
    # card (40GB+) where the round-trip fits. The sample_prompts file is still written either way.
    inline_sample: bool = False
    seed: int = 42


_STEP_RE = re.compile(
    r"step (\d+)/(\d+) ([\d.]+)s \| win50 loss=([\d.]+) v=([\d.]+) a=([\d.]+) "
    r"\| lr=([\d.eE+-]+) .*?peak=([\d.]+)GB.*?elapsed=([\d.]+)m.*?finite=(\w+)")
_CKPT_RE = re.compile(r"checkpoint_step(\d+)")


def parse_mova_training_log(log_path) -> dict:
    """Parse a MOVA train.log into per-step metrics + a summary, for the Gym's training-analytics
    display (so users SEE the convergence curve and make their own stop call). Pure stdlib regex.

    Returns {points: [{step,total,sec,loss,v_loss,a_loss,lr,peak_gb,elapsed_min,finite}],
             checkpoint_steps: [int,...],
             summary: {last_step,total_steps,s_per_step,peak_gb,best_step,best_loss,
                       steps_per_epoch,best_epoch,audio_overtrain_step}}.
    audio_overtrain_step = first logged step after the total-loss minimum where audio loss has
    risen meaningfully (the 'audio overtrains first' signal measured 2026-06-16). None if absent.
    Empty/missing log -> empty points + None summary fields (never raises)."""
    p = Path(log_path)
    points: list[dict] = []
    ckpts: list[int] = []
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            m = _STEP_RE.search(line)
            if m:
                points.append({
                    "step": int(m.group(1)), "total": int(m.group(2)), "sec": float(m.group(3)),
                    "loss": float(m.group(4)), "v_loss": float(m.group(5)),
                    "a_loss": float(m.group(6)), "lr": float(m.group(7)),
                    "peak_gb": float(m.group(8)), "elapsed_min": float(m.group(9)),
                    "finite": m.group(10) == "True"})
            elif "Saved" in line and "checkpoint_step" in line:
                cm = _CKPT_RE.search(line)
                if cm:
                    ckpts.append(int(cm.group(1)))
    ckpts = sorted(set(ckpts))
    steps_per_epoch = ckpts[0] if ckpts else None      # save-every == 1 epoch (gym convention)
    summary = {"last_step": None, "total_steps": None, "s_per_step": None, "peak_gb": None,
               "best_step": None, "best_loss": None, "steps_per_epoch": steps_per_epoch,
               "best_epoch": None, "audio_overtrain_step": None}
    if points:
        last = points[-1]
        summary["last_step"] = last["step"]
        summary["total_steps"] = last["total"]
        summary["peak_gb"] = max(pt["peak_gb"] for pt in points)
        # sustained s/step = elapsed wall over steps done (more honest than any single step's timer)
        summary["s_per_step"] = round(last["elapsed_min"] * 60.0 / max(last["step"], 1), 1)
        best = min(points, key=lambda d: d["loss"])
        summary["best_step"] = best["step"]
        summary["best_loss"] = round(best["loss"], 4)
        if steps_per_epoch:
            summary["best_epoch"] = max(1, round(best["step"] / steps_per_epoch))
        # audio-overtrain signal: first post-minimum point whose a_loss exceeds the best point's
        # a_loss by >2% (video can keep improving while audio degrades).
        for pt in points:
            if pt["step"] > best["step"] and pt["a_loss"] > best["a_loss"] * 1.02:
                summary["audio_overtrain_step"] = pt["step"]
                break
    return {"points": points, "checkpoint_steps": ckpts, "summary": summary}


def format_mova_analytics_md(parsed: dict) -> str:
    """One-glance markdown summary of a run's training analytics for the Gym panel."""
    s = parsed.get("summary", {})
    if not s.get("last_step"):
        return "_No training metrics yet — start a run to see the loss curve + convergence._"
    spe = s.get("steps_per_epoch")
    ep = (lambda st: f"epoch {max(1, round(st / spe))}" if spe and st else f"step {st}")
    lines = [
        f"**Training analytics** — {s['last_step']}/{s['total_steps']} steps "
        f"(~{s['s_per_step']}s/step, peak {s['peak_gb']:.1f} GB)",
        f"- **Best (lowest loss): {ep(s['best_step'])}** (step {s['best_step']}, loss {s['best_loss']}) "
        f"-> the checkpoint to prefer.",
    ]
    if s.get("audio_overtrain_step"):
        lines.append(f"- Audio loss starts rising at {ep(s['audio_overtrain_step'])} "
                     f"(step {s['audio_overtrain_step']}) — the audio tower overtrains *before* the "
                     f"video; epochs past here usually sound worse. Sample to confirm.")
    if s.get("steps_per_epoch"):
        lines.append(f"- Checkpoints every {s['steps_per_epoch']} steps (per epoch); "
                     f"pick by the per-epoch samples, not just the loss.")
    return "\n".join(lines)


def gen_mova_metadata(cfg: MovaTrainConfig, output_path: Path) -> Path:
    """Assemble MOVA's metadata.json from the dataset dir's .mp4 + .txt pairs.

    This is the MOVA analogue of gen_dataset_toml: the gym uses .mp4+.txt everywhere, MOVA's
    loader wants [{"video_path","caption"}]. Reads each clip's .txt (the gym captioner output);
    prepends the trigger word (matching the Wan gym, where the trigger is prepended post-caption);
    clips with no .txt get an empty caption. Writes metadata.json INTO dataset_dir (where the
    VideoAudioDataset + the .mp4s live)."""
    ds = Path(cfg.dataset_dir)
    # Exclude the _dropped/ curation quarantine (tools/mova_curate.py --apply moves flagged clips
    # there, reversibly). Without this, curated-out clips would silently re-enter training.
    vids = sorted([p for p in ds.glob("**/*.mp4") if "_dropped" not in p.parts])
    items = []
    for v in vids:
        txt = v.with_suffix(".txt")
        cap = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
        if cfg.trigger_word and cap and cfg.trigger_word.lower() not in cap.lower():
            cap = f"{cfg.trigger_word}, {cap}"
        elif cfg.trigger_word and not cap:
            cap = cfg.trigger_word
        rel = v.relative_to(ds).as_posix()
        items.append({"video_path": rel, "caption": cap})
    output_path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return output_path


def _count_clips(dataset_dir: str) -> int:
    # Excludes _dropped/ so the auto-schedule sees the CURATED clip count (see gen_mova_metadata).
    return len([p for p in Path(dataset_dir).glob("**/*.mp4") if "_dropped" not in p.parts])


def gen_mova_train_command(cfg: MovaTrainConfig, output_dir: Path, mova_python: str) -> list[str]:
    """Build the runner invocation (parallels gen_train_command). The runner is step-based;
    we convert epochs -> steps (epochs * clips * num_repeats), the gym's epoch-first convention."""
    p = cfg.preset
    h, w = cfg.target_resolution
    frames = min(cfg.target_frames, p.max_target_frames)
    n_clips = max(1, _count_clips(cfg.dataset_dir))
    steps = cfg.max_train_epochs * n_clips * cfg.num_repeats
    save_every = max(1, cfg.save_every_n_epochs * n_clips * cfg.num_repeats)
    cmd = [
        mova_python, str(MOVA_RUNNER),
        "--dataset", str(cfg.dataset_dir),
        "--output", str(output_dir),
        "--height", str(h), "--width", str(w), "--frames", str(frames),
        "--rank", str(cfg.rank or p.rank),
        "--alpha", str(getattr(p, "alpha", 0) or 0),
        "--lr", str(cfg.learning_rate or p.learning_rate),
        "--base", str(cfg.base or p.base),
        "--steps", str(steps),
        "--save-every", str(save_every),
    ]
    # v3 style-push levers from the preset (measured 2026-06-19): FFN LoRA on the video tower, and
    # FREEZE the audio tower (audio is solved + overtrains; AV-sync lives in the trainable bridge).
    if getattr(p, "lora_ffn", False):
        cmd.append("--lora-ffn")
    if not getattr(p, "train_audio_tower", True):
        cmd.append("--freeze-audio")
    if cfg.sample_prompts:
        # Always persist the prompts (for POST-HOC sampling of the saved checkpoints). Pass them to
        # the trainer for INLINE per-epoch sampling only if inline_sample is on -- otherwise add
        # --no-sample so training is checkpoint-only (avoids the epoch-boundary OOM; see the field
        # comment). Post-hoc: tools/mova_sample.py --base <CKPT> --lora <output>/checkpoint_stepN ...
        sample_file = output_dir / "sample_prompts.txt"
        sample_file.write_text("\n".join(cfg.sample_prompts) + "\n", encoding="utf-8")
        if cfg.inline_sample:
            cmd += ["--sample-prompts", str(sample_file)]
        else:
            cmd += ["--sample-prompts", str(sample_file), "--no-sample"]
    else:
        cmd += ["--no-sample"]
    return cmd


def gen_mova_launch_script(cmd: list[str], output_dir: Path, mova_python: str) -> Path:
    """Write mova_train.sh: sets the CUDA-13 LD_LIBRARY_PATH (bitsandbytes/torchcodec) + alloc
    conf, then runs the trainer. Mirrors gen_launch_script (Unix; MOVA is WSL/Linux-only)."""
    env_root = Path(mova_python).resolve().parent.parent          # .../envs/mova
    sp = env_root / "lib" / "python3.13" / "site-packages"
    script = output_dir / "mova_train.sh"
    cmd_str = " \\\n    ".join([str(x) for x in cmd])
    # Resolve the MOVA-360p weights portably (BASIWAN_CKPT_DIR or repo/checkpoints) and pin it
    # into the script so the spawned runner finds the weights regardless of the launching shell's
    # env (the runner reads MOVA_CKPT). No dev-box path baked in.
    _ckpt_base = Path(os.environ.get("BASIWAN_CKPT_DIR", str(BASI_ROOT / "checkpoints")))
    _mova_ckpt = _ckpt_base / "MOVA-360p"
    script.write_text(
        "#!/bin/bash\n"
        f"# BASI generated MOVA A/V training script for: {output_dir.name}\n"
        "set -e\n"
        f'SP="{sp}"\n'
        'NVLIBS=$(ls -d "$SP"/nvidia/*/lib 2>/dev/null | tr "\\n" ":")\n'
        'export LD_LIBRARY_PATH="${NVLIBS}${SP}/torch/lib:${LD_LIBRARY_PATH}"\n'
        f'export MOVA_CKPT="{_mova_ckpt}"\n'
        # NO expandable_segments: it pins large growable segments that torch.cuda.empty_cache() can't
        # release (any live block holds the whole segment) -> reserved stuck near 24GB, desktop frozen.
        # Our per-step shape is fixed (one clip), so expandable buys nothing; plain gc lets empty_cache
        # actually return the slack (measured: 168MB -> 7160MB free). See mova-m2 memory note.
        # max_split_size_mb:128 caps block-split size to curb FRAGMENTATION (the OOM cause at the
        # sample->resume boundary 2026-06-16: the model CPU<->GPU round-trip re-allocated the 10GB NF4
        # base fragmented, pushing the resumed peak 16.2->20.6GB). Unlike expandable_segments it does
        # NOT bypass set_per_process_memory_fraction, so the desktop-headroom cap still holds.
        'export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:128"\n'
        "export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false\n"
        f"{cmd_str}\n", encoding="utf-8")
    script.chmod(0o755)
    return script


def prepare_mova_training_run(cfg: MovaTrainConfig, mova_python: str | None = None) -> dict:
    """One-shot: output dir + metadata.json + launch script (mirror of prepare_training_run)."""
    mova_python = mova_python or DEFAULT_MOVA_PYTHON
    output_dir = OUTPUTS_ROOT / cfg.lora_name
    output_dir.mkdir(parents=True, exist_ok=True)
    n_clips = _count_clips(cfg.dataset_dir)
    # "Correct for all users": size-adaptive repeats/epochs so the run lands near the ~2400-step
    # target regardless of dataset size (mutates cfg before the command is built). Skipped if the
    # dataset is empty (n_clips==0) or the user opted out (auto_schedule=False).
    schedule = None
    if cfg.auto_schedule and n_clips > 0:
        schedule = plan_training_schedule(n_clips)
        cfg.num_repeats = schedule["repeats"]
        cfg.max_train_epochs = schedule["epochs"]
    metadata = gen_mova_metadata(cfg, Path(cfg.dataset_dir) / "metadata.json")
    cmd = gen_mova_train_command(cfg, output_dir, mova_python)
    script_path = gen_mova_launch_script(cmd, output_dir, mova_python)
    return {
        "output_dir": str(output_dir),
        "metadata": str(metadata),
        "train_script": str(script_path),
        "command": cmd,
        "clips": n_clips,
        "schedule": schedule,
    }


def spawn_mova_training(script_path: str) -> subprocess.Popen:
    """Spawn the MOVA training subprocess (mirror of spawn_training)."""
    return subprocess.Popen(
        ["bash", str(script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
