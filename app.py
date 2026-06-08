"""BASI WAN K3N0B1 — Flux-Gym-style UI for Wan2.2 video LoRA training.

Three-column layout (Configure → Dataset → Train):
  - Configure: LoRA name, trigger word, expert (low/high/both), preset, steps, sample prompt
  - Dataset: drop videos, auto-caption (Qwen2.5-VL on-demand), bucket histogram
  - Train: cache precompute → train (gr.Textbox live log) → preview button → live sample gallery

Backend: musubi-tuner (CLI subprocess pattern, identical to Flux-Gym wrapping sd-scripts).
Output: .safetensors LoRAs in outputs/{name}/, drop-in compatible with Faster-Wan2.2 P19 loader.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basi.presets import (
    PRESETS, detect_vram_gb, auto_select,
    detect_capability, recommend_inference_scheme,
)
from basi.dataset import scan_dataset, bucket_distribution
from basi.train import TrainConfig, prepare_training_run, OUTPUTS_ROOT
from basi.preview import PreviewSpec, generate_preview, latest_lora_in
from basi.caption import caption_dataset, select_model_for_vram

BASI_ROOT = Path(__file__).resolve().parent
WORKSPACES = BASI_ROOT / "outputs"

# Wan2.2 default ckpts. Resolution order:
# 1. BASIWAN_CKPT_DIR (canonical env name)
# 2. BASI_CKPT_DIR (legacy env name, back-compat)
# 3. ./checkpoints/Wan2.2-T2V-A14B (relative to app.py; cross-platform default
#    that works on Pinokio Win/Linux/macOS without manual env config).
DEFAULT_CKPT_DIR = (
    os.environ.get("BASIWAN_CKPT_DIR")
    or os.environ.get("BASI_CKPT_DIR")
    or str(BASI_ROOT / "checkpoints" / "Wan2.2-T2V-A14B")
)
DEFAULT_DIT_HIGH = f"{DEFAULT_CKPT_DIR}/high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors"
DEFAULT_DIT_LOW = f"{DEFAULT_CKPT_DIR}/low_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors"
DEFAULT_T5 = f"{DEFAULT_CKPT_DIR}/models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_VAE = f"{DEFAULT_CKPT_DIR}/Wan2.1_VAE.pth"

# Common ComfyUI lora dirs we'll probe for the "auto-link" feature.
COMFYUI_LORA_CANDIDATES = [
    Path.home() / "ComfyUI/models/loras",
    Path("/mnt/d/Ai/ComfyUI/models/loras"),
    Path("/mnt/d/ComfyUI/models/loras"),
]


def _detect_initial_preset() -> str:
    chosen = auto_select(detect_vram_gb())
    # Map the Preset back to its dict key for the dropdown
    for key, p in PRESETS.items():
        if p is chosen:
            return key
    return "24g"


def _ingest_uploaded_videos(files: list, workspace_name: str) -> tuple[str, str]:
    """Copy uploaded files into outputs/{workspace}/dataset/ and rescan."""
    if not workspace_name or not files:
        return "no files / no workspace", ""
    ds_dir = WORKSPACES / workspace_name / "dataset"
    ds_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        src = Path(f.name if hasattr(f, "name") else f)
        if not src.exists():
            continue
        shutil.copy2(src, ds_dir / src.name)
    clips = scan_dataset(str(ds_dir))
    rep = [f"# Dataset scan ({len(clips)} clips)\n"]
    for c in clips:
        cap = "✓ caption" if c.caption_text else "✗ NO CAPTION"
        issues = f" — {', '.join(c.issues)}" if c.issues else ""
        rep.append(f"- `{Path(c.path).name}` {c.width}×{c.height} @{c.fps:.1f}fps × {c.frame_count}f [{cap}]{issues}")
    buckets = bucket_distribution(clips)
    rep.append(f"\n## Bucket distribution ({len(buckets)} buckets)\n")
    for (res, frames), count in sorted(buckets.items(), key=lambda x: -x[1]):
        rep.append(f"- {res[0]}×{res[1]} × {frames}f: **{count} clips**")
    return "\n".join(rep), str(ds_dir)


def _auto_caption(dataset_dir: str, trigger_word: str, progress=gr.Progress()):
    """Run Qwen2.5-VL captioning over the dataset (tier-aware model select)."""
    if not dataset_dir or not Path(dataset_dir).exists():
        return "error: ingest a dataset first"
    videos = [p for p in sorted(Path(dataset_dir).iterdir())
              if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}]
    if not videos:
        return "no videos found in dataset"
    model_id = select_model_for_vram()
    progress(0, desc=f"loading {model_id}")
    results = caption_dataset(
        [str(p) for p in videos],
        trigger_word=(trigger_word.strip() or None),
        skip_existing=True,
    )
    lines = [f"# Auto-caption done ({len(results)} clips, model: `{model_id}`)\n"]
    for path, cap in results.items():
        lines.append(f"- **{Path(path).name}**: {cap[:160]}{'…' if len(cap) > 160 else ''}")
    return "\n".join(lines)


def _generate_scripts(workspace_name: str, dataset_dir: str, preset_key: str,
                      expert_choice: str, trigger_word: str, sample_prompt: str,
                      max_epochs: int, sample_every: int, target_frames: int,
                      resolution_choice: str,
                      num_repeats: int = 1, resume_state: str = "",
                      lr_scheduler: str = "", lr_warmup_steps=None,
                      max_grad_norm=None, weight_decay=None):
    """resolution_choice is one of 'WxH' from the Wan2.2 supported buckets."""
    target_w, target_h = (int(s) for s in resolution_choice.split("x"))
    """Render train.sh + cache.sh without spawning. Returns the script preview."""
    if not workspace_name or not dataset_dir:
        return "error: workspace name + dataset dir required", "", ""
    preset = PRESETS[preset_key]
    # expert_choice maps to musubi's --dit (single-expert path) and optional
    # --dit_high_noise (dual-expert path). Single-expert is the default and
    # trains whichever expert is fed to --dit. "both" enables dual-expert
    # training with timestep_boundary=0.875.
    if expert_choice == "high":
        single_dit = DEFAULT_DIT_HIGH
    else:  # "low" or "both" — "both" still needs a low for --dit
        single_dit = DEFAULT_DIT_LOW
    cfg = TrainConfig(
        lora_name=workspace_name, dataset_dir=dataset_dir,
        dit_low_path=single_dit,
        dit_high_path=(DEFAULT_DIT_HIGH if expert_choice == "both" else None),
        t5_path=DEFAULT_T5, vae_path=DEFAULT_VAE, preset=preset,
        target_resolution=(target_w, target_h), target_frames=target_frames,
        max_train_epochs=max_epochs, sample_every_n_steps=sample_every,
        sample_prompts=[sample_prompt] if sample_prompt.strip() else None,
        trigger_word=trigger_word.strip() or None,
        num_repeats=int(num_repeats),
        resume_state=resume_state.strip() or None,
        lr_scheduler=lr_scheduler.strip() or None,
        lr_warmup_steps=int(lr_warmup_steps) if lr_warmup_steps else None,
        max_grad_norm=float(max_grad_norm) if max_grad_norm is not None else None,
        weight_decay=float(weight_decay) if weight_decay is not None else None,
    )
    run = prepare_training_run(cfg)
    md = (
        f"# Scripts ready\n\n"
        f"**1. Cache (T5 + VAE) — REQUIRED before training:**\n```\nbash {run['cache_script']}\n```\n\n"
        f"**2. Train:**\n```\nbash {run['train_script']}\n```\n\n"
        f"Click **Run cache** then **Run training** to execute in-app, or run the scripts manually."
    )
    return md, run["cache_script"], run["train_script"]


_active_procs: dict[str, subprocess.Popen] = {}


def _spawn_script(script_path: str):
    """Stream a generated .sh script's output. Registers under script path
    so a separate Stop button can signal the same Popen."""
    if not script_path or not Path(script_path).exists():
        yield f"error: script not found: {script_path}"
        return
    # Kill any prior run of the same script (user re-clicked Run)
    prior = _active_procs.pop(script_path, None)
    if prior is not None and prior.poll() is None:
        prior.terminate()
        try:
            prior.wait(timeout=2)
        except subprocess.TimeoutExpired:
            prior.kill()
    proc = subprocess.Popen(
        ["bash", script_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        bufsize=1, text=True,
        preexec_fn=os.setsid,  # process group so we can SIGTERM the whole tree
    )
    _active_procs[script_path] = proc
    buf = []
    yield f"$ bash {script_path}\n"
    for line in proc.stdout:
        buf.append(line.rstrip())
        # Yield growing tail (last 200 lines) so UI shows progress without unbounded growth.
        yield "\n".join(buf[-200:])
    proc.wait()
    _active_procs.pop(script_path, None)
    buf.append(f"\n[exit code {proc.returncode}]")
    yield "\n".join(buf[-200:])


def _stop_script(script_path: str) -> str:
    """SIGTERM (then SIGKILL) the running script's process group."""
    proc = _active_procs.get(script_path)
    if proc is None or proc.poll() is not None:
        return f"no running process for {script_path}"
    import signal
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return "process already gone"
    try:
        proc.wait(timeout=3)
        return f"stopped: {script_path}"
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        return f"force-killed: {script_path}"


def _latest_samples(workspace_name: str):
    """Return list of newest sample videos. Wan2.2 is a video model — musubi
    writes samples to {output_dir}/sample/{ts}_{seed}.mp4 via save_videos_grid
    (ext/musubi-tuner/src/musubi_tuner/training/trainer_base.py:992-993).
    Auto-refresh wired via gr.Timer at the demo level."""
    if not workspace_name:
        return []
    samples_dir = WORKSPACES / workspace_name / "sample"
    if not samples_dir.exists():
        return []
    vids = sorted(samples_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in vids[:8]]


def _generate_preview(workspace_name: str, prompt: str, size: str, frames: int,
                      steps: int, seed: int, expert_choice: str = "low",
                      lora_override = None, extra_loras = None):
    if not workspace_name:
        return None, "error: workspace name required"
    ws = WORKSPACES / workspace_name
    # User-uploaded LoRA wins; falls back to newest in workspace.
    if lora_override is not None:
        lora = Path(lora_override.name if hasattr(lora_override, 'name') else lora_override)
    else:
        lora = latest_lora_in(ws)
    if lora is None or not lora.exists():
        return None, f"no .safetensors LoRA found in {ws} yet — train first or save a checkpoint, or upload one"
    # T4.C: stack extra LoRAs via BASIWAN_USER_LORA_LIST.
    if extra_loras:
        paths = [str(Path(x.name if hasattr(x, 'name') else x)) for x in extra_loras]
        os.environ["BASIWAN_USER_LORA_LIST"] = ",".join([str(lora)] + paths)
    else:
        os.environ.pop("BASIWAN_USER_LORA_LIST", None)
    out_dir = ws / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{lora.stem}_seed{seed}.mp4"
    # lora_expert tells Faster-Wan2.2 which expert (high/low/both) to stack
    # the user LoRA onto — must match the expert that was actually trained.
    spec = PreviewSpec(
        lora_path=str(lora), prompt=prompt, output_path=str(out_path),
        size=size, frame_num=frames, sample_steps=steps, base_seed=seed,
        lora_expert=expert_choice,
    )
    t0 = time.time()
    try:
        path = generate_preview(spec)
    except Exception as e:
        return None, f"preview failed: {e}"
    return str(path), f"rendered in {time.time()-t0:.1f}s → {path.name}"


def _link_to_comfyui(workspace_name: str):
    """Symlink workspace's newest LoRA into the first found ComfyUI loras dir."""
    if not workspace_name:
        return "error: workspace name required"
    lora = latest_lora_in(WORKSPACES / workspace_name)
    if lora is None:
        return "error: no LoRA in workspace yet"
    target_dir = next((p for p in COMFYUI_LORA_CANDIDATES if p.exists()), None)
    if target_dir is None:
        return ("no ComfyUI dir found. Tried: " +
                ", ".join(str(p) for p in COMFYUI_LORA_CANDIDATES))
    link = target_dir / lora.name
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(lora)
    return f"linked: {link} → {lora}"


def build_ui():
    initial_preset = _detect_initial_preset()
    cap = detect_capability()
    vram_gb = cap.vram_gb
    rec_scheme = recommend_inference_scheme(cap)
    # Plain-language hint about what the preview path will use on this GPU.
    scheme_hint = {
        "fp8":     "FP8 prequant (Ada/Hopper hardware FP8 — fastest path)",
        "int8":    "Int8 weight-only prequant (Ampere — no FP8 tensor cores)",
        "gguf_q4": "GGUF Q4_K_M (Turing / AMD ROCm / Apple MPS fallback)",
        "bf16":    "BF16 (no prequant — debug path)",
    }.get(rec_scheme, rec_scheme)

    with gr.Blocks(title="BASI WAN K3N0B1") as demo:
        gr.Markdown("# BASI WAN K3N0B1 — Wan2.2 Video LoRA Trainer")
        gr.Markdown(
            f"**GPU**: {cap.name} ({vram_gb} GB"
            + (f", sm_{cap.compute_major}{cap.compute_minor}"
               if cap.compute_major else "")
            + f") → training preset `{initial_preset}`, "
            f"preview path: {scheme_hint}.  "
            "Backend: musubi-tuner. Output: `.safetensors` in `outputs/<lora_name>/`."
        )
        with gr.Row():
            # COLUMN 1: Configure
            with gr.Column():
                gr.Markdown("## 1 · Configure")
                lora_name = gr.Textbox(
                    label="LoRA name",
                    value="my_first_lora",
                    info="Letters, digits, underscore, hyphen only.",
                )
                lora_name_warning = gr.Markdown(visible=False)
                trigger_word = gr.Textbox(label="Trigger word", placeholder="e.g. mycharname")
                preset_dd = gr.Dropdown(
                    label="VRAM preset",
                    choices=list(PRESETS.keys()),
                    value=initial_preset,
                )
                expert_dd = gr.Dropdown(
                    label="Expert(s) to train",
                    choices=["low", "high", "both"],
                    value="low",
                    info="'both' needs 40GB+ (offloads inactive DiT)",
                )
                # Wan2.2 only supports 8 specific resolution buckets — see
                # basi/dataset.py SUPPORTED_RESOLUTIONS. Free sliders would let
                # users pick unsupported sizes and musubi silently snaps,
                # wasting training time. Dropdown of valid pairs enforces.
                resolution_choice = gr.Dropdown(
                    [
                        ("480×832 (portrait 480p)", "480x832"),
                        ("832×480 (landscape 480p)", "832x480"),
                        ("704×1024 (portrait 720p)", "704x1024"),
                        ("1024×704 (landscape 720p)", "1024x704"),
                        ("704×1280 (portrait 720p wide)", "704x1280"),
                        ("1280×704 (landscape 720p wide)", "1280x704"),
                        ("720×1280 (portrait 720p std)", "720x1280"),
                        ("1280×720 (landscape 720p std)", "1280x720"),
                    ],
                    value="832x480",
                    label="Resolution (Wan2.2 supported buckets)",
                )
                # Default to 33f (safe everywhere except 8/12g — see preset
                # max_target_frames). The actual cap per preset is enforced
                # via the may-OOM warning below; we don't lock the slider so
                # users with headroom can opt up.
                target_frames = gr.Slider(17, 81, value=33, step=4,
                                          label="Frames (4n+1)")
                frame_warning = gr.Markdown(visible=False)
                # T4.G: per-clip repeats — useful for small datasets.
                num_repeats = gr.Slider(1, 20, value=1, step=1,
                                        label="Num repeats per clip per epoch")
                # T4.A: optional resume from a saved state dir.
                resume_state = gr.Textbox(
                    label="Resume from state (optional)",
                    placeholder="path to musubi-tuner --save_state directory",
                    info="Leave empty for fresh training. Set --save_state on previous run to save state.",
                )
                # T4.H: advanced optimizer args (hidden by default).
                with gr.Accordion("Advanced optimizer", open=False):
                    adv_lr_scheduler = gr.Dropdown(
                        ["", "constant_with_warmup", "cosine", "linear",
                         "polynomial", "adafactor"],
                        value="", label="LR scheduler (empty = musubi default)",
                    )
                    adv_lr_warmup = gr.Number(value=None, label="Warmup steps",
                                              precision=0)
                    adv_grad_clip = gr.Number(value=None, label="Max grad norm (0 = disable)")
                    adv_weight_decay = gr.Number(value=None, label="Weight decay")
                max_epochs = gr.Slider(1, 100, value=20, step=1, label="Max train epochs")
                sample_every = gr.Slider(0, 1000, value=200, step=50,
                                         label="Sample every N steps (0=off)")
                sample_prompt = gr.Textbox(
                    label="Sample prompt",
                    value="a cat walking through a sunlit garden, cinematic",
                    lines=2,
                )

            # COLUMN 2: Dataset
            with gr.Column():
                gr.Markdown("## 2 · Dataset")
                gr.Markdown("Drop `.mp4` / `.mov` clips. Captions: `.txt` next to each clip "
                            "(use **Auto-caption** below to generate).")
                upload = gr.File(file_count="multiple", label="Videos (+ optional .txt captions)")
                with gr.Row():
                    ingest_btn = gr.Button("Ingest + Scan", variant="primary")
                    caption_btn = gr.Button("Auto-caption (Qwen2.5-VL)", variant="secondary")
                dataset_report = gr.Markdown("(no dataset yet)")
                dataset_dir_box = gr.Textbox(label="Dataset dir", interactive=False)
                caption_report = gr.Markdown()
                # T4.F: trigger-word coverage lint output (Markdown banner)
                trigger_lint = gr.Markdown()
                # T4.B: in-UI caption editor — gr.Dataframe with editable text
                # column; row = (clip filename, caption). User edits + clicks
                # Save to persist. Refresh re-reads from .txt sidecars.
                caption_editor = gr.Dataframe(
                    headers=["clip", "caption"], datatype=["str", "str"],
                    label="Caption editor (edit and Save to update .txt sidecars)",
                    interactive=True, wrap=True, row_count=(0, "dynamic"),
                )
                with gr.Row():
                    refresh_captions_btn = gr.Button("Reload from disk")
                    save_captions_btn = gr.Button("Save captions", variant="primary")
                caption_save_status = gr.Markdown()

            # COLUMN 3: Train
            with gr.Column():
                gr.Markdown("## 3 · Train")
                with gr.Row():
                    gen_btn = gr.Button("Generate scripts", variant="primary")
                    run_cache_btn = gr.Button("Run cache (T5+VAE)", variant="secondary")
                    run_train_btn = gr.Button("Run training", variant="secondary")
                with gr.Row():
                    stop_cache_btn = gr.Button("Stop cache", variant="stop")
                    stop_train_btn = gr.Button("Stop train", variant="stop")
                scripts_md = gr.Markdown(label="Scripts")
                cache_script_box = gr.Textbox(label="cache.sh path", visible=False)
                train_script_box = gr.Textbox(label="train.sh path", visible=False)
                logs = gr.Textbox(label="Live logs", lines=20, max_lines=20, autoscroll=True)
                stop_status = gr.Markdown()
                with gr.Row():
                    link_comfy_btn = gr.Button("Link latest LoRA → ComfyUI")
                    open_folder_btn = gr.Button("Open output folder")
                    link_status = gr.Markdown()

                gr.Markdown("### Sample gallery")
                gr.Markdown(
                    "Auto-refreshes every 10s while training runs. "
                    "Newest 8 samples shown; click to download."
                )
                with gr.Row():
                    refresh_samples_btn = gr.Button("Refresh samples")
                # gr.Files accepts mp4 and supports multi-file display +
                # download. gr.Gallery does not render video inline.
                sample_gallery = gr.Files(label="Samples (.mp4)", file_count="multiple")
                sample_timer = gr.Timer(10.0)

                gr.Markdown("### Preview (Faster-Wan2.2 Lightning + FP8 + TAEHV)")
                gr.Markdown("~3-4 s at 832×480×17 frames. Trainer must be paused — single "
                            "4090 cannot run trainer + preview concurrently.")
                with gr.Row():
                    preview_size = gr.Dropdown(["832*480", "480*832", "720*1280", "1280*720"],
                                               value="832*480", label="Size")
                    preview_frames = gr.Slider(5, 81, value=17, step=4, label="Frames (4n+1)")
                    preview_steps = gr.Slider(2, 8, value=4, step=1, label="Lightning steps")
                    preview_seed = gr.Number(value=42, label="Seed", precision=0)
                # Optional LoRA file override — by default the preview uses
                # the newest .safetensors in the workspace. Picker lets users
                # preview an earlier checkpoint or a LoRA from another workspace.
                preview_lora_override = gr.File(
                    label="Preview LoRA (optional — default: newest in workspace)",
                    file_types=[".safetensors"],
                )
                # T4.C: extra LoRAs to stack on top of the trained user LoRA.
                # Comma-separated paths via BASIWAN_USER_LORA_LIST env var
                # (Faster-Wan2.2 falls back to BASIWAN_USER_LORA single-path
                # when LIST is unset). Strength applied uniformly here; a
                # future refactor can expose per-LoRA strength sliders.
                preview_extra_loras = gr.File(
                    label="Extra LoRAs to stack (optional)",
                    file_types=[".safetensors"],
                    file_count="multiple",
                )
                # T4.E: live loss chart. Parses 'loss=X.YYY' from training
                # stdout lines and plots as a line chart. Updates every 5s
                # via the same Timer tick wiring as the sample gallery.
                loss_chart = gr.LinePlot(
                    x="step", y="loss", title="Training loss",
                    height=200,
                )
                # T4.D: export converter button — uses basi/export.py.
                with gr.Row():
                    export_format = gr.Dropdown(
                        ["musubi", "diffusers", "peft", "comfyui"],
                        value="comfyui", label="Export format",
                    )
                    export_btn = gr.Button("Export latest LoRA", variant="secondary")
                export_status = gr.Markdown()
                preview_btn = gr.Button("Generate preview from latest LoRA", variant="secondary")
                preview_video = gr.Video(label="Preview")
                preview_status = gr.Markdown()

        # Wiring
        ingest_btn.click(_ingest_uploaded_videos,
                         inputs=[upload, lora_name],
                         outputs=[dataset_report, dataset_dir_box])
        caption_btn.click(_auto_caption,
                          inputs=[dataset_dir_box, trigger_word],
                          outputs=[caption_report])
        def _check_frame_cap(preset_key: str, frames: int):
            cap = PRESETS[preset_key].max_target_frames
            if frames > cap:
                return gr.Markdown(
                    value=(f"⚠️ **{frames} frames may OOM on {preset_key} preset** "
                           f"(safe cap: {cap}f). Larger cap presets have more "
                           f"VRAM headroom. If your card has slack you can keep "
                           f"this; if training OOMs, drop to {cap}f."),
                    visible=True,
                )
            return gr.Markdown(visible=False)

        preset_dd.change(_check_frame_cap, inputs=[preset_dd, target_frames],
                         outputs=[frame_warning])
        target_frames.change(_check_frame_cap, inputs=[preset_dd, target_frames],
                             outputs=[frame_warning])
        gen_btn.click(_generate_scripts,
                      inputs=[lora_name, dataset_dir_box, preset_dd, expert_dd, trigger_word,
                              sample_prompt, max_epochs, sample_every, target_frames,
                              resolution_choice, num_repeats, resume_state,
                              adv_lr_scheduler, adv_lr_warmup, adv_grad_clip,
                              adv_weight_decay],
                      outputs=[scripts_md, cache_script_box, train_script_box])
        run_cache_btn.click(_spawn_script, inputs=[cache_script_box], outputs=[logs])
        run_train_btn.click(_spawn_script, inputs=[train_script_box], outputs=[logs])
        stop_train_btn.click(_stop_script, inputs=[train_script_box], outputs=[stop_status])
        link_comfy_btn.click(_link_to_comfyui, inputs=[lora_name], outputs=[link_status])
        # Stop both cache + training scripts independently (cache precompute
        # can itself take minutes; a single Stop only covered training).
        stop_cache_btn.click(_stop_script, inputs=[cache_script_box], outputs=[stop_status])
        # lora_name validation — must be a safe path component
        import re as _re
        _LORA_NAME_RE = _re.compile(r"^[A-Za-z0-9_-]+$")
        def _validate_lora_name(name: str):
            if name and not _LORA_NAME_RE.match(name):
                return gr.Markdown(
                    value=(f"⚠️ `{name}` contains characters that break path "
                           "operations downstream. Use letters, digits, "
                           "underscore, hyphen only."),
                    visible=True,
                )
            return gr.Markdown(visible=False)
        lora_name.change(_validate_lora_name, inputs=[lora_name],
                         outputs=[lora_name_warning])
        # Open output folder — cross-platform. Prints path; on local installs
        # also tries to open via the OS file browser.
        def _open_output(name: str):
            if not name:
                return "error: workspace name required"
            ws = WORKSPACES / name
            ws.mkdir(parents=True, exist_ok=True)
            uri = ws.as_uri()
            try:
                import webbrowser
                webbrowser.open(uri)
                return f"opened: {ws}"
            except Exception as e:
                return f"path: {ws} (open failed: {e})"
        open_folder_btn.click(_open_output, inputs=[lora_name],
                              outputs=[link_status])
        refresh_samples_btn.click(_latest_samples, inputs=[lora_name], outputs=[sample_gallery])
        sample_timer.tick(_latest_samples, inputs=[lora_name], outputs=[sample_gallery])
        preview_btn.click(_generate_preview,
                          inputs=[lora_name, sample_prompt, preview_size, preview_frames,
                                  preview_steps, preview_seed, expert_dd,
                                  preview_lora_override, preview_extra_loras],
                          outputs=[preview_video, preview_status])
        # T4.B: caption editor wiring
        def _load_caption_rows(dataset_dir: str):
            if not dataset_dir or not Path(dataset_dir).exists():
                return []
            rows = []
            ds = Path(dataset_dir)
            for clip in sorted(ds.iterdir()):
                if clip.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
                    txt = clip.with_suffix(".txt")
                    caption = txt.read_text(encoding="utf-8").strip() if txt.exists() else ""
                    rows.append([clip.name, caption])
            return rows

        def _save_caption_rows(dataset_dir: str, rows):
            if not dataset_dir or not Path(dataset_dir).exists():
                return "error: no dataset dir"
            if rows is None or len(rows) == 0:
                return "error: no rows to save"
            # rows can be a list of lists or a pandas DataFrame (Gradio quirk)
            try:
                iter_rows = rows.values.tolist() if hasattr(rows, "values") else rows
            except Exception:
                iter_rows = list(rows)
            ds = Path(dataset_dir)
            n_written = 0
            for r in iter_rows:
                if not r or len(r) < 2:
                    continue
                clip_name, caption = str(r[0]), str(r[1]) if r[1] is not None else ""
                clip_path = ds / clip_name
                if not clip_path.exists():
                    continue
                clip_path.with_suffix(".txt").write_text(caption + "\n",
                                                          encoding="utf-8")
                n_written += 1
            return f"wrote {n_written} caption .txt sidecars to {ds}"

        refresh_captions_btn.click(_load_caption_rows,
                                   inputs=[dataset_dir_box],
                                   outputs=[caption_editor])
        save_captions_btn.click(_save_caption_rows,
                                inputs=[dataset_dir_box, caption_editor],
                                outputs=[caption_save_status])

        # T4.F: trigger-word coverage check — uses the fast path that reads
        # .txt sidecars directly (no ffprobe per clip). Triggered on every
        # trigger_word.change() event, so it MUST stay O(file-count), not
        # O(file-content) — never re-probe video metadata here.
        def _check_trigger_coverage(dataset_dir: str, trigger: str):
            if not dataset_dir or not Path(dataset_dir).exists() or not trigger:
                return ""
            from basi.dataset import trigger_word_coverage_fast
            n_cov, n_total = trigger_word_coverage_fast(dataset_dir, trigger)
            if n_total == 0:
                return ""
            pct = 100 * n_cov / n_total
            if pct < 80:
                return (f"⚠️ Trigger word **'{trigger}'** appears in only "
                        f"{n_cov}/{n_total} captions ({pct:.0f}%). Edit "
                        f"captions to include the trigger consistently.")
            return f"✓ Trigger coverage: {n_cov}/{n_total} ({pct:.0f}%)"
        trigger_word.change(_check_trigger_coverage,
                            inputs=[dataset_dir_box, trigger_word],
                            outputs=[trigger_lint])
        caption_btn.click(_check_trigger_coverage,
                          inputs=[dataset_dir_box, trigger_word],
                          outputs=[trigger_lint])

        # T4.D: export converter wiring
        def _export_lora(workspace_name: str, target_format: str):
            if not workspace_name:
                return "error: workspace name required"
            ws = WORKSPACES / workspace_name
            lora = latest_lora_in(ws)
            if lora is None:
                return f"no .safetensors LoRA found in {ws}"
            from basi.export import convert
            dst = ws / f"{lora.stem}_{target_format}.safetensors"
            try:
                summary = convert(lora, target_format, dst)
                return (f"exported to **{dst}** ({summary['n_keys']} keys)\n\n"
                        f"_{summary['note']}_")
            except Exception as e:
                return f"export failed: {e}"
        export_btn.click(_export_lora,
                         inputs=[lora_name, export_format],
                         outputs=[export_status])

        # T4.E: live loss chart via stdout parsing
        import re as _re_loss
        _LOSS_RE = _re_loss.compile(r"loss\s*[=:]\s*(\d+\.\d+)")
        _loss_history = []

        def _update_loss_chart(log_text: str):
            if not log_text:
                return None
            new_entries = []
            for m in _LOSS_RE.finditer(log_text):
                new_entries.append(float(m.group(1)))
            if not new_entries:
                return None
            start = len(_loss_history)
            _loss_history.extend(new_entries[start:])
            if not _loss_history:
                return None
            return {"step": list(range(len(_loss_history))),
                    "loss": _loss_history}
        # The chart updates on the same 10s timer as the sample gallery —
        # cheap, no extra wiring. logs is the source of truth.
        sample_timer.tick(_update_loss_chart, inputs=[logs],
                          outputs=[loss_chart])
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
