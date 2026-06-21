"""BASI WAN KENOBI — Flux-Gym-style UI for Wan2.2 video LoRA training.

Three-column layout (Configure → Dataset → Train):
  - Configure: LoRA name, trigger word, expert (low/high/both), preset, steps, sample prompt
  - Dataset: drop videos, auto-caption (tier-selected Qwen-VL), bucket histogram
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
import base64
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basi.presets import (
    PRESETS, detect_vram_gb, auto_select,
    detect_capability, recommend_inference_scheme,
    inference_tier, detect_host_ram_gb,  # adaptive inference VRAM tiers
    s2v_caps,  # S2V is heavier than T2V -> its own measured capability gate
    MOVA_PRESETS, auto_select_mova,  # MOVA A/V Gym mode preset swap
)
from basi.dataset import scan_dataset, bucket_distribution
from basi.train import TrainConfig, prepare_training_run, OUTPUTS_ROOT
from basi.defaults import RESTYLE_SDEDIT_RECIPE, STUDIO, GYM  # single source of truth for UI defaults
from basi.mova_train import (format_mova_estimate_md,  # pre-flight time/VRAM estimate
                             parse_mova_training_log, format_mova_analytics_md)  # live analytics
from basi.preview import PreviewSpec, generate_preview, latest_lora_in
from basi.caption import caption_dataset, select_model_for_vram
from basi import eta as _eta  # pre-flight estimate + live countdown + EMA cache

BASI_ROOT = Path(__file__).resolve().parent
WORKSPACES = BASI_ROOT / "outputs"


def _list_studio_loras():
    """Discover trained LoRAs in Gym workspaces for the Studio dropdown.
    Returns [(label, payload)] where payload is None for "(none)" or a dict
    {high, low, single} of resolved file paths. Auto-pairs <base>_high /
    <base>_low dual-expert workspaces into one entry. Uses ONLY the final
    <name>.safetensors (not epoch -NNNNNN checkpoints)."""
    import re as _re
    out = [("(none — Lightning only)", None)]
    if not WORKSPACES.exists():
        return out
    finals = {}  # workspace stem -> final safetensors path
    for ws in sorted(WORKSPACES.iterdir()):
        if not ws.is_dir():
            continue
        f = ws / f"{ws.name}.safetensors"
        if f.exists():
            finals[ws.name] = f
    paired = set()
    for name in sorted(finals):
        if name in paired:
            continue
        m = _re.match(r"(.+)_high$", name)
        if m and f"{m.group(1)}_low" in finals:
            base = m.group(1)
            paired.update({f"{base}_high", f"{base}_low"})
            out.append((f"{base} (high+low pair)",
                        {"high": str(finals[f"{base}_high"]),
                         "low": str(finals[f"{base}_low"]), "single": None}))
        elif name.endswith("_low") and f"{name[:-4]}_high" in finals:
            continue  # handled when the _high sibling is seen
        else:
            out.append((f"{name}", {"high": None, "low": None,
                                    "single": str(finals[name])}))
    return out


def _plan_user_lora(*, mtype, vace_restyle, lora, sel, lora_strength, light_s,
                    lightning_lora, useronly_root, combo_root):
    """Pure routing decision for the Studio user-LoRA load — NO I/O, so the
    branch logic is unit-testable instead of buried in the generate closure.
    The caller performs the build + set_lora described by the returned plan.

    Modes:
      'none'          : no user LoRA → clear (set_lora dir None).
      'vace_useronly' : VACE depth-lock — user-only dir (no Lightning; it would
                        distort the 50-step full-CFG path), runtime-scalable so
                        the request slider scales it with NO rebuild. Cache key
                        is strength-INDEPENDENT.
      't2v_combo'     : T2V — Lightning+user rank-concat, strength BAKED at build,
                        request strength pinned to 1.0.
    Returns: {mode, want_key, set_lora:{dir, runtime_scalable?}, build|None,
              request_strength}. want_key matches the worker's _lora_key cache
      (so an unchanged selection skips rebuild+re-attach)."""
    import re as _re
    from basi import combo as _combo
    rs = float(lora_strength)
    # User LoRA only on T2V and VACE. I2V Continue uses the Seko-I2V Lightning
    # LoRA; a T2V-trained user LoRA there is unvalidated → treat as none.
    if sel is None or mtype not in ("t2v", "vace"):
        return {"mode": "none", "want_key": None,
                "set_lora": {"dir": None}, "build": None, "request_strength": rs}
    _hi = sel.get("single") or sel.get("high")
    _lo = sel.get("single") or sel.get("low")
    if vace_restyle:
        udir = Path(useronly_root) / _re.sub(r"[^\w.-]", "_", str(lora))
        return {"mode": "vace_useronly", "want_key": f"vace-useronly:{lora}",
                "set_lora": {"dir": udir, "runtime_scalable": True},
                "build": {"kind": "user_only", "out_dir": udir,
                          "user_high": _hi, "user_low": _lo, "user_strength": 1.0},
                "request_strength": rs}
    _ufiles = ([sel["single"]] if sel.get("single") else [sel["high"], sel["low"]])
    cdir = _combo.combo_cache_path(str(combo_root), str(lightning_lora),
                                   _ufiles, rs, lightning_strength=light_s)
    return {"mode": "t2v_combo", "want_key": cdir.name,
            "set_lora": {"dir": cdir},
            "build": {"kind": "combo", "out_dir": cdir,
                      "lightning_dir": str(lightning_lora),
                      "user_high": _hi, "user_low": _lo,
                      "user_strength": rs, "lightning_strength": light_s},
            "request_strength": 1.0}   # combo bakes strength → pin


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
    # Override order: BASIWAN_COMFY_LORA_DIR, then user home, then the Pinokio
    # peer-app layout (../comfy.git is the cocktailpeanut convention).
    *([Path(os.environ["BASIWAN_COMFY_LORA_DIR"])]
      if os.environ.get("BASIWAN_COMFY_LORA_DIR") else []),
    Path.home() / "ComfyUI/models/loras",
    Path(os.environ.get("USERPROFILE", str(Path.home()))) / "ComfyUI/models/loras",
    Path(__file__).resolve().parent.parent / "comfy.git/models/loras",
]


# --- Inference checkpoint resolution (module-level + pure for testability).
# Everything resolves under BASIWAN_CKPT_DIR (default <repo>/checkpoints), so
# one env var repoints the whole model set. ---
def _basiwan_resolve_ckpt(env_val, bases, rel):
    """Resolve a non-GGUF checkpoint dir/file. env override (if it exists) ->
    <base>/rel for each base in priority order -> the FIRST base's path as the
    not-found hint (names THIS machine, not a dev box). `bases` is an ordered
    list of Path roots; env_val is os.environ.get(env_key)."""
    if env_val and Path(env_val).exists():
        return env_val
    for b in bases:
        cand = Path(b) / rel
        if cand.exists():
            return str(cand)
    return str(Path(bases[0]) / rel)


def _basiwan_resolve_gguf(env_val, bases, repo, which, fname):
    """Resolve one GGUF expert file, LAYOUT-AGNOSTIC so a plain `huggingface-cli
    download <org>/<repo>` just works. For each base/gguf it globs, in order:
      1. models--<repo>/snapshots/*/<which>/*Q4_K_M.gguf   (HF cache layout)
      2. <repo-short>/<which>/*Q4_K_M.gguf                 (renamed flat dir)
      3. <which>/<fname>                                    (bare flat)
    repo is the HF id with '/'->'--' (e.g. QuantStack--Wan2.2-T2V-A14B-GGUF).
    Returns the first hit, else a repo-local not-found hint."""
    import glob as _g
    if env_val and Path(env_val).exists():
        return env_val
    short = repo.split("--")[-1]
    for b in bases:
        gd = Path(b) / "gguf"
        for pat in (str(gd / f"models--{repo}" / "snapshots" / "*" / which / "*Q4_K_M.gguf"),
                    str(gd / short / which / "*Q4_K_M.gguf"),
                    str(gd / which / fname)):
            hits = sorted(_g.glob(pat))
            if hits:
                return hits[0]
    return str(Path(bases[0]) / "gguf" / which / fname)


_INFER_TIER_CACHE: dict = {}


def _get_infer_tier() -> dict:
    """The inference recipe for THIS machine (VRAM + host RAM), computed
    once and cached. Falls back to the measured 24GB champion tier if probing
    fails."""
    if "t" not in _INFER_TIER_CACHE:
        try:
            _INFER_TIER_CACHE["t"] = inference_tier(
                detect_capability().vram_gb, detect_host_ram_gb())
        except Exception:
            _INFER_TIER_CACHE["t"] = inference_tier(24, 0)
    return _INFER_TIER_CACHE["t"]


_S2V_CAPS_CACHE: dict = {}


def _get_s2v_caps() -> dict:
    """This machine's S2V capability (availability + max resolution/window),
    cached. Falls back to 'available at 480p' if probing fails (a 24GB-class default;
    the runtime residency cap still self-limits, so a probe failure won't brick)."""
    if "c" not in _S2V_CAPS_CACHE:
        try:
            _S2V_CAPS_CACHE["c"] = s2v_caps(detect_capability().vram_gb)
        except Exception:
            _S2V_CAPS_CACHE["c"] = s2v_caps(24)
    return _S2V_CAPS_CACHE["c"]


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
        # Long-source guard: a >60s "clip" is almost certainly raw footage
        # that belongs in Auto-split, not the dataset — training samples 2s
        # windows from it and the frame snap throws the rest away.
        if c.duration_s > 60:
            issues += (" — ⚠️ **this is a long source video, not a "
                       "training clip. Use 'Auto-split long video into "
                       "clips' below instead; it will replace this file "
                       "with 2-5s shot clips.**")
        rep.append(f"- `{Path(c.path).name}` {c.width}×{c.height} @{c.fps:.1f}fps × {c.frame_count}f [{cap}]{issues}")
    buckets = bucket_distribution(clips)
    rep.append(f"\n## Bucket distribution ({len(buckets)} buckets)\n")
    for (res, frames), count in sorted(buckets.items(), key=lambda x: -x[1]):
        rep.append(f"- {res[0]}×{res[1]} × {frames}f: **{count} clips**")
    return "\n".join(rep), str(ds_dir)


def _auto_caption(dataset_dir: str, trigger_word: str, mova_av_mode: bool = False):
    """Caption the dataset with live STREAMING status text.

    mova_av_mode=True (set when the Gym training type is MOVA): caption with the joint A/V prompt
    (one diegetic sound-event clause, no art-style words) AND then append VERBATIM spoken dialogue
    via CPU-Whisper -- required for intelligible MOVA speech (measured CER 0.0-0.06 with dialogue
    vs non-English babble without). Without this routing the MOVA dataset gets the video-only prompt
    and no dialogue -> garbled audio.

    A generator: every yield replaces the visible report markdown, so the user
    has feedback from the first second of the click. Long operations stream text
    into a visible component rather than into a gr.Progress overlay (which renders
    on an initially-empty zero-height Markdown and would be invisible).
    """
    import threading as _th
    import time as _tm

    if not dataset_dir or not Path(dataset_dir).exists():
        yield ("**No dataset yet.** Upload clips (or Auto-split a long "
               "video) in column 2 first.")
        return
    videos = [p for p in sorted(Path(dataset_dir).iterdir())
              if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}]
    if not videos:
        yield "**No video clips found** in the dataset folder."
        return
    done_already = sum(1 for p in videos if p.with_suffix(".txt").exists())
    model_id = select_model_for_vram()
    yield (f"🔎 {len(videos)} clips found, {done_already} already captioned "
           f"— captioner: `{model_id}`")

    # ── Stage 1: ensure model on disk, with a live GB counter.
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(model_id, local_files_only=True)
    except Exception:
        from huggingface_hub import HfApi, constants as _hfc
        _dl: dict = {}

        def _download():
            try:
                snapshot_download(model_id)
            except Exception as e:
                _dl["err"] = e

        _t = _th.Thread(target=_download, daemon=True)
        _t.start()
        try:
            _info = HfApi().model_info(model_id, files_metadata=True)
            _total = sum((s.size or 0) for s in _info.siblings) or None
        except Exception:
            _total = None
        _folder = (Path(_hfc.HF_HUB_CACHE)
                   / f"models--{model_id.replace('/', '--')}")
        _t0 = _tm.time()
        while _t.is_alive():
            _sz = (sum(f.stat().st_size for f in _folder.rglob("*")
                       if f.is_file()) if _folder.exists() else 0)
            _rate = _sz / max(_tm.time() - _t0, 1)
            _eta = ((_total - _sz) / _rate / 60
                    if (_total and _rate > 1e6) else None)
            yield (f"⬇️ downloading `{model_id}` (one time): "
                   f"**{_sz / 1e9:.1f}{f' / {_total / 1e9:.1f}' if _total else ''} GB**"
                   + (f" — ~{_eta:.0f} min left" if _eta else ""))
            _t.join(timeout=2)
        if "err" in _dl:
            yield f"❌ **model download failed**: {_dl['err']}"
            return

    # ── Stage 2+3: load model + caption, worker thread + 1s status poll.
    state = {"i": 0, "n": 0, "name": "", "done": False,
             "result": None, "err": None}

    def _cb(i, n, name):
        state.update(i=i, n=n, name=name)

    def _work():
        try:
            state["result"] = caption_dataset(
                [str(p) for p in videos],
                trigger_word=(trigger_word.strip() or None),
                skip_existing=True,
                progress_cb=_cb,
                mova_av_mode=mova_av_mode,
            )
        except Exception as e:
            state["err"] = e
        state["done"] = True

    _w = _th.Thread(target=_work, daemon=True)
    _w.start()
    _t0 = _tm.time()
    _cap_t0 = None
    _prev_name = None
    _recent: list[str] = []  # last few finished captions, shown live
    while not state["done"]:
        el = _tm.time() - _t0
        if state["n"] == 0:
            yield (f"📦 loading `{model_id}` into VRAM… ({el:.0f}s — "
                   f"expect 1-2 min for the 17 GB 8B tier)")
        else:
            if _cap_t0 is None:
                _cap_t0 = _tm.time()
            i, n = state["i"], state["n"]
            name = state["name"]
            # A name change means the previous clip just finished — show
            # its freshly-written caption so quality is judgeable live.
            if _prev_name and name != _prev_name:
                _txt = Path(dataset_dir) / Path(_prev_name).with_suffix(".txt").name
                if _txt.exists():
                    _cap = _txt.read_text(encoding="utf-8").strip()
                    _recent.append(f"- ✅ `{_prev_name}` → {_cap[:140]}"
                                   f"{'…' if len(_cap) > 140 else ''}")
                    _recent = _recent[-4:]
            _prev_name = name
            if i > 0:
                rate = (_tm.time() - _cap_t0) / i
                tail = f" — ~{rate:.0f}s/clip, ~{rate * (n - i) / 60:.0f} min left"
            else:
                tail = ""
            yield (f"🎬 captioning **{i + 1}/{n}**: `{name}`{tail}\n\n"
                   + "\n".join(_recent))
        _w.join(timeout=1)
    if state["err"] is not None:
        yield f"❌ **captioning failed**: {state['err']}"
        return
    results = state["result"] or {}
    lines = [f"# ✅ Auto-caption done ({len(results)} clips, model: `{model_id}`)\n"]
    for path, cap in results.items():
        lines.append(f"- **{Path(path).name}**: {cap[:160]}{'…' if len(cap) > 160 else ''}")
    if not mova_av_mode:
        yield "\n".join(lines)
        return

    # ── MOVA stage 4: append VERBATIM spoken dialogue (required for audio intelligibility).
    # MOVA conditions speech on the literal words; without them generated audio is non-English
    # babble (measured CER 0.0-0.06 with dialogue). CPU-Whisper, idempotent (.txt.orig backup).
    lines.append("\n🎙️ adding spoken dialogue to captions (ASR · CPU Whisper) — required for "
                 "intelligible MOVA speech…")
    yield "\n".join(lines)
    from basi.caption import asr_dialogue_recaption
    astate = {"done": False, "res": None, "err": None, "i": 0, "n": 0, "name": ""}

    def _acb(i, n, name):
        astate.update(i=i, n=n, name=name)

    def _awork():
        try:
            astate["res"] = asr_dialogue_recaption(dataset_dir, progress_cb=_acb)
        except Exception as e:
            astate["err"] = e
        astate["done"] = True

    _aw = _th.Thread(target=_awork, daemon=True)
    _aw.start()
    while not astate["done"]:
        if astate["n"]:
            yield ("\n".join(lines)
                   + f"\n\n🎙️ ASR **{astate['i'] + 1}/{astate['n']}**: `{astate['name']}`")
        _aw.join(timeout=1)
    if astate["err"] is not None:
        yield ("\n".join(lines) + f"\n\n⚠️ **dialogue step skipped**: {astate['err']}\n\n"
               "Captions are written but WITHOUT spoken dialogue — MOVA speech will be garbled. "
               "Run `python tools/mova_recaption_asr.py <dataset> --apply` in an env with "
               "faster-whisper before training.")
        return
    r = astate["res"] or {}
    lines.append(f"\n✅ **dialogue added to {r.get('speech', 0)}/{r.get('total', 0)} clips** "
                 "(music/SFX/silence clips kept as-is). Captions are MOVA-ready.")
    yield "\n".join(lines)


def _probe_dataset_fps(dataset_dir: str) -> float | None:
    """Median fps of up to 5 dataset clips; None when ~16 (no resample
    needed) or unprobeable."""
    try:
        from basi.dataset import probe_video
        clips = sorted(Path(dataset_dir).glob("*.mp4"))[:5]
        if not clips:
            return None
        rates = sorted(probe_video(p).fps for p in clips)
        med = rates[len(rates) // 2]
        return round(med, 2) if abs(med - 16.0) > 1.0 else None
    except Exception:
        return None


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
        # Auto-derive source fps from the first clip so musubi resamples
        # to its 16fps training rate (a 24fps TV rip's 33-frame window
        # should span 2 real seconds, not 1.4). None when already ~16.
        source_fps=_probe_dataset_fps(dataset_dir),
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

# Cross-platform process-group management for Gym scripts, for reliable
# kill of the whole training tree:
#   - Unix: setsid + killpg (preexec_fn is unavailable on Windows)
#   - Windows: CREATE_NEW_PROCESS_GROUP creationflags + psutil tree walk to
#     kill the script process AND any musubi/accelerate child workers.
_IS_WIN = sys.platform == "win32"


def _spawn_kwargs():
    """subprocess.Popen kwargs that make the new child the head of a process
    group so we can signal the whole tree on stop."""
    if _IS_WIN:
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        return {"preexec_fn": os.setsid}


def _kill_process_tree(proc: subprocess.Popen, timeout: float = 3.0) -> str:
    """Cross-platform polite-then-force kill of proc + all descendants.

    On Windows uses psutil to enumerate and terminate; on Unix uses killpg
    for whole-group SIGTERM/SIGKILL. Returns a human-readable status string.
    """
    if proc.poll() is not None:
        return "process already gone"
    try:
        import psutil
        parent = psutil.Process(proc.pid)
        try:
            children = parent.children(recursive=True)
        except psutil.NoSuchProcess:
            children = []
        # Polite shutdown
        if _IS_WIN:
            for c in children:
                try: c.terminate()
                except psutil.NoSuchProcess: pass
            try: parent.terminate()
            except psutil.NoSuchProcess: pass
        else:
            import signal as _sig
            try:
                os.killpg(os.getpgid(proc.pid), _sig.SIGTERM)
            except (ProcessLookupError, PermissionError):
                # Fall back to per-process terminate via psutil
                for c in children:
                    try: c.terminate()
                    except psutil.NoSuchProcess: pass
                try: parent.terminate()
                except psutil.NoSuchProcess: pass
        # Wait for graceful exit
        all_procs = children + [parent]
        gone, alive = psutil.wait_procs(all_procs, timeout=timeout)
        if not alive:
            return f"stopped: pid {proc.pid}"
        # Force kill stragglers
        if _IS_WIN:
            for p in alive:
                try: p.kill()
                except psutil.NoSuchProcess: pass
        else:
            import signal as _sig
            try:
                os.killpg(os.getpgid(proc.pid), _sig.SIGKILL)
            except (ProcessLookupError, PermissionError):
                for p in alive:
                    try: p.kill()
                    except psutil.NoSuchProcess: pass
        psutil.wait_procs(alive, timeout=2)
        return f"force-killed: pid {proc.pid}"
    except Exception as e:
        # Last-ditch native fallback: subprocess.terminate then kill
        try: proc.terminate()
        except Exception: pass
        try:
            proc.wait(timeout=timeout)
            return f"stopped (fallback): pid {proc.pid}"
        except subprocess.TimeoutExpired:
            try: proc.kill()
            except Exception: pass
            return f"force-killed (fallback after {type(e).__name__}): pid {proc.pid}"


def _stream_proc(cmd, reg_key=None, banner=None, env=None):
    """Shared streaming core for one-off / streamed subprocess jobs: spawn `cmd`, yield
    a growing 200-line tail of its combined output, then the exit code. If reg_key is
    given, register the Popen in _active_procs[reg_key] so a Stop button can signal it.

    This is the ONE streaming idiom for ephemeral jobs (training/cache scripts via
    _spawn_script, one-off commands via _stream_cmd). BasiwanRunner is the SEPARATE,
    intentionally-distinct lifecycle (a persistent, model-swappable inference worker) --
    not folded in here because its lifecycle (singleton, long-lived, thread-locked) is
    nothing like a streamed one-shot."""
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True, env=env, **_spawn_kwargs())
    except FileNotFoundError as e:
        yield f"error: failed to spawn: {e}"
        return
    if reg_key is not None:
        _active_procs[reg_key] = proc
    buf = []
    if banner:
        yield banner
    for line in proc.stdout:
        buf.append(line.rstrip())
        yield "\n".join(buf[-200:])   # growing tail: progress without unbounded growth
    proc.wait()
    if reg_key is not None:
        _active_procs.pop(reg_key, None)
    buf.append(f"\n[exit code {proc.returncode}]")
    yield "\n".join(buf[-200:])


def _spawn_script(script_path: str):
    """Stream a generated training/cache script's output. Registers under
    script path so a separate Stop button can signal the same Popen.

    Cross-platform invocation: on Unix runs `bash script_path`; on Windows
    runs the script directly (must be .bat/.cmd, see basi/train.py).
    """
    if not script_path or not Path(script_path).exists():
        yield f"error: script not found: {script_path}"
        return
    # Kill any prior run of the same script (user re-clicked Run)
    prior = _active_procs.pop(script_path, None)
    if prior is not None and prior.poll() is None:
        _kill_process_tree(prior, timeout=2.0)

    # Build platform-appropriate command. .bat/.cmd on Windows runs via
    # cmd.exe shell; .sh on any platform runs via bash (must be on PATH).
    _sp = Path(script_path)
    if _IS_WIN and _sp.suffix.lower() in {".bat", ".cmd"}:
        cmd = ["cmd.exe", "/c", str(_sp)]
        banner = f"$ cmd /c {script_path}\n"
    elif _sp.suffix.lower() == ".sh":
        import shutil as _shu
        _bash = _shu.which("bash")
        if not _bash:
            yield (f"error: bash not on PATH (required to run .sh on this "
                   f"platform). On Windows use a .bat script or install Git "
                   f"Bash / WSL.")
            return
        cmd = [_bash, str(_sp)]
        banner = f"$ bash {script_path}\n"
    else:
        # Unknown extension — best-effort direct exec
        cmd = [str(_sp)]
        banner = f"$ {script_path}\n"

    yield from _stream_proc(cmd, reg_key=script_path, banner=banner)


def _stream_cmd(cmd, env=None):
    """Stream a one-off command's combined output (no Stop registration). Thin wrapper
    over the shared _stream_proc; used to surface ensure_weights progress in the UI.
    `env` overrides the child environment (e.g. MOVA re-activates its env_mova venv)."""
    yield from _stream_proc(cmd, banner="$ " + " ".join(str(c) for c in cmd) + "\n", env=env)


def _stop_script(script_path: str) -> str:
    """Cross-platform polite-then-force kill of the running script tree."""
    proc = _active_procs.get(script_path)
    if proc is None or proc.poll() is not None:
        return f"no running process for {script_path}"
    return _kill_process_tree(proc)


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


# ============================================================================
# BASIWAN "liberated" theme — the dragon-ouroboros palette (teal/cyan <-> amber on
# near-black) in a Pliny/bt6-style cyberpunk skin (mono, neon glow, scanlines, ASCII
# banner). ALL rules are scoped under `body.basi-liberated`, which is added on load and
# toggled by the "liberated mode" checkbox. Unchecking removes the class -> the UI falls
# straight back to STOCK Gradio ("lame mode"), so there's no second palette to maintain.
# ============================================================================
_BASI_LIBERATED_CSS = """
body.basi-liberated, body.basi-liberated .gradio-container {
  background:#080c0d !important; color:#cfe8e3 !important;
  font-family:ui-monospace,"JetBrains Mono","Cascadia Code","Fira Code","Consolas",monospace !important; }
/* Drive Gradio's OWN theme variables dark so EVERY component (accordions, example panels,
   dropdowns, tables, upload zones, sliders) goes dark in one shot — fixes 'teal text on a
   white panel' wholesale instead of chasing class names. */
body.basi-liberated {
  --body-background-fill:#080c0d; --background-fill-primary:#0d1416; --background-fill-secondary:#0b1113;
  --block-background-fill:#0d1416; --block-border-color:#14403a; --block-label-background-fill:#0d1416;
  --block-label-text-color:#9fded2; --block-title-text-color:#2fe6cf; --block-info-text-color:#94bdb6;
  --panel-background-fill:#0d1416; --panel-border-color:#14403a;
  --input-background-fill:#060a0b; --input-border-color:#1c4b46; --input-placeholder-color:#5f817b;
  --border-color-primary:#14403a; --border-color-accent:#1fe0c4;
  --body-text-color:#cfe8e3; --body-text-color-subdued:#94bdb6;
  --color-accent:#1fe0c4; --color-accent-soft:#0e2a28;
  --link-text-color:#ffb24a; --link-text-color-hover:#ffd08a; --link-text-color-active:#ffb24a;
  --button-secondary-background-fill:#0e1a1b; --button-secondary-text-color:#7fe9da;
  --button-secondary-border-color:#1c5249;
  --table-odd-background-fill:#0d1416; --table-even-background-fill:#0b1113; --table-border-color:#14403a;
  --checkbox-background-color:#060a0b; --checkbox-background-color-selected:#0bbfa6;
  --stat-background-fill:#0d1416; --code-background-fill:#060a0b;
  /* gr.Radio / gr.CheckboxGroup option pills (Mode: Text->Video/Restyle/... ) — these use the
     checkbox-LABEL vars, which default white; that's what made the pills white w/ teal text. */
  --checkbox-label-background-fill:#0e1a1b; --checkbox-label-background-fill-hover:#13302c;
  --checkbox-label-background-fill-selected:#0bbfa6;
  --checkbox-label-text-color:#cfe8e3; --checkbox-label-text-color-selected:#05100e;
  --checkbox-label-border-color:#1c5249; --checkbox-label-border-color-selected:#1fe0c4; }
/* readable body text — Gradio sets gray per-element, which is invisible on near-black, so force it.
   Headings keep their teal rule (more specific element selectors below don't touch h1-h4). */
body.basi-liberated p, body.basi-liberated li, body.basi-liberated label,
body.basi-liberated td, body.basi-liberated .gr-text, body.basi-liberated .prose,
body.basi-liberated .prose p, body.basi-liberated .prose li, body.basi-liberated .prose strong,
body.basi-liberated .gr-check-radio label, body.basi-liberated .wrap,
body.basi-liberated span:not(.basi-title):not(.basi-sub) { color:#cfe8e3 !important; }
body.basi-liberated small, body.basi-liberated .gr-text-sm,
body.basi-liberated .prose em { color:#94bdb6 !important; }
body.basi-liberated::after { content:""; position:fixed; inset:0; pointer-events:none; z-index:9998;
  background:repeating-linear-gradient(180deg,rgba(0,0,0,0) 0 2px,rgba(0,0,0,.14) 2px 3px); opacity:.5; }
body.basi-liberated .block, body.basi-liberated .form, body.basi-liberated .gr-group,
body.basi-liberated .gr-box, body.basi-liberated .panel {
  background:#0d1416 !important; border:1px solid #14403a !important; border-radius:4px !important; }
body.basi-liberated h1, body.basi-liberated h2, body.basi-liberated h3, body.basi-liberated h4 {
  color:#2fe6cf !important; text-shadow:0 0 8px rgba(31,224,201,.45); letter-spacing:.04em; }
body.basi-liberated a { color:#ffb24a !important; }
body.basi-liberated button.primary, body.basi-liberated .primary,
body.basi-liberated button[variant="primary"] {
  background:linear-gradient(90deg,#0bbfa6,#ff8a1e) !important; color:#05100e !important; border:0 !important;
  font-weight:700; letter-spacing:.1em; text-transform:uppercase;
  box-shadow:0 0 16px rgba(24,217,194,.4),0 0 26px rgba(255,138,30,.22) !important; }
body.basi-liberated button.secondary, body.basi-liberated .secondary {
  background:#0e1a1b !important; color:#7fe9da !important; border:1px solid #1c5249 !important; }
body.basi-liberated .tab-nav button { color:#6b8b86 !important; text-transform:uppercase; letter-spacing:.08em;
  background:transparent !important; border:0 !important; }
body.basi-liberated .tab-nav button.selected { color:#2fe6cf !important; border-bottom:2px solid #ff8a1e !important;
  text-shadow:0 0 8px rgba(31,224,201,.5); }
body.basi-liberated textarea, body.basi-liberated input[type="text"], body.basi-liberated input[type="number"],
body.basi-liberated .gr-input, body.basi-liberated select {
  background:#060a0b !important; color:#d6efe9 !important; border:1px solid #1c4b46 !important; }
body.basi-liberated input[type="range"] { accent-color:#1fe0c4; }
/* Belt-and-suspenders for the Mode pills across Gradio builds (:has targets ONLY radio-option
   labels, never field labels or the lame-toggle checkbox). */
body.basi-liberated label:has(input[type="radio"]) {
  background:#0e1a1b !important; color:#cfe8e3 !important; border:1px solid #1c5249 !important; }
body.basi-liberated label:has(input[type="radio"]:checked) {
  background:linear-gradient(90deg,#0bbfa6,#ff8a1e) !important; color:#05100e !important;
  border-color:#1fe0c4 !important; text-shadow:none !important; }
body.basi-liberated .basi-header { display:flex; align-items:center; gap:18px; padding:6px 2px 0; }
body.basi-liberated .basi-logo { filter:drop-shadow(0 0 9px rgba(31,224,201,.5)) drop-shadow(0 0 16px rgba(255,138,30,.28)); }
body.basi-liberated .basi-title { font-size:30px; font-weight:800; letter-spacing:.18em; color:#2fe6cf;
  text-shadow:0 0 10px rgba(31,224,201,.55),0 0 2px #1fe0c4; }
body.basi-liberated .basi-sub { color:#8fb7b0; font-size:12.5px; letter-spacing:.06em; margin-top:3px; }
body.basi-liberated .basi-divider { color:#ff8a1e; text-align:center; letter-spacing:.04em; opacity:.85;
  margin:4px 0 10px; font-size:13px; text-shadow:0 0 8px rgba(255,138,30,.4); }
"""


def _basi_header_html():
    """Brand header: the Pinokio app icon (icon.png — the same logo Pinokio shows for the
    launcher), embedded as a data-URI so it needs no Gradio file route, + wordmark + an ASCII
    'liberated' divider. icon.png is large (launcher resolution), so we downscale it to header
    height before embedding to keep the inline blob small. The logo is inline-sized so it stays
    sane even in lame mode (when the scoped glow CSS is off)."""
    try:
        import io as _io
        from PIL import Image as _PILImage
        _ic = _PILImage.open(Path(__file__).resolve().parent / "icon.png").convert("RGBA")
        _h = 108  # 2x the 54px display height for crisp rendering
        _ic = _ic.resize((max(1, round(_ic.width * _h / _ic.height)), _h), _PILImage.LANCZOS)
        _buf = _io.BytesIO(); _ic.save(_buf, format="PNG")
        _b = base64.b64encode(_buf.getvalue()).decode()
        _img = (f'<img class="basi-logo" alt="BASI WAN KENOBI" '
                f'style="height:54px;width:auto;" '
                f'src="data:image/png;base64,{_b}"/>')
    except Exception:
        _img = ""
    return (
        '<div class="basi-header">' + _img +
        '<div><div class="basi-title">BASI WAN KENOBI</div>'
        '<div class="basi-sub">WAN2.2 VIDEO STUDIO · LoRA GYM · JOINT A/V (MOVA) '
        '— LIBERATED FOR THE GPU-POOR 🐉</div></div></div>'
        '<div class="basi-divider">.-.-.-.-&lt;=|  B4S1W4N : L1B3R4T3D  |=&gt;-.-.-.-.</div>'
    )


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

    with gr.Blocks(title="BASI WAN KENOBI", css=_BASI_LIBERATED_CSS,
                   js="()=>{document.body.classList.add('basi-liberated');}") as demo:
        gr.HTML(_basi_header_html())
        _lame = gr.Checkbox(value=True, container=False,
                            label="🐉 liberated mode  ·  uncheck for stock (lame) UI")
        _lame.change(None, _lame, None,
                     js="(v)=>{document.body.classList.toggle('basi-liberated', !!v);}")
        # Apply the liberated skin on first paint. The gr.Blocks(js=...) load hook fires too
        # early/unreliably (skin only appeared after toggling the checkbox); demo.load is the
        # canonical load event. Retry briefly in case <body> isn't ready on the first tick.
        demo.load(None, None, None, js=(
            "()=>{const a=()=>document.body&&document.body.classList.add('basi-liberated');"
            "a();setTimeout(a,60);setTimeout(a,300);}"))
        gr.Markdown(
            f"**GPU**: {cap.name} ({vram_gb} GB"
            + (f", sm_{cap.compute_major}{cap.compute_minor}"
               if cap.compute_major else "")
            + f") → training preset `{initial_preset}`, "
            f"preview path: {scheme_hint}.  "
            "Backend: musubi-tuner. Output: `.safetensors` in `outputs/<lora_name>/`."
        )
        with gr.Tabs():
            with gr.Tab("Studio"):
                # STUDIO tab: Fooocus-style minimal inference UI. Fully wired to
                # the vendored BASIWAN GGUF runner + v2 CUDA kernel (wan/, tools/):
                # T2V, generate-with-your-LoRA, Continue (I2V), Restyle (SDEdit +
                # sliding-window), Depth-lock (VACE) and Keyframe edit.
                studio_prompt = gr.Textbox(
                    label="Prompt",
                    placeholder=(
                        "Cinematic NYC alley chase: The camera starts shoulder-height "
                        "behind a hooded man steadily tracking forward as he weaves "
                        "through crowds. Cold tones, high contrast, neon lights. Smooth "
                        "glide with intense shake for immersive pursuer tension..."
                    ),
                    lines=3,
                    info="Wan 2.2 responds best to long descriptive prompts that "
                         "specify camera moves, lighting, motion, and atmosphere. "
                         "Try the examples below for shape/style reference.",
                )
                # Curated Wan 2.2 prompts: official Wan2.2 README (cat_boxing,
                # cat_surf) + ViewComfy Wan2.2 prompt guide (nyc_chase,
                # dappled_pan, ironman_dolly). These are the same prompts in
                # tools/prod_shape_bench.py — verified to produce quality output
                # across all production shapes (p480_17f to p720_33f).
                gr.Examples(
                    examples=[
                        ["Two anthropomorphic cats in comfy boxing gear and bright "
                         "gloves fight intensely on a spotlighted stage."],
                        ["Summer beach vacation style, a white cat wearing sunglasses "
                         "sits on a surfboard. The fluffy-furred feline gazes directly "
                         "at the camera with a relaxed expression. Blurred beach scenery "
                         "forms the background featuring crystal-clear waters, distant "
                         "green hills, and a blue sky dotted with white clouds. A close-up "
                         "shot highlights the feline's intricate details and the "
                         "refreshing atmosphere of the seaside."],
                        ["Cinematic NYC alley chase: The camera starts shoulder-height "
                         "behind a hooded man steadily tracking forward as he weaves "
                         "through crowds. Cold tones, high contrast, neon lights. Smooth "
                         "glide with intense shake for immersive pursuer tension. "
                         "Blurred steam and wet pavement. Lens flare, shallow depth of field."],
                        ["A low angle shot of a young man in dappled sunlight. "
                         "Backlighting, warm low-saturation tones. Slow-motion glide "
                         "with handheld tremor for dreamy nostalgia. Blurred foliage "
                         "for emotional focus. Camera pans left to low angle shot of "
                         "a cute girl."],
                        ["In the style of an American drama promotional poster, Iron "
                         "Man sits in a sleek, futuristic metal chair inside a dimly "
                         "lit industrial setting. He is fully suited in his iconic red "
                         "and gold armor, the arc reactor glowing in his chest. Around "
                         "him are scattered high-tech gadgets, and stacks of prototype "
                         "schematics. He sits still, helmet off, revealing Tony Stark's "
                         "face confident, composed, with a subtle smirk. Camera dollies "
                         "out. The background shows an abandoned, dim factory with "
                         "light filtering through the windows. There's a noticeable "
                         "grainy texture. A medium shot with a straight-on close-up "
                         "of the character."],
                    ],
                    inputs=[studio_prompt],
                    label="Wan 2.2 example prompts (click to use)",
                )
                # Explicit mode selector -- every mode is a peer. Picking a mode reveals
                # only that mode's extra inputs below; Text->Video uses just the prompt +
                # shape. The generate handlers stay input-presence based, so this is a
                # pure clarity layer.
                studio_mode = gr.Radio(
                    ["Text → Video", "Restyle", "Continue", "Keyframe", "Talking character",
                     "Joint A/V (MOVA)"],
                    value="Text → Video", label="Mode",
                    info="What to make. The extra inputs for Restyle / Continue / "
                         "Keyframe / Talking-character / Joint A/V appear below when selected.")
                with gr.Accordion("Advanced", open=False) as adv_acc:
                    # System-aware choices: filter resolution + frames to what the
                    # detected card's inference tier can run, so a smaller card never
                    # SEES (let alone picks) a 720p/long config that OOMs. 12/8GB tiers
                    # cap to 480p; 16GB->49 frames, 12GB->33, 8GB->17 (basi.presets
                    # _INFERENCE_TIERS, measured). S2V has its own stricter gate.
                    _tier0 = _get_infer_tier()
                    _all_res = [
                        ("480p preview (832x480)", "832x480"),
                        ("720p std (1280x720)", "1280x720"),
                        ("720p portrait (720x1280)", "720x1280"),
                    ]
                    _res_choices = [c for c in _all_res
                                    if (lambda d: int(d[0]) * int(d[1]))(c[1].split("x"))
                                    <= _tier0["max_px"]] or [_all_res[0]]
                    _res_default = ("1280x720"
                                    if any(c[1] == "1280x720" for c in _res_choices)
                                    else _res_choices[0][1])
                    _all_frames = [("17 (~1s)", 17), ("33 (~2s)", 33),
                                   ("49 (~3s)", 49), ("81 (~5s)", 81)]
                    _frame_choices = [c for c in _all_frames
                                      if c[1] <= _tier0["max_frames"]] or [_all_frames[0]]
                    with gr.Row():
                        studio_resolution = gr.Dropdown(
                            label="Resolution",
                            choices=_res_choices,
                            value=_res_default,
                            info="Output size. Wan2.2 supports 480p (832x480) and 720p "
                                 "(1280x720); higher = sharper but more VRAM + slower. "
                                 "List is filtered to what your card can run.",
                        )
                        studio_frames = gr.Dropdown(
                            label="Frames",
                            choices=_frame_choices,
                            value=_frame_choices[0][1],
                            info="Clip length (Wan needs 4n+1 frames; the video VAE "
                                 "compresses time 4x). More frames = longer clip + more VRAM.",
                        )
                        studio_steps = gr.Slider(
                            label="Steps", minimum=4, maximum=20, step=1, value=STUDIO["t2v_steps"],
                            info="Denoise passes — applies to Text→Video and Continue (Lightning "
                                 "4-step distilled, so 4 is the tuned fast default). Restyle, "
                                 "Keyframe and Talking-character run their own fixed recipe (8 / "
                                 "50 / 40 steps) and ignore this slider.",
                        )
                    with gr.Row():
                        studio_seed = gr.Number(label="Seed", value=STUDIO["seed"], precision=0,
                                                info="-1 = random")
                        studio_guide = gr.Slider(label="Guidance",
                                                 minimum=1.0, maximum=10.0,
                                                 step=0.1, value=STUDIO["t2v_guidance"],
                                                 info="Prompt adherence (CFG) for Text→Video / "
                                                 "Continue. Lightning runs CFG-free, so ~1.0 is "
                                                 "the fast default. Keyframe and Talking-character "
                                                 "use their own guidance (5.0 / 4.5) and ignore "
                                                 "this.")
                    with gr.Row():
                        with gr.Row():
                            studio_lora = gr.Dropdown(
                                label="Character LoRA",
                                choices=[l for l, _ in _list_studio_loras()],
                                value="(none — Lightning only)",
                                scale=5,
                                info="Optional character/style LoRA you trained in the Gym. "
                                     "'(none)' = base Lightning model only.",
                            )
                            studio_lora_refresh = gr.Button("🔄", scale=1,
                                                            min_width=40)
                        studio_lora_strength = gr.Slider(
                            label="Character LoRA strength",
                            minimum=0.0, maximum=1.5, step=0.05, value=STUDIO["user_lora_strength"],
                            info=("Lightning stays at 1.0; this scales only "
                                  "your LoRA. Depth-lock (VACE) applies it "
                                  "instantly at Generate; Fast/T2V rebuilds a "
                                  "cached combo (~seconds). Peaks near 1.0; "
                                  "above ~1.2 over-cooks."),
                        )
                        studio_lora_refresh.click(
                            lambda: gr.update(
                                choices=[l for l, _ in _list_studio_loras()]),
                            outputs=[studio_lora])
                studio_generate_btn = gr.Button("Generate", variant="primary")
                studio_video = gr.Video(label="Output", interactive=False)
                studio_status = gr.Markdown()

                # Restyle: re-render an uploaded clip in the selected
                # Character LoRA's style, preserving its motion/structure
                # (SDEdit partial-noise V2V). Reuses the same worker + combo
                # path as Generate; the prompt + LoRA define the new look.
                with gr.Accordion("Restyle a video (V2V)", open=True, visible=False) as restyle_acc:
                    restyle_video = gr.Video(
                        label="Source clip (uses the shape + prompt above; "
                              "trim to your frame count)")
                    restyle_denoise = gr.Slider(
                        label="Restyle strength (denoise)",
                        minimum=0.4, maximum=0.9, step=0.05, value=STUDIO["restyle_denoise"],
                        info=("Lower = keep more of the source (subtle); "
                              "higher = stronger restyle. <0.5 only touches "
                              "the low-noise expert. (Fast/SDEdit mode only.)"))
                    # Restyle method. Fast = 8-step SDEdit (keeps source
                    # via partial noise; uses the denoise slider + your LoRA).
                    # Depth-lock = VACE-Fun depth control (structure-lock corr
                    # 0.973): full-denoise QUALITY recipe (50 steps/guide 5.0,
                    # ignores the denoise slider), swaps to the VACE model on
                    # first use (~one-time cache build). ~440s/clip p480_17f.
                    restyle_mode = gr.Radio(
                        ["Fast (SDEdit)", "Depth-lock (VACE)"],
                        value="Fast (SDEdit)", label="Restyle method",
                        info="Fast (SDEdit): quick partial-noise restyle from your prompt + "
                             "LoRA (uses the denoise slider). Depth-lock (VACE): preserves the "
                             "source's structure/geometry via depth control -- slower, higher "
                             "fidelity, ignores the denoise slider.")
                    restyle_btn = gr.Button("Restyle", variant="primary")
                # Keyframe-anchored editing (Ray Modify parity). Upload your
                # EDITED frames + their positions; the model keeps them and generates
                # a video anchored to them. Routes to the GGUF-validated VACE
                # vace_edit worker path (anchors hard-locked, near-perfect preserve).
                with gr.Accordion("Keyframe edit (VACE)", open=True, visible=False) as kf_acc:
                    gr.Markdown(
                        "Upload your **edited frames** and their **positions**; the "
                        "model preserves them and generates a video anchored to "
                        "them. 2-4 anchors works best (e.g. first + last). Anchor "
                        "RGB is preserved near-perfectly; propagation to in-between "
                        "frames is best-effort (Fun-A14B checkpoint limit). Uses the "
                        "prompt / shape / frame-count set above.")
                    kf_images = gr.File(
                        label="Anchor frames (ordered, same order as positions)",
                        file_count="multiple", file_types=["image"])
                    kf_positions = gr.Textbox(
                        label="Anchor positions (0-based frame indices, "
                              "comma-separated)", value="0",
                        placeholder="e.g. 0, 16  (first + last of a 17-frame clip)")
                    kf_btn = gr.Button("Generate from keyframes", variant="primary")
                # State carrying continue_image=None for the restyle path,
                # which shares _studio_generate (continue_image is positional).
                _restyle_no_continue = gr.State(value=None)

                # Continue-from-last-frame (FinalFrame pattern, done
                # natively): pick a tail frame of the previous output, write
                # a motion-first continuation prompt (or let Qwen3-VL draft
                # one), generate with the I2V expert pair, stitch.
                with gr.Accordion("Continue last video", open=True, visible=False) as continue_acc:
                    gr.Markdown(
                        "Extends the last generated video from one of its "
                        "final frames (sharpest first — the very last frame "
                        "is often motion-blurred). Continuation prompts work "
                        "best **short and motion-first**: keep the character/"
                        "style words, describe only the next action. First "
                        "Continue click loads the I2V model (one-time cache "
                        "build on the very first use)."
                    )
                    continue_gallery = gr.Gallery(
                        label="Pick the frame to continue from (sharpest first)",
                        columns=5, height=140, interactive=False)
                    continue_pick = gr.State(0)
                    continue_prompt = gr.Textbox(
                        label="Continuation prompt",
                        lines=3,
                        info="What happens next + camera movement. Identity/"
                             "style words copied from your prompt; the frame "
                             "already carries the scene.",
                    )
                    with gr.Row():
                        continue_suggest_btn = gr.Button(
                            "Suggest next prompt (Qwen3-VL)", variant="secondary")
                        continue_btn = gr.Button("Continue", variant="primary")
                    continue_status = gr.Markdown()

                # System-aware S2V accordion: gate on the card's measured S2V
                # capability so a too-small GPU can't launch a brick. The button is
                # disabled (with the reason) when S2V isn't viable; the tier note is
                # always shown so the user knows the resolution ceiling up front.
                _s2v_caps0 = _get_s2v_caps()
                with gr.Accordion("Talking character (S2V)", open=True, visible=False) as s2v_acc:
                    gr.Markdown(
                        "Audio-driven talking character: upload a **reference "
                        "image** (the character/face) and a **speech audio** track; "
                        "Wan2.2-S2V lip-syncs the character to the audio. Runs the "
                        "quality recipe (40 steps). First click loads the S2V model "
                        "(one-time cache build). Portrait/square framing works best "
                        "for talking heads.\n\n"
                        + (f"**Your GPU:** {_s2v_caps0['note']}"
                           if _s2v_caps0["available"]
                           else f"**Unavailable:** {_s2v_caps0['note']}")
                    )
                    s2v_ref = gr.Image(label="Reference character image",
                                       type="filepath")
                    s2v_audio = gr.Audio(label="Driving audio (speech .wav)",
                                         type="filepath")
                    s2v_btn = gr.Button(
                        "Generate talking video", variant="primary",
                        interactive=_s2v_caps0["available"])
                    s2v_status = gr.Markdown()

                # Joint A/V (MOVA): TEXT -> audio+video. Runs the trained MOVA LoRA
                # (or base) in the dedicated env_mova venv via tools/mova_sample.py. T2AV —
                # the prompt drives everything; no image upload (the sampler synthesizes the
                # neutral first frame). Defaults are OUR measured recipe (MovaPreset mova_24g:
                # 240x320, 81f, 50 steps, group offload, NF4-base auto). System-gated on
                # VRAM (~12GB) + host RAM (~50-80GB) so a too-small box can't launch a brick.
                from basi import mova_infer as _mova_infer
                _mova_caps0 = _mova_infer.mova_caps(detect_vram_gb(), _mova_infer.detect_ram_gb())
                _mova_ready = _mova_caps0["available"] and _mova_infer.mova_installed()
                with gr.Accordion("Joint A/V (MOVA)", open=True, visible=False) as mova_acc:
                    gr.Markdown(
                        "Generate **audio + video together** from a text prompt + a **reference "
                        "image** (MOVA is image-to-video). Pick a MOVA LoRA you trained in the Gym "
                        "(or the base). The reference frame is the **opening shot** — it sets both "
                        "the **style** and the **scene/background**, and the prompt drives the "
                        "motion + speech. So for a given setting, use a reference in that setting "
                        "(your LoRA ships an in-style one, or upload your own). 240p, ~3.4s clips; "
                        "first click loads the model (group offload).\n\n"
                        + (f"**Your machine:** {_mova_caps0['note']}"
                           if _mova_caps0["available"]
                           else f"**Unavailable:** {_mova_caps0['note']}")
                        + ("" if _mova_infer.mova_installed()
                           else "\n\n**MOVA not installed** — re-run Install (it provisions "
                                "env_mova + the MOVA-360p weights)."))
                    # MOVA has its OWN prompt box: the magic-rewrite writes the formatted
                    # prompt back HERE (right where you click Format) so you can review/edit
                    # it, instead of the shared T2V box higher up the page. Type a rough idea
                    # + your LoRA's trigger, click Format, review, then Generate.
                    from basi.caption import MOVA_PROMPT_GUIDE as _MOVA_GUIDE
                    gr.Markdown(f"**Prompt format** — {_MOVA_GUIDE}")
                    mova_prompt = gr.Textbox(
                        label="Prompt", lines=2,
                        placeholder='moralorel, a boy speaks, medium shot. He says, in English, "..."')
                    with gr.Row():
                        mova_trigger = gr.Textbox(
                            label="Trigger word", value="", scale=2,
                            placeholder="your LoRA's word, e.g. moralorel")
                        mova_rewrite_btn = gr.Button(
                            "✨ Format prompt for MOVA", variant="secondary", scale=2)
                    with gr.Row():
                        mova_lora = gr.Dropdown(
                            label="MOVA LoRA",
                            choices=[l for l, _ in _mova_infer.list_mova_loras()],
                            value="(none — MOVA base, no LoRA)", scale=4)
                        mova_lora_refresh = gr.Button("🔄", scale=1, min_width=48)
                    # The reference is OPTIONAL and EMPTY by default. Leaving it empty is the whole
                    # point of T2AV: at generate time we PAINT a prompt-matched still in the LoRA's
                    # style (SDXL + IP-Adapter; style source = the LoRA's internal style frame) so the
                    # SCENE follows the prompt. Auto-filling it with the LoRA's bundled ref.png (one
                    # fixed frame) makes "uploaded reference always wins" fire and pins every
                    # generation to that single scene -- which defeats the prompt-matched reference.
                    # An explicit user upload still overrides (to pin a specific scene on purpose).
                    mova_ref = gr.Image(
                        label="Reference image (OPTIONAL). Leave empty — a prompt-matched still is "
                              "painted in your LoRA's style so the scene follows your prompt. Upload "
                              "only to pin a specific scene.",
                        type="filepath", height=180, value=None)
                    # Measured ceilings (user-confirmed). Pinokio is cross-OS: 240p all tiers; 480p
                    # ≥16GB; 720p sits at the 24GB edge and works on LINUX with a clean (non-display)
                    # GPU — but not on Windows, where the desktop compositor shares the card. So
                    # tier+platform gate rather than hide it from Linux users.
                    _v = detect_vram_gb(); _linux = sys.platform.startswith("linux")
                    # 240p + 360p on every MOVA-capable card (inference is far lighter than training,
                    # so 360p generates fine even where 360p TRAINING needs 16GB). 360p uses the same
                    # 464x352 bucket the Gym trains at, so a 360p LoRA generates at its trained dims.
                    _mova_res_choices = ["320x240 (240p — trained baseline, fastest)",
                                         "464x352 (360p — sharper, still light)"]
                    if _v >= 16:
                        _mova_res_choices.append("640x480 (480p — sharper, more VRAM/time)")
                    if _v >= 24 and _linux:
                        _mova_res_choices.append("960x720 (720p — Linux + clean GPU; at the VRAM edge)")
                    mova_res = gr.Dropdown(
                        label="Resolution", choices=_mova_res_choices, value=_mova_res_choices[0],
                        info="240p & 360p on all cards (360p matches the Gym's 360p training bucket); "
                             "480p needs ~16GB+. 720p sits at the 24GB edge — shown only on Linux with "
                             "a clean (non-display) GPU; on Windows the desktop shares the GPU.")
                    with gr.Row():
                        mova_frames = gr.Slider(25, 81, value=81, step=8,
                            label="Frames (24fps)",
                            info="81 = ~3.4s, the trained length. Fewer = faster.")
                        mova_steps = gr.Slider(20, 50, value=50, step=5,
                            label="Denoise steps",
                            info="MOVA recipe = 50; fewer underdenoises.")
                        mova_cfg = gr.Slider(1.0, 12.0, value=5.0, step=0.5,
                            label="Guidance (CFG)",
                            info="MOVA recipe = 5.0. Higher follows the prompt harder (risk of "
                                 "artifacts); lower is looser.")
                    mova_btn = gr.Button(
                        "Generate A/V", variant="primary", interactive=_mova_ready)
                    mova_status = gr.Markdown()
                    # Interactive continuation (mirrors the Wan 'Continue' mode): each step adds a clip
                    # that starts from a chosen FINAL FRAME of the current video, driven by a NEW
                    # prompt — the per-segment prompting that makes a multi-beat A/V coherent. The
                    # resident MOVA worker holds the running clip; this only sends the next beat.
                    with gr.Accordion("➕ Continue — extend with a new beat", open=False):
                        gr.Markdown("Add a clip that continues from a final frame of the current "
                                    "video, driven by a **new** prompt (a small motion beat). Pick the "
                                    "frame to continue from, write what happens next, or let Qwen3-VL "
                                    "suggest it. Repeat to build a longer, coherent scene.")
                        mova_cont_gallery = gr.Gallery(
                            label="Continue from this frame (ranked best-first: sharp + at a speech "
                                  "pause). Click to override.",
                            columns=5, height=120, interactive=False)
                        mova_cont_pick = gr.State(0)
                        mova_cont_prompt = gr.Textbox(
                            label="Continuation prompt (the next beat)", lines=2,
                            placeholder='moralorel, the boy turns to the dragon. He says, in English, "..."')
                        with gr.Row():
                            mova_cont_suggest = gr.Button("✨ Suggest next beat (Qwen3-VL)", scale=2)
                            mova_cont_btn = gr.Button("Continue ▶", variant="primary", scale=2)
                        mova_cont_status = gr.Markdown()
                    mova_lora_refresh.click(
                        lambda: gr.update(choices=[l for l, _ in _mova_infer.list_mova_loras()]),
                        outputs=[mova_lora])

                # Studio Generate: subprocess the vendored runner using the
                # SAME Pinokio venv Python. No WSL, native Windows path.
                # The runner (tools/run_one_video_gguf.py) is vendored
                # alongside wan/ and tools/gguf_vendor/ in this app.
                _BASIWAN_RUNNER_DIR = str(BASI_ROOT)
                # All inference checkpoints resolve under BASIWAN_CKPT_DIR
                # (default <repo>/checkpoints) — set that one env var to point at
                # a shared model drive. Each model also keeps its own BASIWAN_*
                # override. GGUF resolution is layout-agnostic (HF-cache snapshot
                # OR flat) via _basiwan_resolve_gguf, so a plain `huggingface-cli
                # download` into checkpoints/gguf just works.
                _CKPT_BASE = Path(os.environ.get(
                    "BASIWAN_CKPT_DIR", str(BASI_ROOT / "checkpoints")))
                _BASES = [_CKPT_BASE]

                _BASIWAN_CKPT_BASE = _basiwan_resolve_ckpt(
                    os.environ.get("BASIWAN_CKPT_BASE_INFERENCE"),
                    _BASES, "Wan2.2-T2V-A14B")
                _BASIWAN_GGUF_HIGH = _basiwan_resolve_gguf(
                    os.environ.get("BASIWAN_GGUF_HIGH"), _BASES,
                    "QuantStack--Wan2.2-T2V-A14B-GGUF", "HighNoise",
                    "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf")
                _BASIWAN_GGUF_LOW = _basiwan_resolve_gguf(
                    os.environ.get("BASIWAN_GGUF_LOW"), _BASES,
                    "QuantStack--Wan2.2-T2V-A14B-GGUF", "LowNoise",
                    "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf")
                _BASIWAN_LIGHTNING_LORA = _basiwan_resolve_ckpt(
                    os.environ.get("BASIWAN_LIGHTNING_LORA"), _BASES,
                    "Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928")
                # I2V expert pair + I2V Lightning LoRA for the Continue-from-
                # last-frame feature. The T2V Lightning LoRA must NEVER be
                # applied to I2V experts (it destroys generation) — hence the
                # separate Seko-V1 dir.
                _BASIWAN_GGUF_HIGH_I2V = _basiwan_resolve_gguf(
                    os.environ.get("BASIWAN_GGUF_HIGH_I2V"), _BASES,
                    "QuantStack--Wan2.2-I2V-A14B-GGUF", "HighNoise",
                    "Wan2.2-I2V-A14B-HighNoise-Q4_K_M.gguf")
                _BASIWAN_GGUF_LOW_I2V = _basiwan_resolve_gguf(
                    os.environ.get("BASIWAN_GGUF_LOW_I2V"), _BASES,
                    "QuantStack--Wan2.2-I2V-A14B-GGUF", "LowNoise",
                    "Wan2.2-I2V-A14B-LowNoise-Q4_K_M.gguf")
                _BASIWAN_LIGHTNING_LORA_I2V = _basiwan_resolve_ckpt(
                    os.environ.get("BASIWAN_LIGHTNING_LORA_I2V"), _BASES,
                    "Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1")
                # VACE-Fun depth-control GGUF pair — the depth-locked QUALITY
                # restyle model (depth-lock corr 0.973). NO Lightning
                # (50-step/guide-5.0 recipe). Same layout-agnostic glob.
                _BASIWAN_GGUF_HIGH_VACE = _basiwan_resolve_gguf(
                    os.environ.get("BASIWAN_GGUF_HIGH_VACE"), _BASES,
                    "QuantStack--Wan2.2-VACE-Fun-A14B-GGUF", "HighNoise",
                    "Wan2.2-VACE-Fun-A14B-HighNoise-Q4_K_M.gguf")
                _BASIWAN_GGUF_LOW_VACE = _basiwan_resolve_gguf(
                    os.environ.get("BASIWAN_GGUF_LOW_VACE"), _BASES,
                    "QuantStack--Wan2.2-VACE-Fun-A14B-GGUF", "LowNoise",
                    "Wan2.2-VACE-Fun-A14B-LowNoise-Q4_K_M.gguf")
                # Wan2.2-S2V (audio-driven talking character) — SINGLE-expert
                # GGUF (no HighNoise/LowNoise split, so the standard which-subdir
                # resolver doesn't fit) + the checkpoint dir holding config.json +
                # wav2vec2-large-xlsr-53-english/ (T5/VAE come from the T2V base).
                def _resolve_s2v_gguf():
                    import glob as _gg
                    ev = os.environ.get("BASIWAN_GGUF_S2V")
                    if ev and Path(ev).exists():
                        return ev
                    for b in _BASES:
                        hits = sorted(_gg.glob(str(
                            Path(b) / "gguf" / "models--QuantStack--Wan2.2-S2V-14B-GGUF"
                            / "snapshots" / "*" / "Wan2.2-S2V-14B-Q4_K_M.gguf")))
                        if hits:
                            return hits[0]
                    return str(Path(_BASES[0]) / "gguf" / "Wan2.2-S2V-14B-Q4_K_M.gguf")
                _BASIWAN_GGUF_S2V = _resolve_s2v_gguf()
                _BASIWAN_S2V_DIR = _basiwan_resolve_ckpt(
                    os.environ.get("BASIWAN_S2V_DIR"), _BASES, "Wan2.2-S2V-14B")

                def _studio_env() -> dict:
                    """Ship-recipe env for the runner (worker AND legacy paths).

                    setdefault so start.js daemon inheritance wins when set;
                    a bare `python app.py` without Pinokio still gets the
                    canonical recipe."""
                    env = os.environ.copy()
                    env.setdefault("BASIWAN_V2", "1")
                    env.setdefault("BASIWAN_USE_PACK_CACHE", "1")
                    env.setdefault("BASIWAN_NO_POOL", "1")
                    env.setdefault("BASIWAN_VAE_BF16", "1")
                    # RMS/LN_BF16 deliberately NOT set: model.py auto-gates
                    # norms to bf16 only at seq>50000 (p720_81f) where fp32
                    # transients blow the 24 GB VRAM wall. Pinning "0" here
                    # causes an 81f allocator-thrash hang; scrub any inherited
                    # value:
                    env.pop("BASIWAN_RMS_BF16", None)
                    env.pop("BASIWAN_LN_BF16", None)
                    # Same default everywhere; _cache_dir() expanduser()s it.
                    env.setdefault("BASIWAN_PACK_CACHE_DIR",
                                   str(Path.home() / ".cache" / "marlin_packs"))
                    # Phase + per-step events power the live progress bar in
                    # worker mode. Without this the bar freezes from "generation
                    # started" until the result — at 20 steps × guide>1 that's
                    # many opaque minutes. ~25 prints per generation; negligible.
                    env.setdefault("BASIWAN_PHASE_PROFILE", "1")
                    return env

                # ---- persistent-worker singleton (shared by the eager
                # warm-up thread and Generate clicks) ----
                import threading as _threading
                _worker_holder: dict = {}
                _worker_lock = _threading.Lock()
                # Last successful Studio generation, for Continue:
                # {"tail_pngs": [paths sharpest-first], "mp4": str,
                #  "prompt": str, "resolution": str}. Only updated on
                # success, so a failed continuation keeps the previous
                # result continuable.
                _last_gen: dict = {}

                def _import_runner_client():
                    try:
                        from tools.runner_client import (
                            BasiwanRunner, RunnerDied)
                    except Exception:
                        sys.path.insert(0, str(BASI_ROOT / "tools"))
                        from runner_client import (  # type: ignore
                            BasiwanRunner, RunnerDied)
                    return BasiwanRunner, RunnerDied

                def _ensure_worker(progress_cb=None, model_type="t2v"):
                    """Return a ready BasiwanRunner for `model_type`,
                    starting (or swapping) one if needed.

                    Thread-safe: the eager warm-up thread and a Generate
                    click can race here; the loser blocks on the lock and
                    inherits the winner's worker instead of spawning a
                    second ~27 GB process. A live worker of the WRONG
                    model type is shut down first — T2V and I2V experts
                    are separate checkpoint pairs baked into the worker
                    CLI, and two workers can't coexist in host RAM.
                    Raises RunnerDied on failure (with the worker's log
                    tail folded into the message)."""
                    _BR, _RD = _import_runner_client()
                    t0 = time.time()
                    # Non-blocking acquire loop: if the eager warm-up thread
                    # (or another click) is mid-cold-start, surface ITS log
                    # tail through our progress_cb instead of freezing.
                    while not _worker_lock.acquire(timeout=1.0):
                        if progress_cb is not None:
                            r = _worker_holder.get("starting")
                            tail = r.get_human_tail(1) if r is not None else []
                            try:
                                progress_cb(time.time() - t0,
                                            tail[0] if tail else "")
                            except Exception:
                                pass
                    try:
                        runner = _worker_holder.get("runner")
                        if runner is not None and runner.is_alive():
                            if _worker_holder.get("model_type", "t2v") == model_type:
                                return runner
                            # Wrong expert pair loaded — swap.
                            try:
                                runner.shutdown()
                            except Exception:
                                pass
                            _worker_holder.pop("runner", None)
                        # Spawning a Wan worker: free the MOVA worker first (it holds ~17GB VRAM; the
                        # two can't coexist on a 24GB card). Symmetric to MOVA freeing the Wan worker.
                        _shutdown_mova_worker_if_idle()
                        if model_type == "i2v":
                            _gguf_high, _gguf_low = _BASIWAN_GGUF_HIGH_I2V, _BASIWAN_GGUF_LOW_I2V
                            _lora_dir = _BASIWAN_LIGHTNING_LORA_I2V
                        elif model_type == "vace":
                            # depth-locked QUALITY restyle — VACE-Fun pair,
                            # NO Lightning LoRA (runs the 50-step/guide-5.0 recipe
                            # the worker enforces from basi.vace.VACE_DEPTH_RECIPE).
                            _gguf_high, _gguf_low = _BASIWAN_GGUF_HIGH_VACE, _BASIWAN_GGUF_LOW_VACE
                            _lora_dir = None
                        elif model_type == "s2v":
                            # audio-driven talking character — SINGLE-expert
                            # GGUF (same path for high/low; the runner ignores low),
                            # NO Lightning (worker enforces S2V_RECIPE 40-step/4.5).
                            _gguf_high = _gguf_low = _BASIWAN_GGUF_S2V
                            _lora_dir = None
                        else:
                            _gguf_high, _gguf_low = _BASIWAN_GGUF_HIGH, _BASIWAN_GGUF_LOW
                            _lora_dir = _BASIWAN_LIGHTNING_LORA
                        # VACE boots FULLY RESIDENT (block-swap -1 = no swap). At
                        # p480_17f the VACE pair fits on 24GB, and block-swap
                        # fragments the allocator enough to trip the cuBLASLt
                        # F.linear 256 GiB heuristic bug — so it runs resident.
                        # T2V/I2V take the VRAM-tier N: 24GB->2 (champion),
                        # 16GB->10, 12GB->20, <12->30 (the tier adapts per card).
                        _bswap = ("-1" if model_type == "vace"
                                  else str(_get_infer_tier()["block_swap_n"]))
                        _serve_cli = [
                            "--model-type", model_type,
                            "--gguf-high", _gguf_high,
                            "--gguf-low", _gguf_low,
                            "--vae", str(Path(_BASIWAN_CKPT_BASE) / "Wan2.1_VAE.pth"),
                            "--base-dir", _BASIWAN_CKPT_BASE,
                            "--lora-mode", "forward",
                            "--block-swap-n", _bswap,
                            "--ffn-chunk-size", "4096",
                        ]
                        if _lora_dir and Path(_lora_dir).exists():
                            _serve_cli += ["--lora-dir", _lora_dir]
                        if model_type == "s2v":
                            _serve_cli += ["--s2v-dir", _BASIWAN_S2V_DIR]
                        # Prefer the explicit venv python (start.js sets
                        # BASIWAN_VENV_PYTHON). sys.executable under Pinokio
                        # resolves to bin\miniconda\python.exe — it works
                        # (venv packages still load via the activated env)
                        # but ties the worker to PATH/activation state we
                        # don't control.
                        _worker_py = os.environ.get(
                            "BASIWAN_VENV_PYTHON", sys.executable)
                        if not Path(_worker_py).exists():
                            _worker_py = sys.executable
                        runner = _BR(
                            python=_worker_py,
                            runner_script=str(BASI_ROOT / "tools" / "run_one_video_gguf.py"),
                            cli_args=_serve_cli, env=_studio_env(),
                            cwd=str(BASI_ROOT))
                        _worker_holder["starting"] = runner
                        try:
                            runner.start(progress_cb)
                        except _RD as e:
                            tail = "\n".join(runner.get_human_tail(40))
                            raise _RD(f"{e}\n\nworker log tail:\n{tail}") from e
                        _worker_holder["runner"] = runner
                        _worker_holder["model_type"] = model_type
                        return runner
                    finally:
                        _worker_holder.pop("starting", None)
                        _worker_lock.release()

                def _shutdown_worker_if_idle():
                    """Free the worker's ~20 GB host RAM (e.g. before
                    training). Next Generate pays a cached cold start
                    (~1-2 min), not the full re-pack."""
                    runner = _worker_holder.get("runner")
                    if runner is not None and runner.is_alive():
                        try:
                            runner.shutdown()
                        except Exception:
                            pass
                    _worker_holder.pop("runner", None)

                # --- MOVA persistent worker (env_mova) — the joint A/V model loads ONCE and serves
                # the base generation + every interactive Continue, mirroring the Wan worker above.
                _mova_worker_holder: dict = {}
                _mova_worker_lock = _threading.Lock()
                # Current MOVA continuation session, for the Continue UI:
                # {"mp4": str, "tail_pngs": [best-first PNG paths], "prompt": str}.
                _mova_last: dict = {}

                def _ensure_mova_worker(lora_dir=None, progress_cb=None):
                    """Return a ready BasiwanRunner for the persistent MOVA worker (mova_sample.py
                    --serve, env_mova). Starts one if needed; a LoRA change forces a respawn (the LoRA
                    is baked at load). Frees the Wan worker first — MOVA's group-offload onload needs
                    the VRAM. Raises RunnerDied on failure with the worker log tail folded in."""
                    _BR, _RD = _import_runner_client()
                    from basi import mova_infer as _mi
                    t0 = time.time()
                    while not _mova_worker_lock.acquire(timeout=1.0):
                        if progress_cb is not None:
                            r = _mova_worker_holder.get("starting")
                            tail = r.get_human_tail(1) if r is not None else []
                            try:
                                progress_cb(time.time() - t0, tail[0] if tail else "")
                            except Exception:
                                pass
                    try:
                        runner = _mova_worker_holder.get("runner")
                        if (runner is not None and runner.is_alive()
                                and _mova_worker_holder.get("lora", "") == (lora_dir or "")):
                            return runner
                        if runner is not None:
                            try:
                                runner.shutdown()
                            except Exception:
                                pass
                            _mova_worker_holder.pop("runner", None)
                        # Free the Wan worker (~9GB VRAM); MOVA onload needs ~12GB free.
                        _shutdown_worker_if_idle()
                        _serve_cli = ["--base", str(_mi.mova_base_dir()), "--serve", "--offload", "group"]
                        if lora_dir and Path(lora_dir).exists():
                            _serve_cli += ["--lora", lora_dir]
                        runner = _BR(
                            python=_mi.resolve_mova_python(),
                            runner_script=str(BASI_ROOT / "tools" / "mova_sample.py"),
                            cli_args=_serve_cli, env=_mi.mova_spawn_env(),
                            cwd=str(BASI_ROOT))
                        _mova_worker_holder["starting"] = runner
                        try:
                            runner.start(progress_cb)
                        except _RD as e:
                            tail = "\n".join(runner.get_human_tail(40))
                            raise _RD(f"{e}\n\nMOVA worker log tail:\n{tail}") from e
                        _mova_worker_holder["runner"] = runner
                        _mova_worker_holder["lora"] = (lora_dir or "")
                        return runner
                    finally:
                        _mova_worker_holder.pop("starting", None)
                        _mova_worker_lock.release()

                def _shutdown_mova_worker_if_idle():
                    """Free the MOVA worker's VRAM + host RAM (e.g. before a Wan generation or Gym run)."""
                    runner = _mova_worker_holder.get("runner")
                    if runner is not None and runner.is_alive():
                        try:
                            runner.shutdown()
                        except Exception:
                            pass
                    _mova_worker_holder.pop("runner", None)

                # Eager warm-up: start the cold load at app launch, not on the
                # first Generate click. The first ever start re-packs both
                # experts (~12 min, writes the pack cache); cached starts are
                # ~1-2 min. Either way the user's first click finds a warm (or
                # warming) worker. BASIWAN_WORKER_EAGER=0 disables.
                if (os.environ.get("BASIWAN_PERSISTENT_WORKER", "1") == "1"
                        and os.environ.get("BASIWAN_WORKER_EAGER", "1") == "1"
                        # Belt-and-suspenders: a refused second instance
                        # exits before build_ui() runs, so this normally never
                        # sees a False flag — but never spawn a 20 GB worker
                        # without the lock if build_ui() is ever called bare.
                        and _SINGLETON_LOCK_HELD):
                    def _eager_warm():
                        try:
                            _ensure_worker()
                            print("[basiwan] eager worker warm-up complete", flush=True)
                        except Exception as e:
                            print(f"[basiwan] eager worker warm-up failed: {e}", flush=True)
                    _threading.Thread(
                        target=_eager_warm, name="basiwan-eager-warm",
                        daemon=True).start()

                # Prefetch the Suggest-prompt VLM weights in the background at
                # launch. Without this, the first "Suggest next prompt" click
                # triggers an 8.3 GB HF download mid-click (~13 min of opaque
                # wait). snapshot_download is network+disk only (no GPU/RAM), so
                # it coexists with the worker warm-up. BASIWAN_PREFETCH_VLM=0
                # disables.
                if os.environ.get("BASIWAN_PREFETCH_VLM", "1") == "1":
                    def _prefetch_vlm():
                        # Suggest-button model (4B) first — small and
                        # interactive — then the tier-selected captioner
                        # (8B at 24G: ~17 GB).
                        from basi.caption import MODEL_TIERS, select_model_for_vram
                        for mid in dict.fromkeys(
                                [MODEL_TIERS[12], select_model_for_vram()]):
                            try:
                                from huggingface_hub import snapshot_download
                                snapshot_download(mid)
                                print(f"[basiwan] VLM prefetch complete: {mid}",
                                      flush=True)
                            except Exception as e:
                                print(f"[basiwan] VLM prefetch failed for "
                                      f"{mid} (will download on first use): "
                                      f"{e}", flush=True)
                    _threading.Thread(
                        target=_prefetch_vlm, name="basiwan-vlm-prefetch",
                        daemon=True).start()

                def _studio_generate(prompt, resolution, frames, steps, seed,
                                     guide, lora, lora_strength,
                                     continue_image=None,
                                     restyle_video=None, denoise_strength=1.0,
                                     restyle_mode="Fast (SDEdit)",
                                     kf_images=None, kf_positions="",
                                     s2v_audio=None, s2v_ref=None,
                                     progress=gr.Progress()):
                    """Subprocess the vendored runner using same Pinokio Python.

                    Inherits the canonical ship-recipe env from start.js
                    (BASIWAN_V2=1, USE_PACK_CACHE=1, NO_POOL=1, VAE_BF16=1, etc.).
                    The v2 CUDA kernel is built on Windows
                    (%LOCALAPPDATA%/torch_extensions/.../wan_q4k_q6k_basiwan_ext_r128.pyd)
                    and the ship-recipe path produces 63s wall at p720_17f vs the
                    legacy path's 200s+."""
                    if not prompt or not prompt.strip():
                        return None, "**Error**: empty prompt"
                    if not Path(_BASIWAN_GGUF_HIGH).exists():
                        return None, (
                            "### Wan2.2 GGUF weights not found\n\n"
                            f"Expected at `{_BASIWAN_GGUF_HIGH}`. Set "
                            "`BASIWAN_GGUF_HIGH` / `BASIWAN_GGUF_LOW` env vars."
                        )
                    width, height = (int(s) for s in resolution.split("x"))
                    out_dir = WORKSPACES / "_studio"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stamp = str(int(time.time()))
                    out_pt = out_dir / f"studio_{stamp}.pt"
                    out_meta = out_dir / f"studio_{stamp}.json"
                    out_mp4 = out_dir / f"studio_{stamp}.mp4"

                    cli = [
                        sys.executable,
                        str(BASI_ROOT / "tools" / "run_one_video_gguf.py"),
                        "--gguf-high", _BASIWAN_GGUF_HIGH,
                        "--gguf-low", _BASIWAN_GGUF_LOW,
                        "--vae", str(Path(_BASIWAN_CKPT_BASE) / "Wan2.1_VAE.pth"),
                        "--base-dir", _BASIWAN_CKPT_BASE,
                        "--lora-strength", str(lora_strength),
                        "--lora-mode", "forward",
                        "--prompt", prompt.strip(),
                        "--width", str(width),
                        "--height", str(height),
                        "--frames", str(int(frames)),
                        "--steps", str(int(steps)),
                        "--guide", str(guide),
                        "--block-swap-n", str(_get_infer_tier()["block_swap_n"]),
                        "--ffn-chunk-size", "4096",
                        "--out", str(out_pt),
                        "--meta", str(out_meta),
                    ]
                    if Path(_BASIWAN_LIGHTNING_LORA).exists():
                        cli += ["--lora-dir", _BASIWAN_LIGHTNING_LORA]

                    env = _studio_env()

                    # Persistent-worker path. When BASIWAN_PERSISTENT_WORKER is
                    # "1" (default ON in start.js), reuse a singleton
                    # subprocess.Popen of the runner --serve mode across Studio
                    # Generate clicks. Saves 250-700s subprocess re-init cost per
                    # click after the first. Set to "0" to fall back to the legacy
                    # subprocess-per-click path below (see tools/runner_client.py).
                    # Pre-flight wall estimate + VRAM-wall warning. Anchor: 64s at
                    # p720/17f/4-steps/guide=1 (measured champion recipe). Scaling:
                    # steps linear; guide != 1.0 doubles every step (CFG runs cond
                    # + uncond); frames ~(f/17)^1.5 (attention superlinear);
                    # resolution ~linear in pixels. ROUGH — order-of-magnitude
                    # honesty, not a promise: high frames at a fixed block-swap N
                    # can hit the 24 GB VRAM wall and allocator-thrash for hours.
                    # RAM-axis worker gate: the persistent worker + eager prewarm
                    # hold the ~19GB pack in page cache; on <32GB-RAM hosts that
                    # thrashes. If the env is UNSET, follow the tier (off on
                    # low-RAM hosts). An explicit BASIWAN_PERSISTENT_WORKER value
                    # always wins (honor user intent).
                    _persist_env = os.environ.get("BASIWAN_PERSISTENT_WORKER")
                    if _persist_env is None:
                        _use_worker = _get_infer_tier()["persistent_worker"]
                    else:
                        _use_worker = _persist_env == "1"
                    # S2V is EXCLUSIVE: when a driving audio is present the
                    # talking-character path takes precedence; null the other modes so
                    # mode-detection + request-building below can't double-fire.
                    if s2v_audio:
                        continue_image = None
                        restyle_video = None
                        kf_images = None
                    # Continuation rides the I2V expert pair; plain
                    # generation rides T2V. _ensure_worker swaps the worker
                    # when the loaded pair doesn't match.
                    # VACE depth-lock restyle rides the VACE-Fun pair.
                    _vace_restyle = bool(
                        restyle_video and not continue_image
                        and restyle_mode == "Depth-lock (VACE)")
                    # Keyframe-anchored editing: anchor images uploaded
                    # (no restyle/continue). Parse + validate positions; routes to
                    # the VACE worker (the vace_edit path).
                    _kf_paths, _kf_pos, _kf_err = [], [], None
                    _kf_edit = bool(kf_images and not restyle_video
                                    and not continue_image)
                    if _kf_edit:
                        _kf_list = (kf_images if isinstance(kf_images, (list, tuple))
                                    else [kf_images])
                        _kf_paths = [getattr(f, "name", f) for f in _kf_list]
                        try:
                            _kf_pos = [int(p.strip()) for p in
                                       str(kf_positions).split(",") if p.strip()]
                        except ValueError:
                            _kf_err = "positions must be comma-separated integers"
                        if _kf_err is None and len(_kf_pos) != len(_kf_paths):
                            _kf_err = (f"{len(_kf_paths)} anchor image(s) but "
                                       f"{len(_kf_pos)} position(s) — counts must match")
                        elif _kf_err is None and any(
                                not (0 <= p < int(frames)) for p in _kf_pos):
                            _kf_err = f"positions must be in [0, {int(frames)})"
                    _mtype = ("s2v" if s2v_audio
                              else "i2v" if continue_image
                              else "vace" if (_vace_restyle or _kf_edit) else "t2v")
                    if s2v_audio and not _use_worker:
                        return None, ("### Talking character (S2V) requires the "
                                      "persistent worker (BASIWAN_PERSISTENT_WORKER=1)")
                    if s2v_audio and not s2v_ref:
                        return None, ("### S2V needs a reference image\n\nUpload the "
                                      "character/face image to drive with the audio.")
                    if s2v_audio and not Path(_BASIWAN_GGUF_S2V).exists():
                        return None, (
                            "### S2V weights not found\n\nTalking character needs "
                            "Wan2.2-S2V-14B-Q4_K_M.gguf (QuantStack/Wan2.2-S2V-14B-GGUF) "
                            "+ the Wan2.2-S2V-14B dir (config.json + wav2vec2). Set "
                            "`BASIWAN_GGUF_S2V` / `BASIWAN_S2V_DIR`.")
                    # System-aware gate: refuse S2V configs this GPU can't run so a
                    # too-small card (or a 720p pick) can't brick. S2V is ~2x heavier than
                    # T2V; caps are measured (s2v_caps). The runtime residency cap is the
                    # backstop, but blocking here gives a clear message instead of an OOM.
                    if s2v_audio:
                        _sc = _get_s2v_caps()
                        if not _sc["available"]:
                            return None, ("### Talking character not supported on this GPU"
                                          f"\n\n{_sc['note']}")
                        if width * height > _sc["max_px"]:
                            return None, (
                                "### Resolution too high for talking character\n\n"
                                f"{_sc['note']}\n\nThis GPU ({_sc['tier']} tier) runs S2V "
                                "up to **480p (832x480)** — pick that resolution. 720p "
                                "talking-character sits at the VRAM wall even on 24GB "
                                "cards; 480p is the supported S2V resolution.")
                    if _kf_edit and _kf_err:
                        return None, f"### Keyframe edit input error\n\n{_kf_err}"
                    if _kf_edit and not _use_worker:
                        return None, ("### Keyframe edit requires the persistent "
                                      "worker (BASIWAN_PERSISTENT_WORKER=1)")
                    if _kf_edit and not Path(_BASIWAN_GGUF_HIGH_VACE).exists():
                        return None, ("### VACE weights not found\n\nKeyframe edit "
                                      "needs the Wan2.2-VACE-Fun-A14B GGUF pair "
                                      "(QuantStack/Wan2.2-VACE-Fun-A14B-GGUF Q4_K_M).")
                    if _vace_restyle and not _use_worker:
                        return None, ("### Depth-lock (VACE) requires the "
                                      "persistent worker (BASIWAN_PERSISTENT_WORKER=1)")
                    if _vace_restyle and not Path(_BASIWAN_GGUF_HIGH_VACE).exists():
                        return None, (
                            "### VACE weights not found\n\nDepth-lock restyle needs "
                            "the Wan2.2-VACE-Fun-A14B GGUF pair. Download "
                            "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF (Q4_K_M) into "
                            "checkpoints/gguf/, or use Fast (SDEdit) restyle.")
                    if continue_image and not _use_worker:
                        return None, ("### Continue requires the persistent "
                                      "worker (BASIWAN_PERSISTENT_WORKER=1)")
                    if continue_image and not Path(_BASIWAN_GGUF_HIGH_I2V).exists():
                        return None, (
                            "### I2V weights not found\n\nContinue needs the "
                            "Wan2.2-I2V-A14B GGUF pair. Expected at "
                            f"`{_BASIWAN_GGUF_HIGH_I2V}`. Set "
                            "`BASIWAN_GGUF_HIGH_I2V` / `BASIWAN_GGUF_LOW_I2V`."
                        )
                    log_tail: list[str] = []
                    if _use_worker:
                        _, _RD = _import_runner_client()
                        _existing = _worker_holder.get("runner")
                        if (_existing is not None and _existing.is_alive()
                                and _worker_holder.get("model_type", "t2v") == _mtype):
                            progress(0.05, desc="reusing warm BASIWAN worker")
                        elif continue_image:
                            progress(0.02, desc="switching to I2V model "
                                                 "(one-time load; first ever "
                                                 "I2V start builds a cache)")
                        else:
                            progress(0.02, desc="loading model (first run after "
                                                 "launch; later runs are instant)")

                        def _warm_cb(elapsed, line):
                            # Surface the worker's latest log line so the
                            # user sees pack/load phases instead of a frozen
                            # bar. Cap the fraction so generation phases
                            # still own 0.08+.
                            frac = 0.02 + min(0.05, elapsed / 7200.0)
                            tail = (line or "").strip()[-90:]
                            progress(frac, desc=(
                                f"loading model ({int(elapsed)}s)"
                                + (f": {tail}" if tail else "")))

                        try:
                            runner = _ensure_worker(_warm_cb, model_type=_mtype)
                        except Exception as e:
                            return None, f"### Worker failed to start\n\n```\n{e}\n```"
                        # User-LoRA hot-swap. Resolve the dropdown selection →
                        # build a cached Lightning+user combo → set_lora on the
                        # live worker (no restart, ~0.3 s). combo state is tracked
                        # PER RUNNER (runner._lora_key) so a fresh/respawned worker
                        # re-applies rather than silently dropping to plain Lightning.
                        # Restyle quality recipe: 8-step mid-entry with Lightning
                        # at 0.7 (denoise 0.6) scored style_sim 0.812 + tightest
                        # structure vs 4-step/L1.0 at 0.750. Reduced Lightning lets
                        # the high-noise expert (global style) impose the style
                        # without distill dilution. The 0.7 only applies on the
                        # restyle path; plain T2V keeps 1.0 (and its byte-identical
                        # combo cache key).
                        _restyle_active = bool(
                            restyle_video and not continue_image
                            and float(denoise_strength) < 1.0
                            and not _vace_restyle)  # SDEdit XOR VACE
                        _light_s = (RESTYLE_SDEDIT_RECIPE["lightning_strength"]
                                    if _restyle_active else 1.0)
                        _combo_on = False        # Lightning+user combo (T2V): strength baked → pinned
                        _vace_lora_on = False     # user-only LoRA (VACE): strength scaled at request
                        _plan = {"want_key": None, "request_strength": float(lora_strength)}
                        try:
                            # User LoRA on T2V (Lightning+user combo) and VACE
                            # depth-lock (user-only). The I2V Continue flow uses
                            # the Seko-I2V Lightning LoRA; a T2V-trained user LoRA
                            # there is unvalidated → force "(none)". The routing
                            # decision is the pure, unit-tested _plan_user_lora();
                            # here we only execute its build + set_lora I/O.
                            _sel = (next((p for l, p in _list_studio_loras()
                                          if l == lora), None)
                                    if _mtype in ("t2v", "vace") else None)
                            _plan = _plan_user_lora(
                                # keyframe-edit (_kf_edit) is ALSO a 50-step full-CFG VACE path with NO
                                # Lightning, so it must take the user-only LoRA branch like restyle --
                                # else the T2V 4-step Lightning LoRA gets baked into the VACE-Fun experts
                                # and silently degrades the edit.
                                mtype=_mtype, vace_restyle=(_vace_restyle or _kf_edit), lora=lora,
                                sel=_sel, lora_strength=lora_strength,
                                light_s=_light_s,
                                lightning_lora=_BASIWAN_LIGHTNING_LORA,
                                useronly_root=WORKSPACES / "_studio" / "_useronly_cache",
                                combo_root=WORKSPACES / "_studio" / "_combo_cache")
                            _combo_on = _plan["mode"] == "t2v_combo"
                            _vace_lora_on = _plan["mode"] == "vace_useronly"
                            if getattr(runner, "_lora_key", "__init__") != _plan["want_key"]:
                                _sl = _plan["set_lora"]
                                if _sl["dir"] is None:
                                    runner.set_lora(None)
                                else:
                                    from basi import combo as _combo
                                    progress(0.04, desc=("building style LoRA…"
                                             if _vace_lora_on
                                             else "building LoRA combo…"))
                                    _b = _plan["build"]; _od = Path(_b["out_dir"])
                                    if not (_od / "low_noise_model.safetensors").exists():
                                        if _b["kind"] == "user_only":
                                            _combo.build_user_only(
                                                _b["user_high"], _b["user_low"], _od,
                                                user_strength=_b["user_strength"])
                                        else:
                                            _combo.build_combo(
                                                _b["lightning_dir"], _b["user_high"],
                                                _b["user_low"], _od,
                                                user_strength=_b["user_strength"],
                                                lightning_strength=_b["lightning_strength"])
                                    runner.set_lora(str(_sl["dir"]),
                                                    **{k: v for k, v in _sl.items()
                                                       if k != "dir"})
                                runner._lora_key = _plan["want_key"]
                        except Exception as _le:
                            return None, (f"### LoRA load failed\n\n```\n{_le}\n```")
                        # Build per-request args (W2: lora_strength must be
                        # in the dict, else worker uses 1.0 baked at startup
                        # and the UI slider does nothing). When a combo is
                        # active the user strength is baked in → pin 1.0.
                        _req_args = {
                            "prompt": prompt.strip(),
                            "width": width, "height": height,
                            "frames": int(frames), "steps": int(steps),
                            "guide": float(guide), "seed": int(seed),
                            # Strength policy is decided in _plan_user_lora:
                            # T2V combo bakes+pins 1.0; VACE user-only (un-pinned)
                            # and plain T2V honor the slider.
                            "lora_strength": _plan["request_strength"],
                            "out":  str(out_pt), "meta": str(out_meta),
                        }
                        if continue_image:
                            _req_args["image"] = str(continue_image)
                        # S2V talking character: reference image + driving audio.
                        # frames (slider) = infer_frames per chunk; the worker enforces
                        # the S2V_RECIPE (40 steps / guide 4.5 / shift 3.0) regardless
                        # of the steps/guide passed, and uses width*height as max area.
                        if s2v_audio:
                            _req_args["audio"] = str(s2v_audio)
                            _req_args["image"] = str(s2v_ref)
                        # SDEdit restyle: pass the uploaded source +
                        # denoise to the worker (T2V path; mutually exclusive
                        # with continue_image). denoise>=1.0 is plain T2V.
                        # Force 8-step mid-entry (NOT the 4-step T2V slider): at
                        # 4 steps the SDEdit tail barely crosses t>=875, starving
                        # the high-noise expert that sets global style. 8-step +
                        # L0.7 (combo above) is the measured optimum — style_sim
                        # 0.812 vs the 4-step corner-cut 0.750, AND better
                        # structure.
                        if _restyle_active:
                            _req_args["video"] = str(restyle_video)
                            _req_args["denoise_strength"] = float(denoise_strength)
                            _req_args["steps"] = 8
                            # Any-length restyle: when the upload is longer
                            # than one window (`frames`), tile into overlapping
                            # windows — the worker injects each window's leading
                            # latents from the previous styled tail (continuity)
                            # and LAB color-matches the seam, then stitches. Probe
                            # the frame count cheaply (packet timestamps, no decode).
                            try:
                                import torchvision as _tv
                                _pts, _ = _tv.io.read_video_timestamps(
                                    str(restyle_video), pts_unit="sec")
                                _nframes = len(_pts)
                            except Exception:
                                _nframes = 0
                            if _nframes > int(frames):
                                _req_args["sliding"] = True
                        # Depth-lock (VACE) restyle: send the source + the
                        # vace_depth flag; the worker extracts depth and ENFORCES
                        # the 50-step / guide-5.0 / shift-3.0 recipe (steps/guide
                        # here are advisory — the worker overrides from
                        # basi.vace.VACE_DEPTH_RECIPE so the 0.973 depth-lock
                        # settings can't drift). Full denoise (no SDEdit denoise).
                        elif _vace_restyle:
                            _req_args["video"] = str(restyle_video)
                            _req_args["vace_depth"] = True
                            _req_args["steps"] = 50
                            _req_args["guide"] = 5.0
                        # Keyframe-anchored editing: send the anchor frames
                        # + positions; the worker builds the VACE guide+mask and
                        # HARD-LOCKS the anchor latents (near-perfect preservation).
                        # Same 50-step/guide-5.0 recipe.
                        elif _kf_edit:
                            _req_args["vace_edit"] = True
                            _req_args["anchor_images"] = _kf_paths
                            _req_args["anchor_positions"] = _kf_pos
                            _req_args["steps"] = 50
                            _req_args["guide"] = 5.0
                        import uuid as _uuid
                        _req_id = str(_uuid.uuid4())
                        # Pre-flight ETA from the EMA cache (or the formula
                        # until it warms). Uses the EFFECTIVE steps/guide in
                        # _req_args (restyle=8, VACE=50 already applied above), so
                        # the estimate matches what actually runs. Source is
                        # labelled — a formula guess is never shown as measured.
                        try:
                            import torch as _t_eta
                            _gpu = (_t_eta.cuda.get_device_name(0)
                                    if _t_eta.cuda.is_available() else "cpu")
                        except Exception:
                            _gpu = "cpu"
                        _eta_s, _eta_src = _eta.estimate(
                            _gpu, _req_args["width"], _req_args["height"],
                            _req_args["frames"], _req_args["steps"], _req_args["guide"])
                        _eta_label = (f"~{_eta.human(_eta_s)} "
                                      f"{'(measured)' if _eta_src == 'measured' else '(estimate)'}")
                        progress(0.05, desc=f"starting — {_eta_label}")
                        terminal = None
                        try:
                            _gen_t0 = time.time()
                            for ev in runner.generate(_req_id, _req_args):
                                kind = ev.get("event")
                                if kind == "phase":
                                    nm = ev.get("name", "")
                                    if nm == "weights_page_in":
                                        # First gen after a model swap:
                                        # the ~19 GB pack pages in off disk.
                                        # Show it instead of a frozen bar.
                                        progress(
                                            0.05,
                                            desc=("paging in model weights "
                                                  "(first run after model switch) "
                                                  f"{int(ev.get('elapsed_s', 0))}s"))
                                    elif nm == "t5_encode":
                                        progress(0.10, desc="T5 encode")
                                    elif nm == "step":
                                        # Live per-step progress — the step
                                        # loop is 75% of the bar. Show
                                        # elapsed so a 20-step guide>1 run
                                        # reads as working, not hung.
                                        _i = ev.get("i", 0)
                                        _n = max(ev.get("n", 1), 1)
                                        # Live remaining-time countdown,
                                        # projected from steps done so far — a real
                                        # ETA, not just elapsed.
                                        _rem = _eta.remaining_live(
                                            _i, _n, time.time() - _gen_t0)
                                        progress(
                                            0.10 + 0.75 * (_i / _n),
                                            desc=(f"diffusion step {_i + 1}/{_n} "
                                                  f"— {_eta.human(_rem)} left"))
                                    elif nm == "step_loop":
                                        progress(0.85, desc="diffusion done")
                                    elif nm == "vae_decode":
                                        progress(0.95, desc="VAE decode")
                                elif kind == "started":
                                    progress(0.08, desc="generation started")
                                elif kind == "result":
                                    terminal = ev
                                elif kind == "error":
                                    terminal = ev
                                    break
                        except _RD as e:
                            tail = "\n".join(runner.get_human_tail(40))
                            return None, (
                                f"### Worker died mid-request: {e}\n\n```\n{tail}\n```"
                            )
                        if terminal is None or terminal.get("event") == "error":
                            ek = (terminal or {}).get("kind", "UNKNOWN")
                            em = (terminal or {}).get("msg", "no result")
                            tail = "\n".join(runner.get_human_tail(40))
                            return None, (
                                f"### Worker {ek}: {em}\n\n```\n{tail}\n```"
                            )
                        # .pt + meta written by worker; fall through to MP4
                        # encoder below.
                    else:
                        # --- LEGACY: subprocess.Popen per click ---
                        progress(0.02, desc="launching runner (legacy mode)")
                        proc = subprocess.Popen(
                            cli, cwd=str(BASI_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env=env,
                        )
                        for line in proc.stdout:  # type: ignore
                            log_tail.append(line.rstrip())
                            if len(log_tail) > 300:
                                log_tail = log_tail[-300:]
                            import re
                            m = re.search(r"(\d+)/(\d+)\s*\[", line)
                            if m:
                                cur, tot = int(m.group(1)), int(m.group(2))
                                progress(0.10 + 0.85 * (cur / max(tot, 1)),
                                         desc=f"step {cur}/{tot}")
                            elif "pipe ready" in line:
                                progress(0.08, desc="pipe ready — running steps")
                            elif "generated in" in line:
                                progress(0.95, desc="encoding mp4")
                        proc.wait()
                        if proc.returncode != 0:
                            return None, (
                                f"### Runner exit {proc.returncode}\n\n```\n"
                                + "\n".join(log_tail[-30:]) + "\n```"
                            )
                    # Runner saves .pt — convert to mp4 via imageio.
                    if not out_pt.exists():
                        return None, (
                            "### Runner exited 0 but no .pt produced\n\n```\n"
                            + "\n".join(log_tail[-20:]) + "\n```"
                        )
                    try:
                        import torch
                        import imageio
                        import numpy as _np
                        video = torch.load(str(out_pt))
                        if video.dim() == 4:
                            video_uint = ((video + 1.0) * 127.5).clamp(0, 255).byte()
                            video_uint = video_uint.permute(1, 2, 3, 0).numpy()
                        else:
                            video_uint = ((video[0] + 1.0) * 127.5).clamp(0, 255).byte()
                            video_uint = video_uint.permute(1, 2, 3, 0).numpy()

                        # Continuation stitch: the new clip's frame 0
                        # IS the conditioning frame (== a tail frame of the
                        # previous clip), so drop it before appending —
                        # otherwise the seam holds for 2 frames.
                        _prev_mp4 = _last_gen.get("mp4") if continue_image else None
                        if _prev_mp4 and Path(_prev_mp4).exists():
                            _prev = imageio.mimread(_prev_mp4, memtest=False)
                            _chain = _np.stack([*_prev, *video_uint[1:]])
                            out_mp4 = out_dir / f"studio_{stamp}_chain.mp4"
                            imageio.mimwrite(
                                str(out_mp4), _chain, fps=16,
                                codec="libx264", quality=8, pixelformat="yuv420p",
                            )
                        else:
                            imageio.mimwrite(
                                str(out_mp4), video_uint, fps=16,
                                codec="libx264", quality=8, pixelformat="yuv420p",
                            )

                        # Save the last 5 frames for Continue, ranked
                        # sharpest-first by Laplacian variance — the exact
                        # last frame is often motion-blurred (the FinalFrame
                        # 5-frame-picker insight). Pure numpy, no cv2 dep.
                        def _sharpness(fr):
                            g = fr.astype(_np.float32).mean(axis=2)
                            lap = (g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1]
                                   + g[:-2, 1:-1] - 4 * g[1:-1, 1:-1])
                            return float(lap.var())
                        _tail_src = (_chain if (_prev_mp4 and Path(_prev_mp4).exists())
                                     else video_uint)
                        _tail = list(_tail_src[-5:])
                        _ranked = sorted(range(len(_tail)),
                                         key=lambda i: _sharpness(_tail[i]),
                                         reverse=True)
                        _tail_pngs = []
                        for rank, i in enumerate(_ranked):
                            p = out_dir / f"studio_{stamp}_tail{rank}.png"
                            imageio.imwrite(str(p), _tail[i])
                            _tail_pngs.append(str(p))
                        _last_gen.clear()
                        _last_gen.update({
                            "tail_pngs": _tail_pngs,
                            "mp4": str(out_mp4),
                            "prompt": prompt.strip(),
                            "resolution": resolution,
                        })
                        out_pt.unlink(missing_ok=True)
                    except Exception as exc:
                        return None, (
                            f"### MP4 encoding failed: {exc}\n\n```\n"
                            + "\n".join(log_tail[-15:]) + "\n```"
                        )

                    import json as _json
                    try:
                        meta_data = _json.loads(out_meta.read_text())
                        wall = meta_data.get("wall_s", "?")
                    except Exception:
                        wall = "?"
                    # Fold this run's MEASURED wall into the EMA cache so the
                    # next same-shape estimate is exact. Guarded: _req_args/_gpu
                    # only exist on the worker path; wall may be "?" on failure.
                    if _use_worker and isinstance(wall, (int, float)):
                        try:
                            _eta.record(_gpu, _req_args["width"], _req_args["height"],
                                        _req_args["frames"], _req_args["steps"],
                                        _req_args["guide"], float(wall))
                        except Exception:
                            pass
                    progress(1.0, desc="done")
                    return str(out_mp4), (
                        f"### Generated in {wall}s\n\n"
                        f"**Shape**: {resolution} × {frames}f × {steps} steps  \n"
                        f"**Output**: `{out_mp4.name}`"
                    )

                def _refresh_tail_gallery():
                    pngs = _last_gen.get("tail_pngs") or []
                    return gr.update(value=pngs), 0

                def _pick_tail(evt: gr.SelectData):
                    return evt.index

                def _suggest_continue(direction_text, progress=gr.Progress()):
                    if not _last_gen.get("tail_pngs"):
                        return gr.update(), gr.update(), gr.update()
                    # Honest progress: if the weights aren't cached yet (the
                    # launch-time prefetch didn't finish or was disabled),
                    # this click pays an ~8 GB download first.
                    try:
                        from huggingface_hub import snapshot_download
                        from basi.caption import MODEL_TIERS
                        try:
                            snapshot_download(MODEL_TIERS[12], local_files_only=True)
                            progress(0.1, desc="loading Qwen3-VL-4B")
                        except Exception:
                            progress(0.05, desc="downloading Qwen3-VL-4B "
                                                 "(~8 GB, one time)")
                            snapshot_download(MODEL_TIERS[12])
                            progress(0.4, desc="loading Qwen3-VL-4B")
                    except Exception:
                        pass  # fall through; from_pretrained handles it
                    try:
                        # The VLM sees all 5 tail frames, picks
                        # the best continuation point AND writes the prompt
                        # in one call — it judges mid-blink/motion-smear/
                        # pose continuability, which the Laplacian ranking
                        # can't. Its pick drives the actual generation
                        # (continue_pick state) unless the user re-picks in
                        # the gallery afterwards.
                        from basi.caption import suggest_continuation_with_pick
                        idx, suggestion, why = suggest_continuation_with_pick(
                            _last_gen["tail_pngs"],
                            _last_gen.get("prompt", ""),
                            user_direction=direction_text or "")
                        note = (f"🖼️ VLM picked **frame {idx + 1}** of "
                                f"{len(_last_gen['tail_pngs'])}"
                                + (f" — {why}" if why else "")
                                + ". Click a different gallery frame to override.")
                        return suggestion, idx, note
                    except Exception as e:
                        return f"(suggestion failed: {e})", gr.update(), gr.update()

                def _studio_continue(c_prompt, pick_idx, frames, steps, seed,
                                     guide, lora, lora_strength,
                                     progress=gr.Progress()):
                    if not _last_gen.get("tail_pngs"):
                        return None, ("### Nothing to continue\n\nGenerate a "
                                      "video first.")
                    pngs = _last_gen["tail_pngs"]
                    png = pngs[min(int(pick_idx or 0), len(pngs) - 1)]
                    # Resolution is pinned to the source clip — stitching
                    # mixed resolutions would fail at np.stack, and the
                    # conditioning frame defines the aspect anyway.
                    res = _last_gen["resolution"]
                    c_prompt = (c_prompt or "").strip() or _last_gen.get("prompt", "")
                    return _studio_generate(
                        c_prompt, res, frames, steps, seed, guide,
                        lora, lora_strength, continue_image=png,
                        progress=progress)

                # Mode selector -> reveal only the active mode's controls.
                # T2V keeps the top-level Generate button; each other mode uses its
                # own button inside its (now-revealed) accordion.
                def _select_studio_mode(mode):
                    # The shared top prompt box is only meaningful for text-driven Wan modes.
                    # S2V is audio-driven (no text prompt); MOVA has its OWN prompt box — so hide
                    # the shared prompt for both. The Advanced accordion (resolution/frames/steps/
                    # guide/seed/Wan-LoRA) is used by every Wan mode but NONE of it applies to MOVA
                    # (MOVA has its own res/frames/steps/LoRA) — hide it for MOVA only.
                    _txt = mode in ("Text → Video", "Restyle", "Continue", "Keyframe")
                    return [gr.update(visible=mode == "Restyle"),
                            gr.update(visible=mode == "Continue"),
                            gr.update(visible=mode == "Keyframe"),
                            gr.update(visible=mode == "Talking character"),
                            gr.update(visible=mode == "Joint A/V (MOVA)"),
                            # the shared Generate button drives Text→Video only; Restyle /
                            # Continue / Keyframe / S2V / MOVA each have their own button.
                            gr.update(visible=mode == "Text → Video"),
                            gr.update(visible=_txt),                        # studio_prompt
                            gr.update(visible=mode != "Joint A/V (MOVA)")]  # Advanced accordion
                studio_mode.change(
                    _select_studio_mode, inputs=[studio_mode],
                    outputs=[restyle_acc, continue_acc, kf_acc, s2v_acc, mova_acc,
                             studio_generate_btn, studio_prompt, adv_acc])

                studio_generate_btn.click(
                    _studio_generate,
                    inputs=[studio_prompt, studio_resolution, studio_frames,
                            studio_steps, studio_seed, studio_guide,
                            studio_lora, studio_lora_strength],
                    outputs=[studio_video, studio_status],
                ).then(_refresh_tail_gallery,
                       outputs=[continue_gallery, continue_pick])

                # Restyle button → same handler, with the source video +
                # denoise threaded in (continue_image=None via state).
                restyle_btn.click(
                    _studio_generate,
                    inputs=[studio_prompt, studio_resolution, studio_frames,
                            studio_steps, studio_seed, studio_guide,
                            studio_lora, studio_lora_strength,
                            _restyle_no_continue, restyle_video, restyle_denoise,
                            restyle_mode],  # SDEdit vs depth-lock (VACE)
                    outputs=[studio_video, studio_status],
                )

                # Keyframe edit → same handler. continue_image + restyle_video
                # are both None (via the shared no-continue state); restyle_denoise /
                # restyle_mode are positional fillers the kf path ignores; the anchor
                # files + positions arrive in the trailing kf_images / kf_positions.
                kf_btn.click(
                    _studio_generate,
                    inputs=[studio_prompt, studio_resolution, studio_frames,
                            studio_steps, studio_seed, studio_guide,
                            studio_lora, studio_lora_strength,
                            _restyle_no_continue, _restyle_no_continue,
                            restyle_denoise, restyle_mode,
                            kf_images, kf_positions],
                    outputs=[studio_video, studio_status],
                )

                # Talking character (S2V) → same handler. continue_image /
                # restyle_video / kf_images / kf_positions are all None (via the
                # shared no-continue state) so the s2v branch fires exclusively;
                # denoise / mode are positional fillers the s2v path ignores; the
                # driving audio + reference image arrive in the trailing two slots.
                s2v_btn.click(
                    _studio_generate,
                    inputs=[studio_prompt, studio_resolution, studio_frames,
                            studio_steps, studio_seed, studio_guide,
                            studio_lora, studio_lora_strength,
                            _restyle_no_continue, _restyle_no_continue,
                            restyle_denoise, restyle_mode,
                            _restyle_no_continue, _restyle_no_continue,
                            s2v_audio, s2v_ref],
                    outputs=[studio_video, s2v_status],
                    api_name="generate_s2v",  # unambiguous endpoint (4 btns share fn)
                )

                # Joint A/V (MOVA): text -> audio+video. Spawns tools/mova_sample.py in the
                # env_mova venv (cross-platform, no bash/WSL) and streams it; the resulting mp4
                # (with audio) plays in the shared Studio output. Separate from _studio_generate
                # (different model, different venv, one-off subprocess, not the GGUF worker).
                def _mova_generate(prompt, lora_label, ref_path, trigger, res, frames, steps, cfg):
                    from basi import mova_infer as _mi
                    if not (prompt or "").strip():
                        yield None, "Enter a prompt — MOVA generates audio+video from text."
                        return
                    caps = _mi.mova_caps(detect_vram_gb(), _mi.detect_ram_gb())
                    if not caps["available"]:
                        yield None, f"### MOVA unavailable\n\n{caps['note']}"
                        return
                    if not _mi.mova_installed():
                        yield None, ("### MOVA not installed\n\nRe-run Install — it provisions "
                                     "the env_mova venv and the MOVA-360p weights.")
                        return
                    # Free the GGUF Studio (Wan) worker first — it holds ~9GB VRAM, and MOVA's
                    # group-offload onload needs ~12GB free or it CUDA-OOMs during model load.
                    # MOVA runs in its own venv/process, so the Wan worker isn't needed concurrently.
                    yield None, "🧹 Freeing GPU (closing the Wan video worker) for MOVA…"
                    _shutdown_worker_if_idle()
                    lora_dir = dict(_mi.list_mova_loras()).get(lora_label, "") or None
                    out_dir = WORKSPACES / "_studio" / "mova"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stamp = f"mova_{int(time.time())}"
                    # MOVA is I2V: the reference frame sets the SCENE + STYLE. An uploaded reference
                    # always wins. Otherwise (the general case) we MANUFACTURE a per-prompt reference:
                    # SDXL + IP-Adapter style transfer takes the LoRA's in-style frame (ref.png) +
                    # this prompt -> a styled, scene-matched still -> MOVA animates it. This is the
                    # T2AV per-prompt style reference (IP-Adapter scale 0.6). It runs in env_mova (same
                    # env as MOVA: SDXL+IP via diffusers). If it can't run (SDXL absent / failure) we
                    # fall back to the LoRA's static ref.png but WARN that the scene won't match the
                    # prompt -- never a silent scene-lock.
                    _ref = (ref_path or "").strip() or None
                    _style_src = _mi.mova_lora_style_ref(lora_dir) if lora_dir else None
                    if lora_dir and not _ref and _style_src:
                        try:
                            yield None, ("🎨 Painting a prompt-matched reference in your LoRA's style "
                                         "(SDXL + IP-Adapter)…")
                            _sref = _mi.make_style_reference(
                                prompt=prompt.strip(), style_ref=_style_src,
                                out_dir=str(out_dir / f"styleref_{stamp}"),
                                python=_mi.resolve_mova_python(), env=_mi.mova_spawn_env(),
                                trigger=(trigger or "").strip())
                            if _sref and Path(_sref).exists():
                                _ref = _sref
                            else:
                                _ref = _style_src
                                yield None, ("⚠️ SDXL not provisioned — using the LoRA's default "
                                             "reference; the SCENE may not match your prompt (style "
                                             "will be correct). Re-run Install to enable prompt-matched "
                                             "scenes, or upload your own reference.")
                        except Exception as _e:
                            print(f"[basiwan] style-reference maker failed ({_e}); using bundled ref",
                                  flush=True)
                            _ref = _style_src
                            yield None, ("⚠️ Couldn't paint a prompt-matched reference; using the "
                                         "LoRA's default reference (scene may not match the prompt).")
                    if lora_dir and not _ref and not _style_src:
                        yield None, ("⚠️ No reference image for this LoRA — MOVA is image-to-video, "
                                     "and without an in-style reference the style is lost (photoreal "
                                     "output). Upload a reference frame in your LoRA's style.")
                        return
                    # Spawn (or reuse) the resident MOVA worker, then send a 'start' request: it
                    # generates clip 0 AND seeds the continuation session (the worker holds the running
                    # clip + ranked tail frames for the Continue UI below).
                    yield None, "⏳ Loading MOVA (resident; first run ~minutes)…"
                    try:
                        runner = _ensure_mova_worker(lora_dir=lora_dir)
                    except Exception as _e:
                        yield None, f"### MOVA worker failed to start\n\n```\n{_e}\n```"
                        return
                    import uuid as _uuid, re as _re
                    _rm = _re.match(r"\s*(\d+)x(\d+)", res or "")
                    _w, _h = (int(_rm.group(1)), int(_rm.group(2))) if _rm else (320, 240)
                    _args = {"mode": "start", "prompt": prompt.strip(), "ref": _ref,
                             "out_dir": str(out_dir), "out_tag": stamp, "height": _h, "width": _w,
                             "frames": int(frames), "steps": int(steps), "cfg": float(cfg)}
                    yield None, "🎬 Generating MOVA A/V… (resident model)"
                    outpath, tails = None, []
                    try:
                        for ev in runner.generate(_uuid.uuid4().hex, _args):
                            _e = ev.get("event")
                            if _e == "result":
                                outpath = ev.get("out_mp4"); tails = ev.get("tail_pngs") or []
                            elif _e == "error":
                                yield None, f"### MOVA error: {ev.get('kind')}\n\n{ev.get('msg', '')}"
                                return
                    except Exception as _e:
                        _t = "\n".join(runner.get_human_tail(25)) if runner else str(_e)
                        yield None, f"### MOVA worker died\n\n```\n{_t}\n```"
                        return
                    if outpath and Path(outpath).exists():
                        _mova_last.clear()
                        _mova_last.update({"mp4": outpath, "tail_pngs": tails, "prompt": prompt.strip()})
                        yield outpath, ("✅ **MOVA A/V generated.** Use **➕ Continue** below to extend "
                                        "it with a new beat.")
                    else:
                        yield None, "### MOVA generation produced no output file."

                def _mova_refresh_gallery():
                    return gr.update(value=_mova_last.get("tail_pngs") or []), 0

                def _mova_pick_tail(evt: gr.SelectData):
                    return evt.index

                def _mova_suggest(direction_text, progress=gr.Progress()):
                    if not _mova_last.get("tail_pngs"):
                        return gr.update(), gr.update(), "Generate a MOVA clip first."
                    try:
                        from basi.caption import suggest_continuation_with_pick
                        idx, suggestion, why = suggest_continuation_with_pick(
                            _mova_last["tail_pngs"], _mova_last.get("prompt", ""),
                            user_direction=direction_text or "")
                        note = (f"🖼️ picked **frame {idx + 1}** of {len(_mova_last['tail_pngs'])}"
                                + (f" — {why}" if why else "")
                                + ". Click a different frame to override.")
                        return suggestion, idx, note
                    except Exception as _e:
                        return gr.update(), gr.update(), f"(suggestion failed: {_e})"

                def _mova_continue(c_prompt, pick_idx, frames, steps, cfg, progress=gr.Progress()):
                    if not _mova_last.get("tail_pngs"):
                        yield None, "### Nothing to continue — generate a MOVA clip first."
                        return
                    if not (c_prompt or "").strip():
                        yield None, "Enter a continuation prompt (the next beat)."
                        return
                    runner = _mova_worker_holder.get("runner")
                    if runner is None or not runner.is_alive():
                        yield None, ("### MOVA worker not running\n\nGenerate a clip first — that loads "
                                     "the resident model the Continue step reuses.")
                        return
                    out_dir = WORKSPACES / "_studio" / "mova"
                    stamp = f"mova_{int(time.time())}"
                    import uuid as _uuid
                    _args = {"mode": "continue", "prompt": c_prompt.strip(),
                             "pick_index": int(pick_idx or 0), "out_dir": str(out_dir),
                             "out_tag": stamp, "frames": int(frames), "steps": int(steps),
                             "cfg": float(cfg)}
                    yield None, "🎬 Extending from the chosen frame… (resident model)"
                    outpath, tails = None, []
                    try:
                        for ev in runner.generate(_uuid.uuid4().hex, _args):
                            _e = ev.get("event")
                            if _e == "result":
                                outpath = ev.get("out_mp4"); tails = ev.get("tail_pngs") or []
                            elif _e == "error":
                                yield None, f"### Continue error: {ev.get('kind')}\n\n{ev.get('msg', '')}"
                                return
                    except Exception as _e:
                        _t = "\n".join(runner.get_human_tail(25)) if runner else str(_e)
                        yield None, f"### MOVA worker died\n\n```\n{_t}\n```"
                        return
                    if outpath and Path(outpath).exists():
                        _mova_last.update({"mp4": outpath, "tail_pngs": tails, "prompt": c_prompt.strip()})
                        yield outpath, "✅ Extended. Continue again for another beat, or you're done."
                    else:
                        yield None, "### Continue produced no output file."

                mova_btn.click(
                    _mova_generate,
                    inputs=[mova_prompt, mova_lora, mova_ref, mova_trigger, mova_res, mova_frames, mova_steps, mova_cfg],
                    outputs=[studio_video, mova_status],
                    api_name="generate_mova",
                ).then(_mova_refresh_gallery, outputs=[mova_cont_gallery, mova_cont_pick])
                mova_cont_gallery.select(_mova_pick_tail, outputs=[mova_cont_pick])
                mova_cont_suggest.click(
                    _mova_suggest, inputs=[mova_cont_prompt],
                    outputs=[mova_cont_prompt, mova_cont_pick, mova_cont_status])
                mova_cont_btn.click(
                    _mova_continue,
                    inputs=[mova_cont_prompt, mova_cont_pick, mova_frames, mova_steps, mova_cfg],
                    outputs=[studio_video, mova_cont_status],
                ).then(_mova_refresh_gallery, outputs=[mova_cont_gallery, mova_cont_pick])

                # LLM 'magic rewrite': reshape the user's rough idea into the correct
                # MOVA prompt format (trigger + visual sentence + verbatim spoken-words clause)
                # using the same Qwen VLM as Suggest/caption. Rewrites the shared prompt box.
                def _mova_rewrite(prompt, trigger, progress=gr.Progress()):
                    if not (prompt or "").strip():
                        return gr.update(), "Type a rough idea first, then format it."
                    try:
                        progress(0.3, desc="formatting prompt (Qwen)")
                        from basi.caption import mova_format_prompt
                        out = mova_format_prompt(prompt, trigger=trigger)
                        return out, "✨ Formatted for MOVA — review, then Generate A/V."
                    except Exception as e:
                        return gr.update(), f"Rewrite failed: {type(e).__name__}: {e}"
                mova_rewrite_btn.click(
                    _mova_rewrite, inputs=[mova_prompt, mova_trigger],
                    outputs=[mova_prompt, mova_status],
                )

                continue_gallery.select(_pick_tail, outputs=[continue_pick])
                continue_suggest_btn.click(
                    _suggest_continue,
                    inputs=[continue_prompt],
                    outputs=[continue_prompt, continue_pick, continue_status],
                )
                continue_btn.click(
                    _studio_continue,
                    inputs=[continue_prompt, continue_pick, studio_frames,
                            studio_steps, studio_seed, studio_guide,
                            studio_lora, studio_lora_strength],
                    outputs=[studio_video, studio_status],
                ).then(_refresh_tail_gallery,
                       outputs=[continue_gallery, continue_pick])

            with gr.Tab("Gym"):
                with gr.Row():
                    # COLUMN 1: Configure
                    with gr.Column():
                        gr.Markdown("## 1 · Configure")
                        # Training type: Wan video-only LoRA (musubi) vs MOVA joint
                        # audio+video LoRA. The shared fields below (name, dataset, trigger,
                        # epochs, sample prompts) apply to both; the MOVA path routes to the
                        # validated basi/mova_train (240p NF4, per-epoch A/V samples) and
                        # needs .mp4 clips WITH embedded audio.
                        gym_train_type = gr.Radio(
                            ["Wan video LoRA", "MOVA audio+video LoRA"],
                            value="Wan video LoRA", label="Training type",
                            info="MOVA = joint audio+video LoRA (dataset clips need embedded audio).")
                        # pre-flight estimate (s/step, total time, VRAM, desktop
                        # headroom) -> only shown for MOVA; updates as preset/frames/epochs/
                        # repeats/dataset change. Measured-on-4090 basis (see mova_train.py).
                        mova_estimate = gr.Markdown(visible=False)
                        # live training analytics: parses this run's train.log into the
                        # convergence read-out (best/lowest-loss epoch, audio-overtrain point,
                        # sustained s/step, peak VRAM) so users SEE the data and make their own
                        # stop call. Complements the live loss_chart (the curve) with the
                        # interpretation. MOVA-only; refreshed on demand.
                        mova_analytics = gr.Markdown(visible=False)
                        mova_analytics_btn = gr.Button("↻ Training analytics",
                                                       size="sm", visible=False)
                        lora_name = gr.Textbox(
                            label="LoRA name",
                            value="my_first_lora",
                            info="Letters, digits, underscore, hyphen only.",
                        )
                        lora_name_warning = gr.Markdown(visible=False)
                        trigger_word = gr.Textbox(
                            label="Trigger word",
                            placeholder="e.g. lily_catgirl",
                            info=(
                                "Underscore_compound form (lily_catgirl, not 'ohwx'). "
                                "Auto-prepended to every caption. Use it at inference to "
                                "activate the LoRA. Leave empty for style/concept LoRAs."
                            ),
                        )
                        preset_dd = gr.Dropdown(
                            label="VRAM preset",
                            choices=list(PRESETS.keys()),
                            value=initial_preset,
                            info=(
                                "Picks rank (dim), alpha, lr, blocks_to_swap, fp8 for your GPU tier. "
                                "24g default: dim=32, alpha=16, lr=2e-4 (kohya canonical for Wan2.2)."
                            ),
                        )
                        expert_dd = gr.Dropdown(
                            label="Expert(s) to train",
                            choices=["low", "high", "both"],
                            value="low",
                            info=(
                                "Wan2.2 has two ~14B experts: high_noise (steps 875-1000, "
                                "composition/silhouette) and low_noise (steps 0-875, texture). "
                                "Character LoRAs ideally train 'both' (one .safetensors output) — "
                                "requires --offload_inactive_dit + ~32GB system RAM. "
                                "'low' is safest single-expert; covers texture/refinement well."
                            ),
                        )
                        # Wan2.2 only supports 8 specific resolution buckets — see
                        # basi/dataset.py SUPPORTED_RESOLUTIONS. Free sliders would let
                        # users pick unsupported sizes and musubi silently snaps,
                        # wasting training time. Dropdown of valid pairs enforces.
                        # Resolution choices differ by training type: Wan2.2 has 8
                        # fixed buckets; MOVA A/V trains at 240p (4:3 or 16:9). The
                        # gym_train_type radio swaps these (see _on_train_type below).
                        # Values are "WxH" (width x height), parsed in gen_train_command.
                        _WAN_RES_CHOICES = [
                            ("480×832 (portrait 480p)", "480x832"),
                            ("832×480 (landscape 480p)", "832x480"),
                            ("704×1024 (portrait 720p)", "704x1024"),
                            ("1024×704 (landscape 720p)", "1024x704"),
                            ("704×1280 (portrait 720p wide)", "704x1280"),
                            ("1280×704 (landscape 720p wide)", "1280x704"),
                            ("720×1280 (portrait 720p std)", "720x1280"),
                            ("1280×720 (landscape 720p std)", "1280x720"),
                        ]
                        # MOVA training res (measured, user-confirmed): 240p all tiers; ~360p on
                        # 16GB+; 480p training sits at the 24GB edge and works on LINUX with a clean
                        # GPU (not Windows). 360 isn't %16, so ~360p buckets use 352 lines.
                        _MOVA_RES_CHOICES = [
                            ("320×240 (4:3, 240p)", "320x240"),
                            ("416×240 (16:9, 240p)", "416x240"),
                        ]
                        _mv = detect_vram_gb()
                        if _mv >= 16:
                            _MOVA_RES_CHOICES += [
                                ("464×352 (4:3, ~360p)", "464x352"),
                                ("624×352 (16:9, ~360p)", "624x352"),
                            ]
                        if _mv >= 24 and sys.platform.startswith("linux"):
                            _MOVA_RES_CHOICES += [
                                ("640×480 (4:3, 480p — Linux/clean GPU)", "640x480"),
                                ("832×480 (16:9, 480p — Linux/clean GPU)", "832x480"),
                            ]
                        resolution_choice = gr.Dropdown(
                            _WAN_RES_CHOICES,
                            value="832x480",
                            label="Resolution (Wan2.2 supported buckets)",
                            info=(
                                "Train at 832×480; Wan2.2 generalizes to 1280×720 at "
                                "inference. Smaller=faster training, less VRAM. Off-bucket "
                                "values aren't supported by Wan2.2 so the list is fixed. "
                                "(MOVA A/V: 240p; ~360p on 16GB+; 480p on Linux+24GB clean GPU.)"
                            ),
                        )
                        # Default to 33f (safe everywhere except 8/12g — see preset
                        # max_target_frames). The actual cap per preset is enforced
                        # via the may-OOM warning below; we don't lock the slider so
                        # users with headroom can opt up.
                        target_frames = gr.Slider(
                            17, 81, value=GYM["frames"], step=4,
                            label="Frames (4n+1)",
                            info=(
                                "33f = 2s @ 16fps, the sweet spot for character motion. "
                                "Add a [1] auto for identity anchor. 17f saves VRAM "
                                "(12g/8g tiers). 81f only on 40g+."
                            ),
                        )
                        frame_warning = gr.Markdown(visible=False)
                        # T4.G: per-clip repeats — useful for small datasets.
                        num_repeats = gr.Slider(
                            1, 20, value=GYM["repeats"], step=1,
                            label="Num repeats per clip per epoch",
                            info=(
                                "Multiplier per clip per epoch. Rule of thumb: "
                                "round(50/clip_count), cap 10. With ~20 clips set "
                                "this to ~3 so each epoch sees ~50 samples."
                            ),
                        )
                        # T4.A: optional resume from a saved state dir.
                        resume_state = gr.Textbox(
                            label="Resume from state (optional)",
                            placeholder="path to musubi-tuner --save_state directory",
                            info="Leave empty for fresh training. Set --save_state on previous run to save state.",
                        )
                        # T4.H: advanced optimizer args (hidden by default).
                        with gr.Accordion("Advanced optimizer", open=False) as adv_opt_acc:
                            # Defaults match the Wan2.2 LoRA research brief —
                            # all still user-editable.
                            adv_lr_scheduler = gr.Dropdown(
                                ["", "constant_with_warmup", "cosine",
                                 "cosine_with_restarts", "linear",
                                 "polynomial", "adafactor"],
                                value="cosine_with_restarts",
                                label="LR scheduler (empty = musubi default)",
                                info=(
                                    "cosine_with_restarts prevents late-epoch overfit "
                                    "by cycling lr — pairs well with save_every=2 + "
                                    "16 epochs so each checkpoint sits at a different "
                                    "lr phase."
                                ),
                            )
                            adv_lr_warmup = gr.Number(
                                value=GYM["warmup_steps"], label="Warmup steps", precision=0,
                                info=(
                                    "Linear ramp from 0 to lr over N steps to stabilize "
                                    "first-epoch grads. 100 is modest; scale up for "
                                    "datasets >500 clips."
                                ),
                            )
                            adv_grad_clip = gr.Number(
                                value=None, label="Max grad norm (0 = disable)",
                                info=(
                                    "Clamps gradient norm before optimizer step. "
                                    "Leave empty (musubi default 1.0) unless training "
                                    "diverges — then try 0.5."
                                ),
                            )
                            adv_weight_decay = gr.Number(
                                value=None, label="Weight decay",
                                info=(
                                    "L2 penalty on LoRA weights. Leave empty (musubi "
                                    "default 0.01) — only tune if overfitting persists "
                                    "after lr drops."
                                ),
                            )
                        # max_epochs default 16 per research (8 evaluable
                        # checkpoints at save_every=2). User-editable.
                        max_epochs = gr.Slider(
                            1, 100, value=GYM["epochs"], step=1, label="Max train epochs",
                            info=(
                                "First-attempt sweet spot. With save_every=2 you get "
                                "8 evaluable checkpoints. Overfit usually shows at "
                                "ep 10-12; pick an earlier checkpoint if samples "
                                "start looking copied."
                            ),
                        )
                        sample_every = gr.Slider(
                            0, 1000, value=0, step=50,
                            label="Extra samples every N steps (0 = per-epoch only)",
                            info=(
                                "A sample renders automatically at the start AND "
                                "after every epoch — that's the primary signal, "
                                "one render per checkpoint. Set this only for "
                                "additional mid-epoch samples on large datasets "
                                "(each render pauses training ~1 min)."
                            ),
                        )
                        sample_prompt = gr.Textbox(
                            label="Sample prompt",
                            value="a cat walking through a sunlit garden, cinematic",
                            lines=2,
                            info=(
                                "Used for during-training samples. Include your "
                                "trigger word to verify activation. Add a second line "
                                "without the trigger to detect identity bleed."
                            ),
                        )
                        # MOVA-only: the visual STYLE descriptor for the prompt-driven A/V reference
                        # maker (SDXL+IP-Adapter). 3-5 words naming the medium+look. Authoritative
                        # over the VLM's auto-suggestion (small VLMs mis-ID material, e.g. clay as
                        # "2D cel"). Blank -> the Gym auto-suggests one at train time. Ignored by Wan.
                        mova_style_descriptor = gr.Textbox(
                            label="Style descriptor (MOVA A/V — for prompt-driven references)",
                            value="", placeholder="claymation, stop-motion clay figures",
                            info="3-5 words: the show's medium + look. Used to render per-prompt "
                                 "reference frames in your style. Blank = auto-detect at train time.")
                        # MOVA A/V sample prompts need the trigger + visual + verbatim
                        # spoken-words shape. One-click LLM rewrite (uses the trigger field
                        # above). Only meaningful for MOVA A/V training; harmless otherwise.
                        with gr.Row():
                            gym_mova_rewrite_btn = gr.Button(
                                "✨ Format for MOVA A/V", variant="secondary", scale=2)
                            gym_mova_rewrite_status = gr.Markdown(scale=3)
                        gym_mova_rewrite_btn.click(
                            _mova_rewrite, inputs=[sample_prompt, trigger_word],
                            outputs=[sample_prompt, gym_mova_rewrite_status])

                    # COLUMN 2: Dataset
                    with gr.Column():
                        gr.Markdown("## 2 · Dataset")
                        gr.Markdown("Drop `.mp4` / `.mov` clips. Captions: `.txt` next to each clip "
                                    "(use **Auto-caption** below to generate).")
                        upload = gr.File(file_count="multiple", label="Videos (+ optional .txt captions)")
                        with gr.Row():
                            ingest_btn = gr.Button("Ingest + Scan", variant="primary")
                            caption_btn = gr.Button("Auto-caption (Qwen-VL)", variant="secondary")
                        # Long-source path: scene-detect a big video into 2-5s
                        # single-shot clips (the dataset recipe shape) instead of
                        # asking users to cut by hand.
                        with gr.Accordion("Auto-split long video into clips", open=False):
                            gr.Markdown(
                                "For source material that isn't clip-sized "
                                "yet: detects shot cuts, extracts **2-5s "
                                "single-shot clips** (long shots are sampled "
                                "at spread-out moments, not consecutively), "
                                "drops sub-2s fragments, and **flags** "
                                "blurry clips for your review — nothing is "
                                "auto-deleted. Clips land in the dataset and "
                                "are scanned like any upload."
                            )
                            autosplit_threshold = gr.Slider(
                                0.2, 0.6, value=0.4, step=0.05,
                                label="Cut sensitivity",
                                info="ffmpeg scene-score threshold. 0.4 = "
                                     "standard for hard cuts. Lower catches "
                                     "subtler cuts but false-triggers on "
                                     "fast motion; gradual fades are missed "
                                     "at any setting (they make poor "
                                     "training clips anyway).",
                            )
                            autosplit_max = gr.Slider(
                                10, 80, value=60, step=5,
                                label="Max clips per source video",
                                info="Plan is evenly subsampled across the "
                                     "whole video if over this — 20-50 clips "
                                     "is the training sweet spot, >80 is "
                                     "diminishing returns.",
                            )
                            autosplit_btn = gr.Button(
                                "Auto-split uploaded video(s)", variant="secondary")
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
                        # One button: prepare configs, precompute the latent/text
                        # caches, then train — sequential, with all logs streaming
                        # below and a Stop that works at any stage. The manual
                        # per-step buttons live in the Advanced accordion for
                        # debugging.
                        train_pipeline_btn = gr.Button(
                            "🚀 Train LoRA", variant="primary", size="lg")
                        stop_train_btn = gr.Button("⏹ Stop", variant="stop")
                        scripts_md = gr.Markdown(label="Scripts")
                        cache_script_box = gr.Textbox(label="cache.sh path", visible=False)
                        train_script_box = gr.Textbox(label="train.sh path", visible=False)
                        logs = gr.Textbox(label="Live logs", lines=20, max_lines=20, autoscroll=True)
                        stop_status = gr.Markdown()
                        with gr.Accordion("Advanced: run steps manually", open=False):
                            gr.Markdown(
                                "The Train button runs these three steps for "
                                "you. Use these only to re-run a single stage "
                                "(e.g. re-cache after editing captions without "
                                "retraining from scratch)."
                            )
                            with gr.Row():
                                gen_btn = gr.Button("1 · Generate scripts", variant="secondary")
                                run_cache_btn = gr.Button("2 · Run cache (T5+VAE)", variant="secondary")
                                run_train_btn = gr.Button("3 · Run training", variant="secondary")
                            stop_cache_btn = gr.Button("Stop cache", variant="stop")
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

                        _wan_preview_hdr = gr.Markdown(
                            "### Preview (Faster-Wan2.2 Lightning + FP8 + TAEHV)\n"
                            "~3-4 s at 832×480×17 frames. Trainer must be paused — single "
                            "4090 cannot run trainer + preview concurrently.")
                        with gr.Row():
                            preview_size = gr.Dropdown(["832*480", "480*832", "720*1280", "1280*720"],
                                                       value="832*480", label="Size",
                                                       info="Resolution of the quick test render (not training). Smaller = faster.")
                            preview_frames = gr.Slider(5, 81, value=GYM["preview_frames"], step=4, label="Frames (4n+1)",
                                                       info="Length of the test render (4n+1). Fewer = faster preview.")
                            preview_steps = gr.Slider(2, 8, value=GYM["preview_steps"], step=1, label="Lightning steps",
                                                      info="Denoise steps for the preview (Lightning). 4 is plenty for a quick look.")
                            preview_seed = gr.Number(value=GYM["preview_seed"], label="Seed", precision=0,
                                                     info="Fixed so previews across epochs are comparable; change for a different test scene.")
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
                                info="LoRA file layout: comfyui (ComfyUI), peft / diffusers "
                                     "(HF / diffusers loaders), musubi (raw kohya/musubi keys).",
                            )
                            export_btn = gr.Button("Export latest LoRA", variant="secondary")
                        export_status = gr.Markdown()
                        preview_btn = gr.Button("Generate preview from latest LoRA", variant="secondary")
                        preview_video = gr.Video(label="Preview")
                        preview_status = gr.Markdown()

                # Wiring
                def _autosplit_uploaded(files, workspace_name, threshold,
                                        max_total, progress=gr.Progress()):
                    """Scene-split uploaded long video(s) into the dataset
                    dir, then rescan via the normal ingest report path."""
                    if not workspace_name or not files:
                        return "no files / no workspace", ""
                    from basi.autosplit import autosplit_video, VIDEO_EXTS
                    ds_dir = WORKSPACES / workspace_name / "dataset"
                    ds_dir.mkdir(parents=True, exist_ok=True)
                    lines = ["# Auto-split report\n"]
                    for f in files:
                        src = Path(f.name if hasattr(f, "name") else f)
                        if not src.exists() or src.suffix.lower() not in VIDEO_EXTS:
                            continue
                        try:
                            rep = autosplit_video(
                                src, ds_dir,
                                threshold=float(threshold),
                                max_total=int(max_total),
                                progress_cb=progress)
                        except Exception as e:
                            lines.append(f"- ❌ `{src.name}`: {e}")
                            continue
                        # If the raw source was previously Ingest+Scan'd
                        # into the dataset by mistake, remove it — the
                        # clips replace it; a 20-min AVI in the dataset
                        # would poison captioning and training.
                        _stale = ds_dir / src.name
                        if _stale.exists():
                            _stale.unlink()
                            (_stale.with_suffix(".txt")).unlink(missing_ok=True)
                            lines.append(f"  - removed previously-ingested "
                                         f"raw source `{src.name}` from the "
                                         f"dataset (replaced by its clips)")
                        lines.append(
                            f"- `{src.name}` ({rep.duration_s:.0f}s, "
                            f"{rep.n_shots} shots) → **{len(rep.clips)} "
                            f"clips**"
                            + (f", {len(rep.skipped_shots)} shots <2s skipped"
                               if rep.skipped_shots else "")
                            + (f", {rep.dropped_for_cap} clips dropped "
                               f"for the max-clips cap"
                               if rep.dropped_for_cap else ""))
                        for p, why in rep.flagged.items():
                            lines.append(f"  - ⚠️ `{Path(p).name}`: {why}")
                    # Rescan so the report + bucket distribution match a
                    # normal ingest exactly.
                    from basi.dataset import scan_dataset, bucket_distribution
                    clips = scan_dataset(str(ds_dir))
                    lines.append(f"\n# Dataset scan ({len(clips)} clips)\n")
                    for c in clips:
                        cap = "✓ caption" if c.caption_text else "✗ NO CAPTION"
                        issues = f" — {', '.join(c.issues)}" if c.issues else ""
                        lines.append(
                            f"- `{Path(c.path).name}` {c.width}×{c.height} "
                            f"@{c.fps:.1f}fps × {c.frame_count}f [{cap}]{issues}")
                    buckets = bucket_distribution(clips)
                    lines.append(f"\n## Bucket distribution ({len(buckets)} buckets)\n")
                    for (res, frames), count in sorted(buckets.items(), key=lambda x: -x[1]):
                        lines.append(f"- {res[0]}×{res[1]} × {frames}f: **{count} clips**")
                    return "\n".join(lines), str(ds_dir)

                autosplit_btn.click(_autosplit_uploaded,
                                    inputs=[upload, lora_name,
                                            autosplit_threshold, autosplit_max],
                                    outputs=[dataset_report, dataset_dir_box])

                ingest_btn.click(_ingest_uploaded_videos,
                                 inputs=[upload, lora_name],
                                 outputs=[dataset_report, dataset_dir_box])
                def _caption_with_gpu_room(dataset_dir, trigger, train_type=""):
                    # The 24G-tier captioner (Qwen3-VL-8B bf16) needs
                    # ~17 GB VRAM; the idle Studio worker holds ~5-9 GB
                    # reserved. Free it first — the next Studio click
                    # pays a ~25s cached worker restart, vs an OOM here.
                    # Generator chain: first feedback lands in the report
                    # box within a second of the click.
                    # MOVA training type -> joint A/V prompt + ASR dialogue (the audio fix).
                    yield "🧹 freeing GPU (shutting down idle video worker)…"
                    try:
                        _shutdown_worker_if_idle()
                    except Exception:
                        pass
                    yield from _auto_caption(dataset_dir, trigger,
                                             mova_av_mode=("MOVA" in str(train_type)))

                caption_btn.click(_caption_with_gpu_room,
                                  inputs=[dataset_dir_box, trigger_word, gym_train_type],
                                  outputs=[caption_report])
                def _check_frame_cap(preset_key: str, frames: int):
                    # preset_key is a Wan key on the Wan tab, a mova_* key on the MOVA tab
                    # (separate table) -> resolve from either; unknown -> no warning.
                    p = PRESETS.get(preset_key) or MOVA_PRESETS.get(preset_key)
                    if p is None:
                        return gr.Markdown(visible=False)
                    cap = p.max_target_frames
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

                # Switching Wan <-> MOVA swaps the dependent fields so the visible
                # preset / resolution / expert match the chosen trainer (the MOVA branch in
                # _train_pipeline reads these -> what you see is what runs). MOVA has no
                # high/low expert choice, so that field is hidden for it.
                def _on_train_type(tt):
                    # All the Wan/musubi-only controls below are IGNORED by the MOVA trainer
                    # (gen_mova_train_command takes none of them), so hide them for MOVA and show
                    # them for Wan — otherwise a MOVA user sees musubi schedulers, a musubi
                    # --save_state resume box, a Wan step-sample slider, a Wan loss chart, and a
                    # Faster-Wan2.2 preview that can't render an A/V LoRA. (mova_analytics replaces
                    # the loss chart for MOVA.)
                    if "MOVA" in str(tt):
                        mp = auto_select_mova(detect_vram_gb())
                        pkey = next((k for k, v in MOVA_PRESETS.items() if v is mp),
                                    list(MOVA_PRESETS)[-1])
                        return (gr.update(choices=list(MOVA_PRESETS.keys()), value=pkey,
                                          label="MOVA preset (A/V, NF4)"),
                                gr.update(choices=_MOVA_RES_CHOICES, value="320x240",
                                          label="Resolution (MOVA)"),
                                gr.update(visible=False),   # expert_dd (Wan high/low)
                                gr.update(visible=True),    # mova_analytics_btn
                                # 81f@24fps = 3.4s is the proven MOVA len; widen the cap to the
                                # preset's max (129 on 40G+), which the Wan-shaped 81 slider blocked.
                                gr.update(value=min(81, mp.max_target_frames),
                                          maximum=mp.max_target_frames),
                                gr.update(visible=False),   # resume_state (musubi --save_state)
                                gr.update(visible=False),   # sample_every (Wan step-sampling)
                                gr.update(visible=False),   # adv_opt_acc (musubi optimizer)
                                gr.update(visible=False),   # loss_chart (Wan log format)
                                gr.update(visible=False),   # _wan_preview_hdr
                                gr.update(visible=False),   # preview_btn
                                gr.update(visible=False))   # preview_video
                    return (gr.update(choices=list(PRESETS.keys()), value=initial_preset,
                                      label="VRAM preset"),
                            gr.update(choices=_WAN_RES_CHOICES, value="832x480",
                                      label="Resolution (Wan2.2 supported buckets)"),
                            gr.update(visible=True),        # expert_dd
                            gr.update(visible=False),       # hide MOVA analytics btn on the Wan path
                            gr.update(value=GYM["frames"], maximum=81),  # Wan default + cap
                            gr.update(visible=True),        # resume_state
                            gr.update(visible=True),        # sample_every
                            gr.update(visible=True),        # adv_opt_acc
                            gr.update(visible=True),        # loss_chart
                            gr.update(visible=True),        # _wan_preview_hdr
                            gr.update(visible=True),        # preview_btn
                            gr.update(visible=True))        # preview_video
                gym_train_type.change(_on_train_type, inputs=[gym_train_type],
                                      outputs=[preset_dd, resolution_choice, expert_dd,
                                               mova_analytics_btn, target_frames,
                                               resume_state, sample_every, adv_opt_acc,
                                               loss_chart, _wan_preview_hdr, preview_btn,
                                               preview_video])

                # training analytics: parse outputs/<lora_name>/train.log -> convergence
                # read-out. On demand (cheap; reads a text file). Shows the best epoch + the
                # audio-overtrain warning + sustained s/step, so the user can stop at the right
                # epoch rather than guessing.
                def _mova_analytics(name):
                    log = OUTPUTS_ROOT / str(name or "") / "train.log"
                    md = format_mova_analytics_md(parse_mova_training_log(log))
                    return gr.update(value=md, visible=True)
                mova_analytics_btn.click(_mova_analytics, inputs=[lora_name],
                                         outputs=[mova_analytics])

                # live pre-flight estimate: s/step, total time, VRAM, desktop headroom.
                # Only for MOVA; recomputed when preset / frames / epochs / repeats / dataset
                # change. Clip count read from the dataset dir; numbers are 4090-measured.
                def _mova_est(tt, dataset_dir, preset_key, frames, epochs, repeats):
                    if not tt or "MOVA" not in str(tt):
                        return gr.update(visible=False)
                    n = 0
                    try:
                        if dataset_dir and Path(dataset_dir).is_dir():
                            n = len(list(Path(dataset_dir).glob("**/*.mp4")))
                    except Exception:
                        n = 0
                    mp = MOVA_PRESETS.get(preset_key) or auto_select_mova(detect_vram_gb())
                    fr = min(int(frames), mp.max_target_frames)
                    # auto_schedule (the MovaTrainConfig default) derives repeats/epochs from the
                    # clip count so any dataset size lands near the ~2400-step target -> show the
                    # values that will ACTUALLY run, not the (overridden) manual inputs. With no
                    # dataset yet (n==0) fall back to the manual inputs.
                    eff_epochs, eff_repeats, auto = int(epochs), int(repeats), False
                    if n > 0:
                        from basi.mova_data import plan_training_schedule
                        sch = plan_training_schedule(n)
                        eff_epochs, eff_repeats, auto = sch["epochs"], sch["repeats"], True
                    md = format_mova_estimate_md(n, fr, eff_epochs, eff_repeats,
                                                 mp.name, detect_vram_gb())
                    # MOVA dataset checklist — the requirements that aren't obvious + the measured
                    # audio lesson (dialogue-in-caption is the #1 fix; CER 0.0-0.06 with it, babble
                    # without). Shown on every MOVA estimate so users assemble the set correctly.
                    md = ("**MOVA dataset checklist** — clips need **synced audio** (auto-resampled "
                          "to 48 kHz mono, normalized to −18 LUFS); **≥3.4 s each (≥81f @ 24fps)** so "
                          "a full spoken line fits in-window; **Auto-caption auto-adds verbatim "
                          "dialogue** (required — without it generated speech is non-English babble); "
                          "judge generated samples at **≥81 frames** (shorter truncates the line). "
                          "Trigger word stays constant so it absorbs the style.\n\n" + md)
                    if auto:
                        md += (f"\n\n_Epochs/repeats auto-scaled to {n} clips "
                               f"(x{eff_repeats} repeats, {eff_epochs} epochs) so any dataset size "
                               "trains to a comparable target; the manual fields are overridden._")
                    return gr.update(value=md, visible=True)
                _mova_est_in = [gym_train_type, dataset_dir_box, preset_dd, target_frames,
                                max_epochs, num_repeats]
                for _c in (gym_train_type, preset_dd, target_frames, max_epochs,
                           num_repeats, dataset_dir_box):
                    _c.change(_mova_est, inputs=_mova_est_in, outputs=[mova_estimate])
                def _train_pipeline(gym_train_type, workspace_name, dataset_dir, preset_key,
                                    expert_choice, trigger, samp_prompt, style_descriptor,
                                    n_epochs, samp_every, t_frames, res_choice,
                                    n_repeats, res_state, lr_sched, lr_warm,
                                    grad_clip, w_decay):
                    """One-click train: prepare → cache → train, all logs
                    streaming. Yields (scripts_md, cache_path, train_path,
                    logs) so the Stop button and Advanced re-run buttons
                    keep working on the same script paths. The MOVA branch
                    routes to basi/mova_train (joint A/V); Wan is the default."""
                    def _out(md, c, t, log):
                        return md, c, t, log

                    # MOVA joint audio+video LoRA -> the basi/mova_train path
                    # (240p NF4, latent cache, per-epoch A/V samples). Reuses the
                    # shared name/dataset/trigger/epochs/sample fields + _spawn_script
                    # streaming + Stop button. Wan training falls through below.
                    if gym_train_type and "MOVA" in str(gym_train_type):
                        from basi.mova_train import (MovaTrainConfig,
                                                     prepare_mova_training_run)
                        # Use the UI-selected MOVA preset + resolution so what's shown is
                        # what runs (preset_key is a mova_* key on this tab; res is "WxH").
                        mp = MOVA_PRESETS.get(preset_key) or auto_select_mova(detect_vram_gb())
                        try:
                            _tw, _th = (int(s) for s in (res_choice or "320x240").split("x"))
                        except Exception:
                            _tw, _th = 320, 240
                        sp = [s.strip() for s in (samp_prompt or "").splitlines()
                              if s.strip()] or None
                        try:
                            mcfg = MovaTrainConfig(
                                lora_name=workspace_name, dataset_dir=dataset_dir, preset=mp,
                                target_resolution=(_th, _tw),
                                target_frames=min(int(t_frames), mp.max_target_frames),
                                max_train_epochs=int(n_epochs),
                                save_every_n_epochs=1,   # MOVA: sample/checkpoint every epoch (validated)
                                num_repeats=int(n_repeats), trigger_word=(trigger or None),
                                sample_prompts=sp,
                                style_descriptor=(style_descriptor or "").strip() or None)
                            plan = prepare_mova_training_run(mcfg)
                        except Exception as e:
                            yield _out(f"❌ MOVA prepare failed: {type(e).__name__}: {e}",
                                       "", "", "")
                            return
                        ts = plan["train_script"]
                        yield _out(f"**MOVA A/V** prepared ({plan['clips']} clips, {mp.name}); "
                                   "samples every epoch. Launching…", "", ts, "")
                        for chunk in _spawn_script(ts):
                            yield _out("**MOVA audio+video LoRA** — per-epoch A/V samples land "
                                       "in the output folder; safe to leave running.", "", ts, chunk)
                        yield _out("✅ **MOVA training done** — pick the best epoch from the "
                                   "samples.", "", ts, "")
                        return
                    if not workspace_name or not dataset_dir:
                        yield _out("**Add a dataset first** (column 2: upload "
                                   "clips or Auto-split a long video, then "
                                   "Auto-caption).", "", "", "")
                        return
                    # Caption sanity: training with uncaptioned clips wastes
                    # the run. Warn loudly but let the user proceed — the
                    # trigger word alone is a legitimate minimal caption
                    # strategy some users choose.
                    _vids = [p for p in Path(dataset_dir).iterdir()
                             if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}]
                    _uncap = [p for p in _vids if not p.with_suffix(".txt").exists()]
                    _warn = ""
                    if _uncap:
                        _warn = (f"⚠️ {len(_uncap)}/{len(_vids)} clips have NO "
                                 f"caption — run Auto-caption first for better "
                                 f"results. Continuing anyway.\n\n")
                    md, cache_script, train_script = _generate_scripts(
                        workspace_name, dataset_dir, preset_key, expert_choice,
                        trigger, samp_prompt, n_epochs, samp_every, t_frames,
                        res_choice, n_repeats, res_state, lr_sched, lr_warm,
                        grad_clip, w_decay)
                    if not cache_script:
                        yield _out(_warn + md, "", "", "")
                        return
                    yield _out(_warn + "**Step 1/3 ✓** configs prepared",
                               cache_script, train_script, "")
                    # Ensure the ~50GB Wan training base is present BEFORE caching
                    # (which reads T5+VAE from it). Auto-fetch if missing — idempotent
                    # + resumable; "ensure not missing", not "error if missing".
                    _vpy = os.environ.get("BASIWAN_VENV_PYTHON", sys.executable)
                    _ew_cmd = [_vpy, str(BASI_ROOT / "tools" / "ensure_weights.py"),
                               "--groups", "wan-train"]
                    _ew_last = ""
                    for chunk in _stream_cmd(_ew_cmd):
                        _ew_last = chunk
                        yield _out(_warn + "**Step 1b/3** ensuring Wan training base "
                                   "(~50GB; downloads once if missing, resumable)…",
                                   cache_script, train_script, chunk)
                    if "ENSURE_WEIGHTS_DONE" not in _ew_last:
                        yield _out(_warn + "❌ **Training base weights incomplete** — "
                                   "re-run to resume the download. Training not started.",
                                   cache_script, train_script, _ew_last)
                        return
                    # Free the Studio worker's RAM/VRAM before the heavy work.
                    try:
                        _shutdown_worker_if_idle()
                    except Exception:
                        pass
                    last = ""
                    for chunk in _spawn_script(cache_script):
                        last = chunk
                        yield _out(_warn + "**Step 2/3** precomputing latent + "
                                   "text caches (one pass over the dataset)…",
                                   cache_script, train_script, chunk)
                    _cache_ok = bool(last) and "[exit code 0]" in last.splitlines()[-1]
                    if not _cache_ok:
                        yield _out(_warn + "❌ **Cache step failed** — training "
                                   "not started. Logs below.",
                                   cache_script, train_script, last)
                        return
                    for chunk in _spawn_script(train_script):
                        yield _out(_warn + "**Step 3/3** training — loss chart "
                                   "and per-epoch samples update below; safe "
                                   "to leave running.",
                                   cache_script, train_script, chunk)
                    yield _out(_warn + "✅ **Done** — checkpoints in the output "
                               "folder; check the sample gallery to pick the "
                               "best epoch.", cache_script, train_script, last)

                train_pipeline_btn.click(
                    _train_pipeline,
                    inputs=[gym_train_type, lora_name, dataset_dir_box, preset_dd, expert_dd,
                            trigger_word, sample_prompt, mova_style_descriptor, max_epochs,
                            sample_every, target_frames, resolution_choice,
                            num_repeats, resume_state, adv_lr_scheduler,
                            adv_lr_warmup, adv_grad_clip, adv_weight_decay],
                    outputs=[scripts_md, cache_script_box, train_script_box, logs])

                gen_btn.click(_generate_scripts,
                              inputs=[lora_name, dataset_dir_box, preset_dd, expert_dd, trigger_word,
                                      sample_prompt, max_epochs, sample_every, target_frames,
                                      resolution_choice, num_repeats, resume_state,
                                      adv_lr_scheduler, adv_lr_warmup, adv_grad_clip,
                                      adv_weight_decay],
                              outputs=[scripts_md, cache_script_box, train_script_box])
                run_cache_btn.click(_spawn_script, inputs=[cache_script_box], outputs=[logs])

                def _spawn_train(script_path):
                    # Training needs the host RAM the idle Studio worker is
                    # holding (~20 GB of packed weights). Shut it down first;
                    # the next Studio Generate pays a cached cold start
                    # (~1-2 min), not the full re-pack.
                    try:
                        _shutdown_worker_if_idle()
                    except Exception:
                        pass
                    return _spawn_script(script_path)

                run_train_btn.click(_spawn_train, inputs=[train_script_box], outputs=[logs])
                def _stop_any(cache_p, train_p):
                    # The one-click pipeline's Stop must halt whichever
                    # stage is live (cache OR train).
                    msgs = []
                    for p in (train_p, cache_p):
                        if p:
                            r = _stop_script(p)
                            if "not found" not in r:
                                msgs.append(r)
                    return "; ".join(msgs) or "nothing running"

                stop_train_btn.click(_stop_any,
                                     inputs=[cache_script_box, train_script_box],
                                     outputs=[stop_status])
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
            with gr.Tab("LoRA Library"):
                # LoRA LIBRARY tab.
                gr.Markdown(
                    "### LoRA Library\n\n"
                    "Browse trained LoRAs in `outputs/` plus any community "
                    "LoRAs in standard ComfyUI / peer dirs. Click a LoRA to "
                    "set it as active for the Studio tab."
                )
                lora_refresh_btn = gr.Button("Refresh", size="sm")
                lora_list_md = gr.Markdown("(no LoRAs found yet - train one in the Gym tab)")

                def _scan_loras():
                    items = []
                    if WORKSPACES.exists():
                        for ws in sorted(WORKSPACES.iterdir()):
                            if not ws.is_dir():
                                continue
                            st = list(ws.glob("*.safetensors"))
                            if not st:
                                continue
                            items.append(f"- **{ws.name}** (trained) - `{st[-1].name}`")
                    for cand in COMFYUI_LORA_CANDIDATES:
                        if cand.exists():
                            for f in sorted(cand.glob("*.safetensors"))[:20]:
                                items.append(f"- `{f.name}` (peer: {cand})")
                    if not items:
                        return "(no LoRAs found - train one in the Gym tab)"
                    return "### Available LoRAs (" + str(len(items)) + ")\n" + "\n".join(items)

                lora_refresh_btn.click(_scan_loras, outputs=[lora_list_md])
                demo.load(_scan_loras, outputs=[lora_list_md])

    return demo


# Single-instance guard. The port probe below deliberately walks upward when
# 7860 is busy, so the OS port bind is NOT a cross-process lock — two app.py
# processes happily coexist on 7860/7861, and each eager-spawns a ~20 GB worker
# (host-RAM contention + Continue-hang amplifier). This lock file is the real
# guard. Semantics: REFUSE, never auto-kill. A second instance whose predecessor
# is verifiably alive exits cleanly with a message; a stale lock (dead PID or
# unresponsive port) is reclaimed.
_SINGLETON_LOCK_HELD = False
_SINGLETON_LOCK_PATH = None


def _singleton_lock_path() -> "Path":
    base = os.environ.get("BASIWAN_PACK_CACHE_DIR",
                          str(Path.home() / ".cache" / "marlin_packs"))
    d = Path(base).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d / "basiwan_app.lock"


def _pid_alive(pid: int) -> bool:
    """Cross-platform liveness. On Windows, OpenProcess + exit-code check;
    on POSIX, signal 0. Either way: True only if the PID is a running proc."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k = ctypes.windll.kernel32
        h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k.CloseHandle(h)
    else:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but not ours
        return True


def _app_responds(port: int) -> bool:
    """Identity check: a live BASIWAN serves Gradio's /config. Guards
    against PID reuse (a recycled PID that isn't our app)."""
    import urllib.request
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/config", timeout=2.0) as r:
            return r.status == 200
    except Exception:
        return False


def _acquire_singleton_lock() -> bool:
    """Return True if we now hold the lock; print + return False to signal
    the caller should exit if another live instance owns it."""
    global _SINGLETON_LOCK_HELD, _SINGLETON_LOCK_PATH
    if os.environ.get("BASIWAN_SINGLETON_GUARD", "1") != "1":
        _SINGLETON_LOCK_HELD = True  # guard disabled → behave as before
        return True
    import json
    lock = _singleton_lock_path()
    _SINGLETON_LOCK_PATH = lock
    if lock.exists():
        try:
            prior = json.loads(lock.read_text())
            ppid, pport = int(prior.get("pid", -1)), int(prior.get("port", -1))
        except Exception:
            ppid, pport = -1, -1
        if _pid_alive(ppid) and (pport <= 0 or _app_responds(pport)):
            url = f"http://127.0.0.1:{pport}" if pport > 0 else "(unknown port)"
            print(f"[basiwan] another BASIWAN app is already running at {url} "
                  f"(pid {ppid}). Use that window, or stop it first.",
                  flush=True)
            return False
        # Stale lock — predecessor dead or unresponsive. Reclaim.
        print(f"[basiwan] clearing stale lock (pid {ppid} not responding)",
              flush=True)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass
    _SINGLETON_LOCK_HELD = True
    return True  # port written in _write_singleton_lock once known


def _write_singleton_lock(port: int) -> None:
    """Record our pid+port atomically once the bound port is known, and
    register cleanup. atexit covers normal exit; SIGTERM/SIGINT cover
    Pinokio Stop. Hard kills (TerminateProcess) skip these — handled by the
    next start's stale-lock liveness check."""
    if not _SINGLETON_LOCK_HELD or _SINGLETON_LOCK_PATH is None:
        return
    import json, atexit, signal, time as _t
    payload = json.dumps({"pid": os.getpid(), "port": port,
                          "started": _t.strftime("%Y-%m-%dT%H:%M:%S")})
    tmp = _SINGLETON_LOCK_PATH.with_suffix(".lock.tmp")
    tmp.write_text(payload)
    tmp.replace(_SINGLETON_LOCK_PATH)

    def _release(*_a):
        try:
            if _SINGLETON_LOCK_PATH.exists():
                cur = json.loads(_SINGLETON_LOCK_PATH.read_text())
                if int(cur.get("pid", -1)) == os.getpid():
                    _SINGLETON_LOCK_PATH.unlink()
        except Exception:
            pass
    atexit.register(_release)
    for _sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if _sig is not None:
            try:
                _prev = signal.getsignal(_sig)
                def _handler(signum, frame, _prev=_prev):
                    _release()
                    if callable(_prev):
                        _prev(signum, frame)
                    raise SystemExit(0)
                signal.signal(_sig, _handler)
            except (ValueError, OSError):
                pass  # not main thread / unsupported — atexit still covers


if __name__ == "__main__":
    import socket
    if not _acquire_singleton_lock():
        sys.exit(1)
    # Port resolution: GRADIO_SERVER_PORT env > BASIWAN_PORT env > 7860.
    # Then probe upward up to +20 if busy (Gradio's launch(server_port=N)
    # ONLY tries that exact port, so we have to find one ourselves).
    _port_env = os.environ.get("GRADIO_SERVER_PORT") or os.environ.get("BASIWAN_PORT")
    _start_port = int(_port_env) if _port_env and _port_env.isdigit() else 7860

    def _find_free_port(start: int, max_tries: int = 20) -> int:
        for offset in range(max_tries):
            p = start + offset
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        raise RuntimeError(
            f"No free port in {start}-{start + max_tries - 1}. "
            "Set GRADIO_SERVER_PORT to a known-free port."
        )

    _port = _find_free_port(_start_port)
    if _port != _start_port:
        print(f"[basiwan] port {_start_port} busy; using {_port}")
    _write_singleton_lock(_port)
    build_ui().launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=_port,
        inbrowser=False,
    )
