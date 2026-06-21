"""Correct MOVA A/V sampler -- SEPARATE process, isolated from training (the Flux-Gym /
sd-scripts / musubi pattern: sample between epochs in a fresh process, never sharing or
mutating training state). Loads base (+ optional native LoRA) FRESH via the SHIPPED
MOVA / MOVALoRA inference (BOTH bf16 video experts, real high/low-noise boundary switch,
group/leaf CPU offload ~12GB to fit 24GB), 50 denoise steps -- exactly what end users get.

The runner frees the GPU at an epoch boundary and spawns this; a crash here only logs a
warning (training resumes). Usage:
  mova_sample.py --base <MOVA-360p> [--lora <checkpoint_dir>] --prompts <txt> --ref <png>
                 --out <dir> --tag <baseline|stepN|final> [--height --width --frames --steps --cfg --offload]
ASCII. For users this is launched by the app (Studio MOVA mode) / the Gym trainer via the env_mova
interpreter -- basi/mova_infer.resolve_mova_python() + mova_spawn_env(), which set the env_mova
activation so the right CUDA/MKL DLLs load (cross-platform; no WSL, no dev script needed).
"""
import os, sys, argparse
from pathlib import Path
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# MOVA torch.compile's its rotary/modulation path; Inductor needs Triton. Compile is a SPEED knob
# only -- it does NOT change the trained STYLE (the I2V reference frame carries the style; see the
# --ref handling below). So if Triton is absent we disable torch.compile and run EAGER, which works
# on any box and produces the same style. install_mova.js provides Triton (native on Linux,
# triton-windows on Windows) so the compiled fast path is the norm; this is just a safety net.
try:
    import triton  # noqa: F401
except Exception:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Import the `mova` package (pip-installed by the Pinokio install). Also add the vendored
# ext/mova clone to sys.path when present (covers its configs/ + scripts/training_scripts);
# override with MOVA_REPO for a dev checkout. No home/dev-box path.
_mova_dev = Path(os.environ.get("MOVA_REPO", str(Path(__file__).resolve().parent.parent / "ext" / "mova")))
if _mova_dev.exists():
    sys.path.insert(0, str(_mova_dev))
    sys.path.insert(0, str(_mova_dev / "scripts" / "training_scripts"))
import torch, torch.distributed as dist
from PIL import Image

DEFAULT_NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量"
def log(m): print(f"[mova-sample] {m}", flush=True)


def _generate_chain(pipe, prompt, ref0, a, sr):
    """FinalFrame continuation: chain a.clips clips IN-PROCESS into one (frames, audio). Each clip's
    sharpest last frame (LAB color-anchored to clip 0) seeds the next clip's I2V ref; audio is LUFS
    -normalized per clip + crossfaded ONE video-frame at each seam (declick + keeps A/V frame-synced).
    Bridge frame (clip k's frame 0 ~= clip k-1's last) dropped for k>=1. Helpers: tools/mova_continue.py."""
    import time, numpy as np, torch
    from PIL import Image
    from mova.datasets.transforms.custom import crop_and_resize
    import mova_continue as mc
    fps = 24
    _XF = 3   # seam crossfade frames (pixel blend at each join); audio crossfade matches -> A/V synced
    all_frames, all_waves, anchor, ref = [], [], None, ref0
    for k in range(a.clips):
        ck = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            video, audio = pipe(prompt=prompt, negative_prompt=DEFAULT_NEG, num_frames=a.frames,
                                image=ref, height=a.height, width=a.width, video_fps=float(fps),
                                num_inference_steps=a.steps, cfg_scale=a.cfg, seed=42, cp_mesh=None)
        frames = video[0]
        wave = audio[0].cpu().squeeze().numpy().astype("float32")
        arrs = [np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames]
        if k == 0:                                  # fixed anchor = clip-0 mean frame (research)
            mean_frame = np.mean(np.stack(arrs).astype(np.float64), axis=0).astype(np.uint8)
            anchor = mc.lab_stats(mean_frame)
        else:                                       # re-align this clip to the clip-0 anchor
            arrs = [mc.color_match(fr, anchor, strength=0.5) for fr in arrs]
            frames = [Image.fromarray(fr) for fr in arrs]
        all_frames = list(frames) if k == 0 else mc.crossfade_video_seam(all_frames, frames, _XF)
        all_waves.append(mc.lufs_normalize(wave, sr, -18.0))
        log(f"  clip {k + 1}/{a.clips} ({time.time() - ck:.0f}s)")
        if k < a.clips - 1:                         # next ref = sharpest tail frame (color-anchored),
            _, ref_pil = mc.sharpest_tail_frame(frames, tail=5)            # +mild noise to soften seam
            ref = crop_and_resize(mc.addnoise_pil(ref_pil, sigma=0.02), height=a.height, width=a.width)
    wave = mc.crossfade_concat(all_waves, sr, crossfade_ms=_XF * 1000.0 / fps)
    target = int(round(len(all_frames) * sr / fps))   # frame-exact A/V sync (pad/trim the seam slack)
    wave = wave[:target] if len(wave) > target else np.pad(wave, (0, target - len(wave)))
    return all_frames, torch.from_numpy(wave)


import json as _json


def _emit_event(event, **kw):
    """Emit a [BASIWAN-EVENT] JSON line — the EXACT prefix tools/runner_client.py parses, so the
    persistent MOVA worker reuses BasiwanRunner (the Wan client) unchanged."""
    print("[BASIWAN-EVENT] " + _json.dumps({"event": event, **kw}), flush=True)


def _prep_ref(path, height, width):
    """The I2V conditioning frame: a real in-style image, or a neutral gray frame as last resort."""
    from mova.datasets.transforms.custom import crop_and_resize
    if path and Path(path).exists():
        img = Image.open(path).convert("RGB")
    else:
        img = Image.new("RGB", (width, height), (128, 128, 128))
    return crop_and_resize(img, height=height, width=width)


def generate_one(pipe, prompt, ref_pil, *, frames, steps, cfg, height, width, seed, fps=24):
    """One MOVA forward. Returns (list[PIL] frames, np.float32 mono wave)."""
    import numpy as np
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        video, audio = pipe(prompt=prompt, negative_prompt=DEFAULT_NEG, num_frames=int(frames),
                            image=ref_pil, height=int(height), width=int(width), video_fps=float(fps),
                            num_inference_steps=int(steps), cfg_scale=float(cfg), seed=int(seed),
                            cp_mesh=None)
    return video[0], audio[0].cpu().squeeze().numpy().astype("float32")


def build_pipe(a):
    """Load MOVA (+optional native LoRA, +optional NF4-base emulation) and apply offload. ONE-TIME
    cost; the persistent --serve worker calls this once and reuses the pipe for every request."""
    # Single-GPU: MOVA's pipeline only queries dist rank/world (the tqdm gate) when cp_mesh=None;
    # the collective paths are multi-GPU only. Monkeypatch to single-rank instead of creating a PG
    # (avoids a gloo socket + a noisy-but-harmless Windows hostname-probe warning; NCCL is Linux-only).
    if not dist.is_initialized():
        dist.is_initialized = lambda: True
        dist.get_rank = lambda *a, **k: 0
        dist.get_world_size = lambda *a, **k: 1
        dist.barrier = lambda *a, **k: None
    torch.cuda.set_device(0)
    from mova.diffusion.pipelines import MOVA, MOVALoRA

    has_lora = bool(a.lora) and (Path(a.lora) / "lora_weights.pt").exists()
    # An NF4-trained LoRA MUST sample against an NF4 base or the style is lost; lora_config records it.
    if has_lora and not a.nf4_base:
        try:
            _cfg = torch.load(str(Path(a.lora) / "lora_config.pt"), map_location="cpu", weights_only=False)
            if str(_cfg.get("base", "")).lower() == "nf4":
                a.nf4_base = True
                log("[nf4-base] auto-enabled from lora_config (base=nf4) -- matching training precision")
        except Exception:
            pass
    log(f"build_pipe lora={'yes' if has_lora else 'NONE(base)'} nf4_base={a.nf4_base} offload={a.offload}")

    if has_lora:
        # lora_modules MUST be the TOWER names (video_dit./audio_dit./dual_tower_bridge.) — the trained
        # checkpoint's key prefixes. Falling back to the saved layer-name target_modules injects 0
        # layers (samples the base, not the LoRA). video_dit_2 is the untrained placeholder, excluded.
        pipe = MOVALoRA.from_pretrained_with_lora(
            pretrained_path=a.base, lora_path=a.lora, device="cpu", torch_dtype=torch.bfloat16,
            lora_modules=["video_dit", "audio_dit", "dual_tower_bridge"])
        if a.nf4_base:
            # EMULATE training's NF4 base in pure bf16: quant->dequant each base weight (carrying the
            # exact 4-bit error) and store it as bf16 -> math identical to training, inference stays
            # pure bf16 (Linear4bit-at-inference doesn't compose with the two-expert group offload).
            from bitsandbytes.functional import quantize_4bit, dequantize_4bit
            from mova.engine.trainer.accelerate.lora_utils import LoRALinear
            import torch.nn as nn

            def _nf4_emulate(root):
                n = 0
                for _, m in root.named_modules():
                    if isinstance(m, LoRALinear) and isinstance(m.original_layer, nn.Linear):
                        w = m.original_layer.weight
                        dev = w.device
                        wq, st = quantize_4bit(w.data.to("cuda", torch.bfloat16), quant_type="nf4")
                        m.original_layer.weight.data = dequantize_4bit(wq, st).to(dev, torch.bfloat16)
                        n += 1
                torch.cuda.empty_cache()
                return n
            nq = _nf4_emulate(pipe.video_dit.blocks)
            if getattr(pipe, "dual_tower_bridge", None) is not None:
                nq += _nf4_emulate(pipe.dual_tower_bridge)
            log(f"[nf4-base] NF4-emulated {nq} bases in bf16 (quant->dequant; matches training values)")
    else:
        pipe = MOVA.from_pretrained(a.base, torch_dtype=torch.bfloat16)

    if a.offload == "group":
        pipe.enable_group_offload(onload_device=torch.device("cuda", 0), offload_device=torch.device("cpu"),
                                  offload_type="leaf_level", use_stream=True, low_cpu_mem_usage=True)
        # Quiet diffusers' benign group-offload trace warning (conditional bridge layers don't run in
        # the lazy-prefetch trace pass but DO run in the real denoise — the validated path).
        import logging as _logging
        _logging.getLogger("diffusers.hooks.group_offloading").setLevel(_logging.ERROR)
    elif a.offload == "cpu":
        pipe.enable_model_cpu_offload(0)
    else:
        pipe.to(torch.device("cuda", 0))
    return pipe, has_lora


def _serve_loop(pipe, a):
    """Persistent MOVA worker (mirrors run_one_video_gguf.py --serve): load once, then serve requests
    over stdin (JSON lines) -> [BASIWAN-EVENT] stdout. Holds a continuation SESSION (accumulated
    frames + audio + clip-0 LAB color anchor) so 'continue' extends the running clip from a chosen
    tail frame with a NEW prompt — the per-segment-prompt continuation the app's UI drives."""
    import json, time, gc
    import numpy as np
    from PIL import Image
    from mova.utils.data import save_video_with_audio
    from mova.datasets.transforms.custom import crop_and_resize
    import mova_continue as mc
    sr = pipe.audio_sample_rate
    fps = 24
    XF = 3                                  # seam crossfade frames (video) == audio crossfade frames
    sess = {"frames": [], "wave": None, "anchor": None, "tail": [], "h": a.height, "w": a.width, "seg": 0}

    def _finalize_and_emit(out_dir, tag, rid, t0):
        target = int(round(len(sess["frames"]) * sr / fps))        # frame-exact A/V sync
        wv = sess["wave"]
        sess["wave"] = wv[:target] if len(wv) > target else np.pad(wv, (0, max(0, target - len(wv))))
        mp4 = Path(out_dir) / f"{tag}.mp4"
        save_video_with_audio(sess["frames"], torch.from_numpy(sess["wave"]), str(mp4),
                              fps=fps, sample_rate=sr, quality=5)
        ranked = mc.rank_tail_frames(sess["frames"], sess["wave"], sr, fps=fps, tail=8, k=5)
        sess["tail"] = [pil for _, pil in ranked]
        tails = []
        for j, (_, pil) in enumerate(ranked):
            p = Path(out_dir) / f"{tag}_tail{j}.png"
            pil.save(str(p)); tails.append(str(p.resolve()))
        _emit_event("result", id=rid, out_mp4=str(mp4.resolve()), tail_pngs=tails, auto_best=0,
                    n_frames=len(sess["frames"]), seg=sess["seg"], wall_s=round(time.time() - t0, 1), ok=True)

    _emit_event("ready", resident=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            _emit_event("error", id=None, kind="BAD_JSON", fatal=False); continue
        cmd, rid = req.get("cmd"), req.get("id")
        if cmd == "shutdown":
            _emit_event("bye", id=rid, ok=True); return 0
        if cmd == "ping":
            _emit_event("pong", id=rid, resident=True); continue
        if cmd != "generate":
            _emit_event("error", id=rid, kind="BAD_CMD", fatal=False); continue
        g = req.get("args", {})
        try:
            mode = g.get("mode", "start")
            prompt = g["prompt"]; out_dir = g["out_dir"]; tag = g["out_tag"]
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            frames_n = int(g.get("frames", a.frames)); steps = int(g.get("steps", a.steps))
            cfg = float(g.get("cfg", a.cfg))
            _emit_event("started", id=rid, mode=mode, prompt_first40=prompt[:40])
            t0 = time.time()
            if mode == "start":
                h, w = int(g.get("height", a.height)), int(g.get("width", a.width))
                sess.update({"h": h, "w": w, "seg": 0})
                seed = int(g.get("seed", 42))
                ref = _prep_ref(g.get("ref"), h, w)
                frames, wave = generate_one(pipe, prompt, ref, frames=frames_n, steps=steps,
                                            cfg=cfg, height=h, width=w, seed=seed, fps=fps)
                mean_frame = np.mean(np.stack([np.asarray(f.convert("RGB"), dtype=np.float64)
                                               for f in frames]), axis=0).astype(np.uint8)
                sess["anchor"] = mc.lab_stats(mean_frame)
                sess["frames"] = list(frames)
                sess["wave"] = mc.lufs_normalize(wave, sr, -18.0)
            else:                                                  # continue
                if not sess["tail"]:
                    _emit_event("error", id=rid, kind="NO_SESSION",
                                msg="continue requested with no prior clip", fatal=False); continue
                sess["seg"] += 1
                seed = int(g.get("seed", 42 + sess["seg"]))        # vary noise per segment
                pick = max(0, min(int(g.get("pick_index", 0)), len(sess["tail"]) - 1))
                ref = crop_and_resize(mc.addnoise_pil(sess["tail"][pick], sigma=0.02),
                                      height=sess["h"], width=sess["w"])
                frames, wave = generate_one(pipe, prompt, ref, frames=frames_n, steps=steps,
                                            cfg=cfg, height=sess["h"], width=sess["w"], seed=seed, fps=fps)
                arrs = [mc.color_match(np.asarray(f.convert("RGB"), dtype=np.uint8), sess["anchor"], strength=0.5)
                        for f in frames]
                frames = [Image.fromarray(x) for x in arrs]
                sess["frames"] = mc.crossfade_video_seam(sess["frames"], frames, XF)
                seg_wave = mc.lufs_normalize(wave, sr, -18.0)
                sess["wave"] = mc.equal_power_crossfade_concat([sess["wave"], seg_wave], sr,
                                                               crossfade_ms=XF * 1000.0 / fps)
            _finalize_and_emit(out_dir, tag, rid, t0)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache(); gc.collect()
            _emit_event("error", id=rid, kind="CUDA_OOM", msg=str(e)[:200], fatal=False)
        except Exception as e:
            _emit_event("error", id=rid, kind="RUNTIME", msg=str(e)[:300], fatal=False)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--lora", default=None)
    # Prompt source: a file (one per line, trainer usage) OR a single inline --prompt
    # (interactive/app usage). Exactly one is needed.
    ap.add_argument("--prompts", default=None); ap.add_argument("--prompt", default=None)
    # --ref is OPTIONAL: MOVA's shipped path is TEXT->A/V (T2AV). With no ref we synthesize a
    # neutral mid-gray first frame — the exact no-conditioning input the trainer/sampler used
    # (its ref is a solid gray frame), so inference conditioning == training. Supplying a real
    # image is an advanced I2V control (untrained on neutral-frame LoRAs; off by default).
    ap.add_argument("--ref", default=None)
    ap.add_argument("--out", default=None); ap.add_argument("--tag", default="sample")
    ap.add_argument("--height", type=int, default=240); ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--frames", type=int, default=49); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=5.0); ap.add_argument("--offload", default="group")
    # --clips N: FinalFrame-style continuation. N>1 chains N clips IN-PROCESS (model resident) into
    # ONE longer A/V: each clip's sharpest last frame (color-anchored to clip 0) seeds the next clip's
    # I2V ref; audio is LUFS-normalized per clip + crossfaded one video-frame at each seam (keeps A/V
    # frame-synced + declicks). See tools/mova_continue.py. N=1 = single clip (unchanged).
    ap.add_argument("--clips", type=int, default=1)
    # --nf4-base: NF4-quantize video_dit.blocks + dual_tower_bridge AFTER loading the LoRA, to match
    # TRAINING (which trained the LoRA on top of NF4-quantized bases). Without it, train uses NF4
    # bases but inference uses bf16 -> the LoRA's learned deltas (partly NF4-error compensation) are
    # applied to a different base -> coherent-but-wrong output (style doesn't transfer). Test.
    ap.add_argument("--nf4-base", action="store_true")
    # --serve: persistent worker. Load the model ONCE, then serve generate/continue requests over
    # stdin (JSON) -> [BASIWAN-EVENT] stdout (BasiwanRunner protocol). The app uses this so each
    # Studio MOVA generation + every interactive Continue reuses the resident model. No --out needed.
    ap.add_argument("--serve", action="store_true")
    a = ap.parse_args()
    # Load the model once (the expensive step). Reused for every request in --serve mode.
    pipe, has_lora = build_pipe(a)

    if a.serve:
        sys.exit(_serve_loop(pipe, a))

    # One-shot path: between-epoch trainer sampling + legacy CLI. Inline --prompt wins; else file.
    if not a.out:
        log("ERROR: --out is required without --serve"); sys.exit(2)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from mova.utils.data import save_video_with_audio
    if a.prompt:
        prompts = [a.prompt.strip()]
    elif a.prompts:
        prompts = [l.strip() for l in Path(a.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        log("ERROR: provide --prompt <text> or --prompts <file>"); sys.exit(2)
    import time
    SR = pipe.audio_sample_rate
    # I2V conditioning frame: a real in-style --ref (carries the trained style), else a neutral gray
    # last-resort (a STYLE LoRA loses its style without one — warn).
    ref = _prep_ref(a.ref, a.height, a.width)
    if a.ref and Path(a.ref).exists():
        log(f"ref: image '{Path(a.ref).name}' (I2V anchor)")
    else:
        log("ref: WARNING no --ref -> neutral gray frame; a STYLE LoRA will lose its style. "
            "Pass --ref an in-style frame for correct style.")
    log(f"tag={a.tag} lora={'yes' if has_lora else 'NONE(base)'} prompts={len(prompts)} "
        f"{a.height}x{a.width}x{a.frames}f steps={a.steps}")
    for i, p in enumerate(prompts):
        t0 = time.time()
        if a.clips and a.clips > 1:
            frames, wave = _generate_chain(pipe, p, ref, a, SR)   # FinalFrame continuation -> 1 A/V
        else:
            frames, wave = generate_one(pipe, p, ref, frames=a.frames, steps=a.steps,
                                        cfg=a.cfg, height=a.height, width=a.width, seed=42)
            wave = torch.from_numpy(wave)
        _mp4 = out / f"{a.tag}_p{i}.mp4"
        save_video_with_audio(frames, wave, str(_mp4), fps=24, sample_rate=SR, quality=5)
        log(f"{a.tag}_p{i}.mp4 done ({a.clips} clip(s), {time.time()-t0:.0f}s)")
        # Machine-readable: the app parses this to locate the produced A/V file (abs path).
        print(f"MOVA_SAMPLE_OUT|{_mp4.resolve()}", flush=True)
    log("MOVA_SAMPLE_DONE")


if __name__ == "__main__":
    main()
