"""Correct MOVA A/V sampler -- SEPARATE process, isolated from training (the Flux-Gym /
sd-scripts / musubi pattern: sample between epochs in a fresh process, never sharing or
mutating training state). Loads base (+ optional native LoRA) FRESH via the SHIPPED
MOVA / MOVALoRA inference (BOTH bf16 video experts, real high/low-noise boundary switch,
group/leaf CPU offload ~12GB to fit 24GB), 50 denoise steps -- exactly what end users get.

The runner frees the GPU at an epoch boundary and spawns this; a crash here only logs a
warning (training resumes). Usage:
  mova_sample.py --base <MOVA-360p> [--lora <checkpoint_dir>] --prompts <txt> --ref <png>
                 --out <dir> --tag <baseline|stepN|final> [--height --width --frames --steps --cfg --offload]
ASCII. Launch via ~/mova_m1/run.sh (sets the CUDA-13 LD_LIBRARY_PATH).
"""
import os, sys, argparse
from pathlib import Path
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# MOVA torch.compile's the rotary-embedding path; inductor needs Triton, which has no standard
# Windows wheels. Disable torch.compile when Triton is absent -> run eager (works on all OSes).
# The compile gain is negligible here anyway: group-offload PCIe paging dominates the step time.
try:
    import triton  # noqa: F401
except Exception:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
# Import the `mova` package. For end users it is pip-installed (ext/mova vendored by the
# Pinokio install), so nothing extra is needed. On the dev box it lives at ~/MOVA — add that
# to sys.path ONLY if present, so we never inject a bogus path on a normal install.
_mova_dev = Path.home() / "MOVA"
if _mova_dev.exists():
    sys.path.insert(0, str(_mova_dev))
    sys.path.insert(0, str(_mova_dev / "scripts" / "training_scripts"))
import torch, torch.distributed as dist
from PIL import Image

DEFAULT_NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量"
def log(m): print(f"[mova-sample] {m}", flush=True)


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
    ap.add_argument("--out", required=True); ap.add_argument("--tag", default="sample")
    ap.add_argument("--height", type=int, default=240); ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--frames", type=int, default=49); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--cfg", type=float, default=5.0); ap.add_argument("--offload", default="group")
    # --nf4-base: NF4-quantize video_dit.blocks + dual_tower_bridge AFTER loading the LoRA, to match
    # TRAINING (which trained the LoRA on top of NF4-quantized bases). Without it, train uses NF4
    # bases but inference uses bf16 -> the LoRA's learned deltas (partly NF4-error compensation) are
    # applied to a different base -> coherent-but-wrong output (style doesn't transfer). Test 2026-06-17.
    ap.add_argument("--nf4-base", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # Single-GPU: MOVA's pipeline only calls dist.get_rank()/get_world_size() (the tqdm gate)
    # when cp_mesh=None — the all_gather/_sp_* collective paths are multi-GPU only and never run
    # here. So instead of creating a real process group, monkeypatch the query fns to single-rank.
    # Creating a gloo store on Windows emits a noisy but HARMLESS hostname-probe warning
    # ("...kubernetes.docker.internal:29560 ... not valid in its context") and NCCL is Linux-only;
    # skipping the PG entirely means no socket and no warning, and is fully sufficient for inference.
    if not dist.is_initialized():
        dist.is_initialized = lambda: True
        dist.get_rank = lambda *a, **k: 0
        dist.get_world_size = lambda *a, **k: 1
        dist.barrier = lambda *a, **k: None
    torch.cuda.set_device(0)

    from mova.diffusion.pipelines import MOVA, MOVALoRA
    from mova.utils.data import save_video_with_audio
    from mova.datasets.transforms.custom import crop_and_resize

    # Prompts: inline --prompt wins; else the --prompts file. One is required.
    if a.prompt:
        prompts = [a.prompt.strip()]
    elif a.prompts:
        prompts = [l.strip() for l in Path(a.prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        log("ERROR: provide --prompt <text> or --prompts <file>"); sys.exit(2)
    has_lora = bool(a.lora) and (Path(a.lora) / "lora_weights.pt").exists()
    # Auto-match training precision: an NF4-trained LoRA MUST be sampled against an NF4 base or the
    # style is lost (measured 2026-06-17). lora_config records base=="nf4" -> enable emulation.
    if has_lora and not a.nf4_base:
        try:
            _cfg = torch.load(str(Path(a.lora) / "lora_config.pt"), map_location="cpu", weights_only=False)
            if str(_cfg.get("base", "")).lower() == "nf4":
                a.nf4_base = True
                log("[nf4-base] auto-enabled from lora_config (base=nf4) -- matching training precision")
        except Exception:
            pass
    log(f"tag={a.tag} lora={'yes' if has_lora else 'NONE(base)'} nf4_base={a.nf4_base} "
        f"prompts={len(prompts)} {a.height}x{a.width}x{a.frames}f steps={a.steps} offload={a.offload}")

    if has_lora:
        # lora_modules MUST be the TOWER names. Without it, MOVALoRA falls back to the saved
        # lora_config target_modules (['q','k','v','o'] -- the LAYER names our trainer used) and
        # _inject_lora_layers iterates THOSE as towers -> hasattr(self,'q') is False -> injects 0
        # LoRA layers -> samples are the BASE model, not the trained LoRA (bug found 2026-06-17).
        # These three towers match the trained checkpoint's key prefixes (video_dit./audio_dit./
        # dual_tower_bridge.); video_dit_2 is the untrained placeholder, intentionally excluded.
        pipe = MOVALoRA.from_pretrained_with_lora(
            pretrained_path=a.base, lora_path=a.lora, device="cpu", torch_dtype=torch.bfloat16,
            lora_modules=["video_dit", "audio_dit", "dual_tower_bridge"])
        if a.nf4_base:
            # EMULATE training's NF4 base in pure bf16: training used bnb Linear4bit, whose forward
            # is matmul(x, dequant(W_nf4)). So quant->dequant each base weight back to bf16 (carrying
            # the exact 4-bit error) and store it as a normal bf16 weight -> math identical to
            # training, but inference stays pure bf16 (the proven sampler + group offload just work).
            # Avoids Linear4bit-at-inference, which doesn't compose with the two-expert group offload
            # (3 failures: shape-assert / OOM / illegal-memory-access). 2026-06-17.
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
        # Silence diffusers' group-offload trace warning ("some layers were not executed...").
        # It fires once at step 0 because the dual_tower_bridge conditioner layers are conditional
        # and don't run in the lazy-prefetch trace pass; they DO run in the real denoise (the
        # validated v3 samples were generated this exact way with good lip-sync). It's a benign
        # ~hundreds-of-layer-names wall of text; quiet it so the log stays readable. Errors still show.
        import logging as _logging
        _logging.getLogger("diffusers.hooks.group_offloading").setLevel(_logging.ERROR)
    elif a.offload == "cpu":
        pipe.enable_model_cpu_offload(0)
    else:
        pipe.to(torch.device("cuda", 0))

    # First frame. T2AV (default): a solid neutral mid-gray frame = no image conditioning, exactly
    # the trainer/sampler's neutral ref -> inference conditioning matches training. I2V (advanced):
    # a real --ref image. Both go through crop_and_resize so the pipeline input type is identical.
    if a.ref and Path(a.ref).exists():
        _ref_img = Image.open(a.ref).convert("RGB")
        log(f"ref: image '{Path(a.ref).name}' (I2V)")
    else:
        _ref_img = Image.new("RGB", (a.width, a.height), (128, 128, 128))
        log("ref: synthesized neutral gray frame (T2AV, text-driven)")
    ref = crop_and_resize(_ref_img, height=a.height, width=a.width)
    import time
    for i, p in enumerate(prompts):
        t0 = time.time()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            video, audio = pipe(prompt=p, negative_prompt=DEFAULT_NEG, num_frames=a.frames, image=ref,
                                height=a.height, width=a.width, video_fps=24.0,
                                num_inference_steps=a.steps, cfg_scale=a.cfg, seed=42, cp_mesh=None)
        _mp4 = out / f"{a.tag}_p{i}.mp4"
        save_video_with_audio(video[0], audio[0].cpu().squeeze(), str(_mp4),
                              fps=24, sample_rate=pipe.audio_sample_rate, quality=5)
        log(f"{a.tag}_p{i}.mp4 done ({time.time()-t0:.0f}s)")
        # Machine-readable: the app parses this to locate the produced A/V file (abs path).
        print(f"MOVA_SAMPLE_OUT|{_mp4.resolve()}", flush=True)
    log("MOVA_SAMPLE_DONE")


if __name__ == "__main__":
    main()
