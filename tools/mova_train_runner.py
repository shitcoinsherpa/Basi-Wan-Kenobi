"""MOVA joint-A/V LoRA trainer (the runner the gym's basi/mova_train.py launches; MOVA's
equivalent of musubi's wan_train_network.py).

CORRECTNESS (the whole point -- usable by EVERYONE via MOVA's shipped inference):
  - LoRA is MOVA-NATIVE: lora_utils.inject_lora_to_model + save_lora_weights, the EXACT format
    MOVALoRA.from_pretrained_with_lora loads. (No custom wrapper / converter hack.)
  - NF4-resident video tower (validated: trains, M6 PASS) -> fits a consumer 24GB+32GB box.
  - Per-checkpoint + baseline A/V SAMPLES (like the Wan gym's --sample_prompts + --sample_at_first):
    several prompts incl off-style, rendered at step 0 (baseline) and every checkpoint, via the
    shipped MOVA pipeline. Sampling is best-effort (try/except) -- it NEVER blocks training.

Single high-noise expert (video_dit, global_step=0): per the Wan finding, the high-noise expert
carries global style/color/material -- the right target for a STYLE LoRA. video_dit_2 (low-noise)
is left untrained (its LoRA stays zero -> unchanged at inference, which is correct). A 2nd run
(global_step=1) could later add the low-noise expert.

Usage: see basi/mova_train.py (it builds the args). ASCII.
"""
import os, sys, time, argparse, statistics
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.6")  # NOT expandable_
# segments: it pins growable segments empty_cache can't release -> desktop starved. See mova-m2 memo.
os.environ.pop("ACCELERATE_USE_DEEPSPEED", None); os.environ.pop("ACCELERATE_USE_FSDP", None)

MOVA = Path(os.environ.get("MOVA_REPO", str(Path.home() / "MOVA")))
# Portable: MOVA-360p weights under BASIWAN_CKPT_DIR (default <repo>/checkpoints), env-overridable.
# No dev-box path -- resolves on any install once the weights are present.
CKPT = os.environ.get("MOVA_CKPT", str(
    Path(os.environ.get("BASIWAN_CKPT_DIR",
                        str(Path(__file__).resolve().parent.parent / "checkpoints"))) / "MOVA-360p"))
CFG = str(MOVA / "configs" / "training" / "mova_train_low_resource.py")
sys.path.insert(0, str(MOVA)); sys.path.insert(0, str(MOVA / "scripts" / "training_scripts"))

import torch, torch.nn as nn, psutil
import bitsandbytes as bnb
from bitsandbytes.nn import Linear4bit, Params4bit
from mova.engine.trainer.accelerate.lora_utils import inject_lora_to_model, save_lora_weights, LoRALinear

GB = 1/1e9
DEFAULT_NEG = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量"
def log(m): print(f"[mova-train] {m}", flush=True)
def vram_free(): f, t = torch.cuda.mem_get_info(); return f*GB, t*GB


def nf4_quantize_lora_bases(root):
    """Replace each LoRALinear.original_layer (an nn.Linear) under root with a Linear4bit (NF4).
    Inject runs FIRST on bf16 (so lora_utils' dtype detection sees bf16 -> correct bf16 adapters);
    here we 4-bit only the frozen BASE inside each LoRALinear. LoRALinear.forward calls
    self.original_layer(x) -> works with Linear4bit."""
    n = 0
    for _, m in root.named_modules():
        if isinstance(m, LoRALinear) and isinstance(m.original_layer, nn.Linear):
            lin = m.original_layer
            q = Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None,
                           compute_dtype=torch.bfloat16, quant_type="nf4")
            q.weight = Params4bit(lin.weight.data.clone().to(torch.bfloat16),
                                  requires_grad=False, quant_type="nf4")
            if lin.bias is not None:
                q.bias = nn.Parameter(lin.bias.data.clone().to(torch.bfloat16), requires_grad=False)
            m.original_layer = q; n += 1
    return n


def _encode_latents(model, video, audio, first_frame, dtype=torch.bfloat16):
    """Replicate MOVATrain.training_step's DETERMINISTIC encodes (mova_train.py:1305-1375):
    video latents, the I2V first-frame conditioning y (mask + encoded first frame), DAC audio
    latents. All use .mode() on deterministically-loaded clips -> bit-stable -> safe to cache once.
    The stochastic parts (timestep, noise) are NOT here -- they stay live in cached_training_step."""
    dev = video.device
    if video.dim() == 5 and video.shape[1] != video.shape[2] and video.shape[1] > video.shape[2]:
        video = video.permute(0, 2, 1, 3, 4)                  # [B,T,C,H,W] -> [B,C,T,H,W]
    video = video.to(dtype=dtype, device=dev)
    first_frame = first_frame.to(dtype=dtype, device=dev)
    audio = audio.to(dtype=torch.float32, device=dev)
    B, C, num_frames, H, W = video.shape
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        vlat = model.normalize_video_latents(model.video_vae.encode(video).latent_dist.mode())
    H_l, W_l, T_l = H // 8, W // 8, vlat.shape[2]
    msk = torch.zeros(B, 4, T_l, H_l, W_l, device=dev); msk[:, :, 0, :, :] = 1
    vae_input = torch.concat([first_frame.unsqueeze(2),
                              torch.zeros(B, C, num_frames - 1, H, W, device=dev, dtype=dtype)], dim=2)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        y = model.normalize_video_latents(model.video_vae.encode(vae_input).latent_dist.mode())
    y = torch.concat([msk.to(dtype=dtype), y], dim=1)          # [B,20,T',H',W']
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float32):
        x_pad = model.audio_vae.preprocess(audio, sample_rate=model.sample_rate)
        z = model.audio_vae.encode(x_pad)[0]                   # DAC: (z, codes, latents, cl, cbl)
        alat = z.mode().to(dtype=dtype)
    return vlat, y, alat


def cached_training_step(model, video_latents, y, audio_latents, caption, global_step):
    """The post-encode half of MOVATrain.training_step (mova_train.py:1377-1450) run from CACHED
    latents: timestep sample -> flow-matching noise -> dual-tower forward -> MSE loss. Identical
    math to the live step; only the (deterministic) encodes are skipped. global_step parity selects
    the expert + timestep half exactly as the live step does (here always 0 = high-noise expert)."""
    from mova.diffusion.pipelines.mova_train import TimestepConfig
    dev = video_latents.device
    context = model._get_t5_prompt_embeds(caption, device=dev)            # text cache (_t5_cached)
    tcfg = TimestepConfig(max_timestep_boundary=1.0, min_timestep_boundary=0.0,
                          weighting_scheme="uniform", logit_mean=0.0, logit_std=1.0,
                          mode_scale=1.0, independent_timesteps=False)
    boundary = (model.scheduler.timesteps >= model.boundary_ratio * model.scheduler.num_train_timesteps
                ).sum().item() / model.scheduler.num_train_timesteps
    if global_step % 2 == 0: tcfg.max_timestep_boundary = boundary
    else: tcfg.min_timestep_boundary = boundary
    timestep, audio_timestep = model.sample_timestep_pair(tcfg)
    timestep = timestep.to(device=dev); audio_timestep = audio_timestep.to(device=dev)
    video_noise = torch.randn_like(video_latents); audio_noise = torch.randn_like(audio_latents)
    noisy_video = model.scheduler.add_noise(video_latents, video_noise, timestep).to(device=dev)
    noisy_audio = model.scheduler.add_noise(audio_latents, audio_noise, audio_timestep).to(device=dev)
    cur = model.video_dit if global_step % 2 == 0 else model.video_dit_2
    video_pred, audio_pred = model.inference_single_step(
        visual_dit=cur, visual_latents=noisy_video, audio_latents=noisy_audio, y=y, context=context,
        timestep=timestep.unsqueeze(0) if timestep.dim() == 0 else timestep[:1],
        audio_timestep=audio_timestep.unsqueeze(0) if audio_timestep.dim() == 0 else audio_timestep[:1],
        video_fps=24.0, cp_mesh=None)
    video_target = video_noise - video_latents                           # flow matching v = noise - x
    audio_target = audio_noise - audio_latents
    vloss = torch.nn.functional.mse_loss(video_pred.to(video_target.dtype), video_target)
    aloss = torch.nn.functional.mse_loss(audio_pred.to(audio_target.dtype), audio_target)
    return {"loss": vloss + aloss, "video_loss": vloss, "audio_loss": aloss}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--steps", type=int, default=2000); ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--height", type=int, default=240); ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--frames", type=int, default=81); ap.add_argument("--rank", type=int, default=16)
    # v3 style-push levers (2026-06-18, research-backed). Default OFF = unchanged v1/v2 behavior.
    ap.add_argument("--alpha", type=int, default=0, help="LoRA alpha (0 -> == rank, the v1/v2 default). "
                    "Style convention is alpha=rank/2 (e.g. rank32/alpha16) for smoother transfer.")
    ap.add_argument("--freeze-audio", action="store_true", help="skip LoRA on audio_dit (frozen-encoder "
                    "protocol). Use once audio is already intelligible: the audio tower overtrains while "
                    "video loss still drops (multi-modal gradient imbalance); AV-sync lives in the bridge, "
                    "which stays trainable, so lip-sync is preserved. Redirects capacity to video style.")
    ap.add_argument("--lora-ffn", action="store_true", help="also LoRA the video DiT FFN linears (ffn.0, "
                    "ffn.2), not just attention q/k/v/o. Attention-only LoRA underfits STYLE capacity.")
    ap.add_argument("--warmup", type=int, default=50); ap.add_argument("--save-every", type=int, default=516)
    ap.add_argument("--base", choices=["nf4", "fp8"], default="nf4")
    ap.add_argument("--sample-prompts", default=None, help="file: one sample prompt per line")
    ap.add_argument("--sample-steps", type=int, default=50, help="denoise steps for samples "
                    "(MOVA shipped recipe = 50; fewer underdenoises to near-black)")
    ap.add_argument("--no-sample", action="store_true")
    ap.add_argument("--no-cache-latents", action="store_true",
                    help="disable offline video+audio latent caching (M2). Default: cache ON -> "
                    "encode each clip's video+audio latents ONCE to disk, then EVICT both VAEs "
                    "(kills the per-step VAE encode AND frees their resident VRAM).")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (faster, ~25-33% less recompute, but more "
                    "activation VRAM). Affordable once latent caching frees the VAEs' VRAM.")
    ap.add_argument("--vram-fraction", type=float, default=0.86,
                    help="hard cap on this process's GPU memory (fraction of total) so the Windows "
                    "desktop keeps headroom and can't be frozen. 0.86 of 24GB ~= 20.6GB (~3.4GB free).")
    ap.add_argument("--empty-cache-every", type=int, default=10,
                    help="call torch.cuda.empty_cache() every N steps to return the allocator's "
                    "reserved slack to the OS (real desktop headroom; expandable_segments bypasses "
                    "the fraction cap). 0 disables. ~10-50ms vs a 12s step.")
    args = ap.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    samp_dir = out / "samples"; samp_dir.mkdir(exist_ok=True)

    from mmengine.config import Config
    from mova.registry import DIFFUSION_PIPELINES, DATASETS
    from mova.datasets.video_audio_dataset import collate_fn
    from torch.utils.data import DataLoader
    import mova.diffusion.pipelines.mova_train as MT

    f0, t0g = vram_free()
    log(f"GPU free={f0:.1f}/{t0g:.1f}GB | {args.height}x{args.width}x{args.frames}f base={args.base} "
        f"rank={args.rank} steps={args.steps} lr={args.lr}")

    cfg = Config.fromfile(CFG)
    cfg.merge_from_dict({"diffusion_pipeline.from_pretrained": CKPT, "data.dataset.data_root": args.dataset,
                         "data.dataset.num_frames": args.frames, "data.dataset.height": args.height,
                         "data.dataset.width": args.width})
    dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
    # Leave headroom for the Windows desktop compositor (DWM shares the GPU): hard-cap this process
    # so the caching allocator can't grab the last few GB and freeze the desktop (the prior
    # force-restart incident). Real training need ~16GB << this cap, so no OOM. Configurable.
    try:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction, 0)
        log(f"GPU memory cap: {args.vram_fraction:.0%} (~{args.vram_fraction*vram_free()[1]:.1f}GB) "
            f"-> ~{(1-args.vram_fraction)*vram_free()[1]:.1f}GB reserved for the desktop")
    except Exception as e:
        log(f"vram cap skipped: {e}")
    # MOVA's inference __call__ uses dist.get_rank() (tqdm gate) -> needs a process group even
    # single-GPU. gloo (CPU rendezvous) is enough; cp_mesh stays None (no sequence parallel).
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29555")
        dist.init_process_group(backend="gloo", world_size=1, rank=0)

    dcfg = cfg.data.dataset.copy(); dcfg["transform"] = None
    ds = DATASETS.build(dcfg)
    dl = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True,
                    drop_last=True, collate_fn=collate_fn)
    log(f"dataset: {len(ds)} clips")

    log("building model (CPU bf16, mmap-lazy) ...")
    model = DIFFUSION_PIPELINES.build(cfg.diffusion_pipeline,
                default_args={"device": "cpu", "torch_dtype": torch.bfloat16})
    model.scheduler.set_timesteps(cfg.trainer.get("num_train_timesteps", 1000), training=True)

    sample_prompts = []
    if args.sample_prompts and Path(args.sample_prompts).exists():
        sample_prompts = [l.strip() for l in Path(args.sample_prompts).read_text(encoding="utf-8").splitlines() if l.strip()]
    do_sample = bool(sample_prompts) and not args.no_sample

    # ---- text cache: ALL captions + sample prompts + negative; encode once, EVICT UMT5 (7.2GB) ----
    caps = sorted({(it.get("caption", "") or "") for it in ds.metadata})
    to_cache = list(dict.fromkeys(caps + sample_prompts + [DEFAULT_NEG, ""]))
    log(f"pre-encoding {len(to_cache)} unique texts ({len(caps)} caps + {len(sample_prompts)} sample + neg) -> cache; evicting text_encoder")
    model.text_encoder.to(dev)
    CACHE = {}
    with torch.no_grad():
        for c in to_cache:
            CACHE[c] = model._get_t5_prompt_embeds([c], device=dev).detach().to("cpu")
    model.text_encoder.to("cpu"); model.text_encoder = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    def _t5_cached(self, prompt, device=None, dtype=None):
        key = prompt[0] if isinstance(prompt, (list, tuple)) else prompt
        t = CACHE.get(key)
        if t is None: t = CACHE.get("", next(iter(CACHE.values())))
        return t.to(device) if device is not None else t
    MT.MOVATrain._get_t5_prompt_embeds = _t5_cached
    log(f"text cache ready ({len(CACHE)}); GPU free={vram_free()[0]:.1f}GB")

    # ---- latent cache (M2): encode each clip's video+audio latents ONCE -> disk, EVICT both VAEs.
    # This is the biggest single lever (verified: kohya/musubi/diffusion-pipe all do it) -- it kills
    # the per-step VAE encode AND frees the VAEs' resident VRAM (off the 24GB wall). Incremental /
    # resumable (skip-existing). The encodes are deterministic (.mode()), so cached == live. ----
    USE_LCACHE = not args.no_cache_latents
    LCACHE_PATHS = []
    if USE_LCACHE:
        lc_dir = Path(args.dataset) / "latent_cache" / f"{args.height}x{args.width}x{args.frames}"
        lc_dir.mkdir(parents=True, exist_ok=True)
        model.video_vae.to(dev); model.audio_vae.to(dev)
        n_enc = 0
        for idx in range(len(ds)):
            cp = lc_dir / f"{idx:06d}.pt"
            if not cp.exists():
                item = ds[idx]
                vlat, y, alat = _encode_latents(model,
                    item["video"].unsqueeze(0).to(dev), item["audio"].unsqueeze(0).to(dev),
                    item["first_frame"].unsqueeze(0).to(dev))
                torch.save({"video_latents": vlat.squeeze(0).to(torch.bfloat16).cpu(),
                            "y": y.squeeze(0).to(torch.bfloat16).cpu(),
                            "audio_latents": alat.squeeze(0).to(torch.bfloat16).cpu(),
                            "caption": item.get("caption", "")}, str(cp))
                n_enc += 1
                if n_enc % 50 == 0: log(f"  latent-cache {idx + 1}/{len(ds)} ({n_enc} new) ...")
            LCACHE_PATHS.append(str(cp))
        model.video_vae = None; model.audio_vae = None
        gc.collect(); torch.cuda.empty_cache()
        log(f"latent cache ready ({len(LCACHE_PATHS)} clips, {n_enc} new); evicted video_vae+audio_vae; "
            f"GPU free={vram_free()[0]:.1f}GB")

        class _CachedLatents(torch.utils.data.Dataset):
            def __init__(self, paths): self.paths = paths
            def __len__(self): return len(self.paths)
            def __getitem__(self, i): return torch.load(self.paths[i], map_location="cpu")
        dl = DataLoader(_CachedLatents(LCACHE_PATHS), batch_size=1, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, collate_fn=lambda b: b[0])

    # dump a real ref frame (a dataset clip's first frame) to disk for the SEPARATE sampler process
    ref_png = out / "ref.png"
    if do_sample:
        from PIL import Image
        ff = ds[0]["first_frame"]
        arr = (((ff.float() + 1) / 2).clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(arr).save(str(ref_png))
        log(f"ref frame for sampling -> {ref_png}")

    # ---- MOVA-NATIVE LoRA: inject per-tower on bf16 FIRST (correct dtype), then NF4 the bases ----
    # v3 style-push (2026-06-18): --freeze-audio drops audio_dit from the tower list (audio already
    # solved + overtrains; the bridge carries AV-sync so lip-sync survives a frozen audio tower).
    # --lora-ffn extends targets to the video DiT FFN linears. --alpha decouples alpha from rank.
    alpha = float(args.alpha) if args.alpha else float(args.rank)
    tgt = ["q", "k", "v", "o"] + (["ffn.0", "ffn.2"] if args.lora_ffn else [])
    towers = (["video_dit", "dual_tower_bridge"] if args.freeze_audio
              else ["video_dit", "audio_dit", "dual_tower_bridge"])
    inj = 0
    for tower in towers:
        m = getattr(model, tower, None)
        if m is not None:
            inj += len(inject_lora_to_model(m, rank=args.rank, alpha=alpha,
                                            target_modules=tgt,
                                            exclude_modules=["time_projection", "time_embedding"]))
    log(f"injected {inj} native LoRALinear (towers={'+'.join(towers)}, targets={tgt}, "
        f"rank={args.rank} alpha={alpha:.0f})")
    model.video_dit_2 = nn.Module()  # low-noise expert untrained; placeholder for save_lora_weights

    if args.base == "nf4":
        nq = nf4_quantize_lora_bases(model.video_dit.blocks)
        if getattr(model, "dual_tower_bridge", None) is not None:
            nq += nf4_quantize_lora_bases(model.dual_tower_bridge)
        log(f"NF4-quantized {nq} LoRA bases (video_dit.blocks + bridge)")

    for attr in ("video_dit", "audio_dit", "dual_tower_bridge", "video_vae", "audio_vae"):
        m = getattr(model, attr, None)
        if m is not None: m.to(dev)
    for n_, p in model.named_parameters():
        p.requires_grad_("lora_" in n_)
    train_params = [p for p in model.parameters() if p.requires_grad]
    # Gradient checkpointing recomputes activations in backward (~25-33% slower) to save VRAM.
    # With latent caching (M2) freeing the VAEs' VRAM, we can afford to turn it OFF for a
    # quality-identical speed gain -- IF the activations still fit. Configurable; default ON (safe).
    grad_ckpt = not args.no_grad_ckpt
    for attr in ("video_dit", "audio_dit"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "use_gradient_checkpointing"):
            m.use_gradient_checkpointing = grad_ckpt
            if hasattr(m, "use_gradient_checkpointing_offload"): m.use_gradient_checkpointing_offload = False
    log(f"gradient_checkpointing={'ON' if grad_ckpt else 'OFF (faster, needs more VRAM)'}")
    log(f"trainable LoRA {sum(p.numel() for p in train_params)/1e6:.2f}M; GPU free={vram_free()[0]:.1f}GB")

    opt = bnb.optim.AdamW8bit(train_params, lr=args.lr, weight_decay=0.01)
    import math
    def lr_at(s):
        if s < args.warmup: return s / max(1, args.warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    def save_ckpt(tag):
        d = out / f"checkpoint_{tag}"; d.mkdir(parents=True, exist_ok=True)
        save_lora_weights(model, str(d), module_names=["q", "k", "v", "o"], format="peft")
        # FIX MOVA's save_lora_weights bug: it writes config['rank'] = the class name string
        # ('MOVATrain'), which crashes MOVALoRA.load_lora (rank -> nn.Linear(in, 'MOVATrain')).
        # Overwrite with the real int rank so the LoRA actually loads in shipped inference.
        # Record the BASE precision: an NF4-trained LoRA learned deltas relative to the 4-bit base,
        # so the sampler MUST emulate that base (quant->dequant) or the style is lost (measured
        # 2026-06-17: bf16-base inference vs NF4-emulated differ ~57/255, style only transfers on the
        # matched base). mova_sample.py reads this to auto-enable --nf4-base.
        torch.save({"rank": int(args.rank), "alpha": float(args.rank),
                    "target_modules": ["q", "k", "v", "o"], "base": str(args.base)},
                   str(d / "lora_config.pt"))
        log(f"saved native LoRA -> {d.name}")
        return d

    # ---- CORRECT sampling: a SEPARATE process between epochs (Flux-Gym/musubi pattern). The
    # runner FREES the GPU (training model -> CPU), spawns tools/mova_sample.py which loads
    # base+LoRA FRESH via the shipped MOVALoRA path (BOTH bf16 experts, real high/low boundary,
    # group offload, 50 steps), then restores the GPU. Fully isolated: cannot share/mutate the
    # training scheduler, cannot alias experts. A sampler failure only logs a warning. ----
    import subprocess
    _RESIDENT = ("video_dit", "audio_dit", "dual_tower_bridge", "video_vae", "audio_vae")
    def _move(devnm):
        for a_ in _RESIDENT:
            m = getattr(model, a_, None)
            if m is not None: m.to(devnm)
        # synchronize so the CPU<->GPU transfers + frees actually complete before we measure/resume;
        # without it the resumed training step races a half-freed allocator (fragmentation OOM at the
        # sample->resume boundary, 2026-06-16). Pairs with max_split_size_mb in the launch env.
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    def sample_sep(tag, lora_dir):
        if not do_sample: return
        try:
            _move("cpu")  # hand the whole GPU to the isolated sampler
            cmd = [sys.executable, str(Path(__file__).resolve().parent / "mova_sample.py"),
                   "--base", CKPT, "--prompts", args.sample_prompts, "--ref", str(ref_png),
                   "--out", str(samp_dir), "--tag", tag, "--height", str(args.height),
                   "--width", str(args.width), "--frames", str(args.frames),
                   "--steps", str(args.sample_steps), "--offload", "group"]
            if lora_dir is not None: cmd += ["--lora", str(lora_dir)]
            log(f"sampling[{tag}] -> separate process (GPU freed) ...")
            # the runner holds a gloo PG on MASTER_PORT (29555) for MOVA's dist.get_rank();
            # the child inherits that env -> give it a DISTINCT port so its own TCPStore can bind
            # (else EADDRINUSE on the parent's port). Separate processes, separate ports = no clash.
            cenv = {**os.environ, "MASTER_PORT": "29560"}
            r = subprocess.run(cmd, env=cenv)
            log(f"sampling[{tag}] {'ok' if r.returncode == 0 else 'rc=' + str(r.returncode)}")
        except Exception as e:
            log(f"SAMPLE_WARN ({tag}) -- training continues: {type(e).__name__} {str(e)[:160]}")
        finally:
            _move(dev)  # restore training residency

    log("=== training ===")
    proc = psutil.Process(); losses = []; it = iter(dl); t0 = time.time()
    _b = save_ckpt("baseline"); sample_sep("baseline", _b)  # sample_at_first (lora_B init=0 ~= base)
    for step in range(args.steps):
        try: batch = next(it)
        except StopIteration: it = iter(dl); batch = next(it)
        ts = time.time(); opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if USE_LCACHE:  # cached latents -> skip the per-step VAE encode (M2)
                ld = cached_training_step(model,
                    batch["video_latents"].unsqueeze(0).to(dev, torch.bfloat16),
                    batch["y"].unsqueeze(0).to(dev, torch.bfloat16),
                    batch["audio_latents"].unsqueeze(0).to(dev, torch.bfloat16),
                    [batch["caption"]], global_step=0)
            else:           # live encode (no cache): the original path
                ld = model.training_step(
                    video=batch["video"].to(dev, torch.bfloat16),
                    audio=batch["audio"].to(dev, torch.float32),
                    first_frame=batch["first_frame"].to(dev, torch.bfloat16),
                    caption=batch["caption"], video_fps=24.0, global_step=0)
        loss = ld["loss"]; loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0); opt.step(); sched.step()
        # Return the allocator's reserved slack to the OS so the Windows desktop keeps headroom.
        # Per-step allocated peak is ~16GB but the caching allocator hoards ~24GB reserved (and
        # expandable_segments bypasses set_per_process_memory_fraction); empty_cache frees the unused
        # blocks back to the driver. ~10-50ms vs a 12s step = negligible. Every N steps (configurable).
        if args.empty_cache_every and step % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        losses.append({k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in ld.items()})
        if step % 20 == 0 or step == args.steps - 1:
            w = losses[-50:]
            log(f"step {step}/{args.steps} {time.time()-ts:.1f}s | win50 loss={statistics.fmean([d.get('loss',0) for d in w]):.4f} "
                f"v={statistics.fmean([d.get('video_loss',0) for d in w]):.4f} a={statistics.fmean([d.get('audio_loss',0) for d in w]):.4f} "
                f"| lr={sched.get_last_lr()[0]:.2e} peak={torch.cuda.max_memory_allocated()*GB:.1f}GB rss={proc.memory_info().rss*GB:.1f}GB "
                f"elapsed={(time.time()-t0)/60:.1f}m finite={bool(torch.isfinite(loss).all())}")
        if (step + 1) % args.save_every == 0:
            _d = save_ckpt(f"step{step+1}"); sample_sep(f"step{step+1}", _d)
    _d = save_ckpt("final"); sample_sep("final", _d)
    log("MOVA_TRAIN_DONE")


if __name__ == "__main__":
    main()
