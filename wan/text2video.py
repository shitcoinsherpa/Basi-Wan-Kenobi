# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified from Wan-Video/Wan2.2 (Apache-2.0) for BASI WAN K3N0B1: GGUF
# quantized path, block-swap offload, persistent worker, I2V graft, tiled
# VAE, profiling. See THIRD_PARTY_LICENSES.md.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.cuda_graphs import WanForwardCudaGraph
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_1 import Wan2_1_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.boundary = config.boundary
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        import time as _itime
        _t_t5 = _itime.time()
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)
        print(f"[startup-phase] t5_load {_itime.time() - _t_t5:.1f}s", flush=True)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        # [2026-06-05] BASIWAN_VAE_BF16=1 runs the VAE decode under bf16
        # autocast (default is fp32 because Wan2_1_VAE.dtype defaults to
        # torch.float). At p720_17f the full VAE decode is ~40s of the ~82s
        # ship-recipe wall; bf16 should ~halve it. Probe single-prompt then
        # multi-prompt verify before defaulting on.
        _vae_dtype = torch.bfloat16 if os.environ.get("BASIWAN_VAE_BF16") == "1" else torch.float
        _t_vae = _itime.time()
        self.vae = Wan2_1_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            dtype=_vae_dtype,
            device=self.device)
        print(f"[startup-phase] vae_load {_itime.time() - _t_vae:.1f}s", flush=True)
        # [Faster-Wan2.2 P33 — 2026-05-31] VAE+TAEHV CPU during diffusion.
        # The Wan VAE (~3 GB) and TAEHV decoder (~0.5 GB) only run at the END of
        # generate() for latent→pixel decode, but historically lived on GPU
        # throughout diffusion, eating headroom needed for FFN intermediates at
        # production shapes (1280×720, 81 frames → 20k tokens × 13824 ffn_dim
        # = ~600 MB transient × N concurrent allocs). Moving them to CPU until
        # decode time gives ~3.5 GB back. The .to(device) just before decode is
        # ~0.3 s (3 GB through PCIe at 10 GB/s) — negligible vs total wall.
        # Disable with BASIWAN_NO_DECODER_CPU=1 if needed.
        if os.environ.get("BASIWAN_NO_DECODER_CPU") != "1":
            try:
                self.vae.model.cpu()
                logging.info("[P33] VAE moved to CPU until decode time")
            except Exception as _e:
                logging.warning(f"[P33] VAE→CPU failed: {_e}")
        # [Faster-Wan2.2 P20 — amended 2026-05-31] VAE decoder torch.compile.
        # VAE decode hot path is upsample[0]+upsample[1] — 156 Conv3d calls at dim=1024
        # frame-serially. We use "reduce-overhead" mode (not "max-autotune-no-cudagraphs")
        # because the latter triggers a triton autotune storm: every triton_convolution3d
        # config requires >100 KB shared memory and Ada (sm_89) caps at 99 KB usable per
        # block, so each config fails and inductor falls back to cudnn anyway — but only
        # after burning 30+ minutes per video shape exploring the broken triton space.
        # "reduce-overhead" skips the triton conv search and uses cudnn directly. Gives
        # the same end-state speedup (~10-25% per prior research) without the warmup.
        # Set BASIWAN_NO_VAE_COMPILE=1 to disable entirely; BASIWAN_VAE_COMPILE_MODE
        # to override (e.g. "max-autotune" on Hopper where SMEM is larger).
        if os.environ.get("BASIWAN_NO_VAE_COMPILE") != "1" and hasattr(self.vae, "model"):
            try:
                _mode = os.environ.get("BASIWAN_VAE_COMPILE_MODE", "reduce-overhead")
                self.vae.model.decoder = torch.compile(
                    self.vae.model.decoder,
                    mode=_mode,
                    dynamic=True, fullgraph=False)
            except Exception as e:
                logging.warning(f"[P20] VAE decoder compile failed (continuing uncompiled): {e}")

        # [Faster-Wan2.2 P26] Optional TAEHV (madebyollin/taehv) tiny-VAE decoder.
        # 4-6× faster decode + 12-18× less memory at slight quality cost — ideal for
        # preview / iterative monitoring. Latent layout matches Wan VAE exactly:
        # 16ch, 4× temporal, 8× spatial. Enable via BASIWAN_TAEHV_VAE=1.
        # Weights path: BASIWAN_TAEHV_CKPT (default <checkpoints>/taehv/taew2_1.pth).
        # Auto-downloads from madebyollin/taehv GitHub if missing.
        self._taehv_decoder = None
        if os.environ.get("BASIWAN_TAEHV_VAE") == "1":
            try:
                from pathlib import Path as _P
                # [2026-06-11 #381] Portable default: repo-relative
                # checkpoints/taehv (this file is wan/text2video.py → repo
                # root is two parents up). Auto-downloads below if absent —
                # works on any machine, not just the dev box. Override with
                # BASIWAN_TAEHV_CKPT.
                _repo_root = _P(__file__).resolve().parent.parent
                _taehv_ckpt = _P(os.environ.get(
                    "BASIWAN_TAEHV_CKPT",
                    str(_repo_root / "checkpoints" / "taehv" / "taew2_1.pth")))
                if not _taehv_ckpt.exists():
                    import urllib.request
                    _taehv_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    _url = "https://github.com/madebyollin/taehv/raw/main/taew2_1.pth"
                    logging.info(f"[P26] downloading {_url} → {_taehv_ckpt}")
                    urllib.request.urlretrieve(_url, _taehv_ckpt)
                from taehv import TAEHV
                self._taehv_decoder = TAEHV(checkpoint_path=str(_taehv_ckpt))
                # [P33] Init on CPU; .to(device) at decode time.
                _decoder_cpu_init = os.environ.get("BASIWAN_NO_DECODER_CPU") != "1"
                _init_device = "cpu" if _decoder_cpu_init else self.device
                self._taehv_decoder = self._taehv_decoder.to(_init_device).eval()
                self._taehv_decoder.requires_grad_(False)
                logging.info(f"[P26] TAEHV loaded from {_taehv_ckpt} to {_init_device}; 4-6× decode")
            except Exception as e:
                logging.warning(f"[P26] TAEHV load failed ({e}); falling back to Wan VAE")
                self._taehv_decoder = None

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.low_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.low_noise_checkpoint)
        if hasattr(self.low_noise_model, 'set_teacache_expert_name'):
            self.low_noise_model.set_teacache_expert_name(
                config.low_noise_checkpoint)
        self.low_noise_model = self._configure_model(
            model=self.low_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        self.high_noise_model = WanModel.from_pretrained(
            checkpoint_dir, subfolder=config.high_noise_checkpoint)
        if hasattr(self.high_noise_model, 'set_teacache_expert_name'):
            self.high_noise_model.set_teacache_expert_name(
                config.high_noise_checkpoint)
        self.high_noise_model = self._configure_model(
            model=self.high_noise_model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        # [Faster-Wan2.2 P19] Wan2.2-Lightning LoRA opt-in. Set BASIWAN_LORA_DIR
        # to a directory containing high_noise_model.safetensors + low_noise_model.safetensors
        # LoRAs (e.g. <checkpoints>/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-250928).
        # When set, both experts get merged with the 4-step distilled LoRA. Caller MUST
        # then invoke generate(sampling_steps=4, guide_scale=(1.0, 1.0)) — Lightning is
        # trained CFG-free + 4-step. Expected ~20× raw forward reduction (4 steps × no CFG
        # vs default 40 steps × dual CFG = 4 forward passes vs 80).
        _lora_dir = os.environ.get("BASIWAN_LORA_DIR")
        if _lora_dir:
            from pathlib import Path
            _lora_dir = Path(_lora_dir)
            _hi = _lora_dir / "high_noise_model.safetensors"
            _lo = _lora_dir / "low_noise_model.safetensors"
            _strength = float(os.environ.get("BASIWAN_LORA_STRENGTH", "1.0"))
            if _hi.exists() and hasattr(self.high_noise_model, "apply_lora_safetensors"):
                stats = self.high_noise_model.apply_lora_safetensors(str(_hi), strength=_strength)
                logging.info(f"[P19] high_noise_model LoRA merged: {stats}")
            if _lo.exists() and hasattr(self.low_noise_model, "apply_lora_safetensors"):
                stats = self.low_noise_model.apply_lora_safetensors(str(_lo), strength=_strength)
                logging.info(f"[P19] low_noise_model LoRA merged: {stats}")

        # [Faster-Wan2.2 P24] User LoRA stacked on top of Lightning (BASI preview path).
        # Two protocols supported:
        #   - BASIWAN_USER_LORA_LIST = comma-separated paths (multi-LoRA stacking, T4.C)
        #   - BASIWAN_USER_LORA = single .safetensors path (legacy, single-LoRA)
        # If both set, LIST wins.
        # BASIWAN_USER_LORA_EXPERT = "low" | "high" | "both" (default "low").
        # BASIWAN_USER_LORA_STRENGTH = float, applied uniformly (default 1.0).
        from pathlib import Path as _P
        _user_lora_list = os.environ.get("BASIWAN_USER_LORA_LIST")
        if _user_lora_list:
            _lora_paths = [_P(s.strip()) for s in _user_lora_list.split(",") if s.strip()]
        else:
            _single = os.environ.get("BASIWAN_USER_LORA")
            _lora_paths = [_P(_single)] if _single else []
        if _lora_paths:
            _u_strength = float(os.environ.get("BASIWAN_USER_LORA_STRENGTH", "1.0"))
            _u_expert = os.environ.get("BASIWAN_USER_LORA_EXPERT", "low").lower()
            for _user_lora_p in _lora_paths:
                if not _user_lora_p.exists():
                    logging.warning(f"[P24] user LoRA path missing: {_user_lora_p}")
                    continue
                if _u_expert in ("low", "both") and hasattr(self.low_noise_model, "apply_lora_safetensors"):
                    stats = self.low_noise_model.apply_lora_safetensors(
                        str(_user_lora_p), strength=_u_strength)
                    logging.info(f"[P24] user LoRA {_user_lora_p.name} → low_noise: {stats}")
                if _u_expert in ("high", "both") and hasattr(self.high_noise_model, "apply_lora_safetensors"):
                    stats = self.high_noise_model.apply_lora_safetensors(
                        str(_user_lora_p), strength=_u_strength)
                    logging.info(f"[P24] user LoRA {_user_lora_p.name} → high_noise: {stats}")
        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        # [Faster-Wan2.2 P5] opt-in torch.compile with Morphic-validated settings:
        # mode=max-autotune-no-cudagraphs (see H3 rejection below)
        # fullgraph=False (RoPE dynamic slicing causes graph breaks)
        # dynamic=True (tolerates resolution-induced seq_len changes)
        # Use compile_repeated_blocks for one kernel cache shared across 40 layers.
        # Citations: Morphic 1.45× on H100, PyTorch blog 1.5× Flux DiT (regional compile).
        # Enable: BASIWAN_COMPILE=1
        #
        # H3 rejected: explicit CUDA Graphs / mode=reduce-overhead — three blockers:
        #   1. attention.py:69-80 builds q/k/v via Python list comprehension over
        #      variable-length packing; `flash_attn_varlen_func` ingests cu_seqlens
        #      built by cumsum on a Python list. This preprocessing is OUTSIDE any
        #      capturable region — host-side tensor construction breaks graph replay.
        #   2. 14B BF16 weights ≈ 28 GB; an RTX 4090 has 24 GB. CUDA Graph capture
        #      reserves a private memory pool for ALL captured activations on top
        #      of weights. At 720p (seq=75600, dim=5120) one activation tensor is
        #      ~730 MB; the pool reservation for the 40-block trunk would push
        #      well past 24 GB. Practitioners on 4090 already offload — graph pool
        #      reservation defeats the offloading.
        #   3. TeaCache gate at model.py:566 does `.cpu().item()` (host sync) AND
        #      Wan2.2's MoE expert swap (high_noise/low_noise) is host-side
        #      `t.item() >= boundary` branching at text2video.py:368-371. Both
        #      destroy graph capture even if (1) and (2) were fixed.
        # This rejection is about whole-pipeline compile/cudagraphs. A narrower
        # fixed-shape per-expert capture for Lightning lives in generate() behind
        # BASIWAN_CUDA_GRAPHS=1, after caches are warmed and outside the swap path.
        if os.environ.get("BASIWAN_COMPILE") == "1":
            if hasattr(model, "compile_repeated_blocks"):
                model.compile_repeated_blocks(
                    fullgraph=False, dynamic=True,
                    mode="max-autotune-no-cudagraphs",
                )
            else:
                # Fallback: compile the whole forward
                model.forward = torch.compile(
                    model.forward,
                    fullgraph=False, dynamic=True,
                    mode="max-autotune-no-cudagraphs",
                )

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def _meta_offload_basiwan_dit(self, model):
        """Single-shot meta-tensor offload — mirrors kijai's offload_transformer.

        Replaces every floating-point cuda parameter and every BareGGMLLinear
        `_basiwan_packed` tensor with `torch.empty_like(..., device='meta')`.
        Meta tensors hold zero bytes on any device, so the CUDA allocator can
        actually release its cached segments back to the driver — unlike
        `.to('cpu')` which leaves fragmented free blocks the conv-workspace
        selection can't reuse.

        WARNING: after this call the model is no longer runnable until weights
        are re-loaded (e.g. via `cached_prepack_basiwan_weights`). Use only
        for single-shot CLI pipelines where the DiT is dead after sampling
        and only the VAE decode remains.

        Reason this fix is Windows-specific: on Linux/WSL2,
        `expandable_segments:True` lets the allocator stitch freed fragments
        back into usable contiguous blocks, so `.to('cpu')` is fine. On
        Windows pip wheels, `expandable_segments` is silently non-functional
        (PYTORCH_C10_DRIVER_API_SUPPORTED absent), so fragments accumulate
        until the next large workspace allocation OOMs. Meta offload is the
        only way to actually release segments on Windows.
        """
        # Pass 1: walk named_parameters, replace each cuda fp tensor with meta.
        # The named_parameters list mutates as we go — collect first, then assign.
        _to_meta = []
        for name, param in model.named_parameters():
            if (param.data is not None
                    and param.data.is_floating_point()
                    and param.data.device.type == 'cuda'):
                _to_meta.append(name)
        for name in _to_meta:
            parts = name.split('.')
            obj = model
            for p in parts[:-1]:
                obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
            old = getattr(obj, parts[-1])
            meta_param = torch.nn.Parameter(
                torch.empty_like(old.data, device='meta'),
                requires_grad=False,
            )
            setattr(obj, parts[-1], meta_param)
        # Pass 2: BareGGMLLinear._basiwan_packed tensors (not nn.Parameters,
        # so named_parameters() misses them).
        for sub in model.modules():
            mp = getattr(sub, '_basiwan_packed', None)
            if mp is None:
                continue
            for attr in ('weight', 'weight_hi', 'scales', 'mins'):
                t = getattr(mp, attr, None)
                if t is not None and t.device.type == 'cuda':
                    setattr(mp, attr, torch.empty_like(t, device='meta'))
            # Runtime scratch is device-local — clear it so the next forward
            # would re-init on whatever device weights land on.
            sub._basiwan_runtime = None
        # Also clear the shared runtime/input pools — they hold reused GPU
        # buffers keyed by (M, N, dtype, device).
        try:
            from gguf_vendor.basiwan_q4_kernel import (
                _MARLIN_RUNTIME_POOL,
                _MARLIN_INPUT_POOL,
            )
            for _pool in (_MARLIN_RUNTIME_POOL, _MARLIN_INPUT_POOL):
                for _attr in ('buffers', '_buffers', 'pool', '_pool'):
                    _d = getattr(_pool, _attr, None)
                    if isinstance(_d, dict):
                        _d.clear()
                        break
        except Exception:
            pass

    def _model_names_for_timestep(self, t, boundary):
        if t.item() >= boundary:
            return 'high_noise_model', 'low_noise_model'
        return 'low_noise_model', 'high_noise_model'

    def _prepare_model_by_name(self,
                               required_model_name,
                               offload_model_name=None,
                               offload_previous=False):
        # [Faster-Wan2.2 P23 amended 2026-05-29] At Wan2.2-A14B-FP8 scale each
        # expert is ~13 GB. P23's "load required first, then async-evict" loses
        # at the boundary crossing on a 24 GB GPU — 13 GB resident + 13 GB
        # incoming = 26 GB → OOM. The 175 ms PCIe savings P23 was after only
        # applies when both experts fit simultaneously (~7 GB each on a 24 GB
        # GPU). Fall back to evict-first when the required model is currently
        # on CPU AND another model is on CUDA (boundary crossing case).
        _required_on_cpu = (next(getattr(self, required_model_name)
                                 .parameters()).device.type == 'cpu')
        _other_on_cuda = (offload_model_name is not None
                          and next(getattr(self, offload_model_name)
                                   .parameters()).device.type == 'cuda')
        if offload_previous and _required_on_cpu and _other_on_cuda:
            # boundary crossing: blocking evict first, then load
            getattr(self, offload_model_name).to('cpu')
            torch.cuda.empty_cache()  # release fragmented blocks before reload
            getattr(self, required_model_name).to(self.device)
            return getattr(self, required_model_name)
        # Fast path (no contention): P23's order — load (blocking) then
        # async-evict — saves ~175 ms when both can coexist on GPU.
        if _required_on_cpu:
            getattr(self, required_model_name).to(self.device)
        if offload_previous and _other_on_cuda:
            getattr(self, offload_model_name).to('cpu', non_blocking=True)
        return getattr(self, required_model_name)

    def _prepare_model_for_timestep(self, t, boundary, offload_model):
        r"""
        Prepares and returns the required model for the current timestep.

        Args:
            t (torch.Tensor):
                current timestep.
            boundary (`int`):
                The timestep threshold. If `t` is at or above this value,
                the `high_noise_model` is considered as the required model.
            offload_model (`bool`):
                A flag intended to control the offloading behavior.

        Returns:
            torch.nn.Module:
                The active model on the target device for the current timestep.
        """
        required_model_name, offload_model_name = self._model_names_for_timestep(
            t, boundary)
        if offload_model or self.init_on_cpu:
            # [Faster-Wan2.2 P23] Load BEFORE evict, then async-evict.
            # Original order: evict (blocking) → load (blocking). New order: load (blocking,
            # safe — required for forward to start) → evict (non_blocking — PCIe transfer
            # runs concurrent with subsequent forward). Saves ~175ms per boundary crossing
            # (research: ~7 GB at ~40 GB/s PCIe = ~175ms hidden inside ~4.5s forward).
            return self._prepare_model_by_name(required_model_name,
                                               offload_model_name,
                                               offload_previous=True)
        # legacy fallback path (offload_model=False, init_on_cpu=False) — unchanged below
        if False:  # original order preserved for diff-readability if needed
            if next(getattr(
                    self,
                    offload_model_name).parameters()).device.type == 'cuda':
                getattr(self, offload_model_name).to('cpu')
            if next(getattr(
                    self,
                    required_model_name).parameters()).device.type == 'cpu':
                getattr(self, required_model_name).to(self.device)
        return getattr(self, required_model_name)

    def generate(self,
                 input_prompt,
                 size=(1280, 720),
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True,
                 img=None,
                 video=None,
                 denoise_strength=1.0,
                 vace_video=None,
                 vace_context_scale=1.0,
                 vace_end_percent=1.0,
                 overlap_pixels=None,
                 vace_mask=None):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float` or tuple[`float`], *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
                If tuple, the first guide_scale will be used for low noise model and
                the second guide_scale will be used for high noise model.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        guide_scale = (guide_scale, guide_scale) if isinstance(
            guide_scale, float) else guide_scale
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # [Faster-Wan2.2 P21] T5 prompt embedding cache + batched cond/uncond encode.
        # (a) Cache: sha256(prompt+|||+n_prompt+|||+text_len) → torch-saved BF16 embeddings.
        #     Disk cache at $BASIWAN_T5_CACHE (default ~/.cache/faster-wan2.2/t5).
        # (b) Batched encode: single text_encoder([prompt, n_prompt], device) call replaces
        #     two sequential calls — 50% T5 time reduction (per research streams).
        # Set BASIWAN_NO_T5_CACHE=1 to disable cache; BASIWAN_NO_T5_BATCH=1 to disable batching.
        import hashlib
        from pathlib import Path
        _t5_cache_dir = Path(os.environ.get("BASIWAN_T5_CACHE",
                                            os.path.expanduser("~/.cache/faster-wan2.2/t5")))
        # [BASIWAN-PHASE-PROFILE 2026-06-09] Phase wall markers — gated on
        # BASIWAN_PHASE_PROFILE=1. Localizes the cold-cache regression seen
        # in multi-prompt subprocess benches (memory/cold_cache_penalty_*).
        # JSON event mirror (always-on when phase profile is on) lets the
        # persistent-worker IPC client parse phase walls without regex.
        import time as _time
        import json as _json
        _phase_prof = os.environ.get("BASIWAN_PHASE_PROFILE") == "1"
        _json_only = os.environ.get("BASIWAN_RUNNER_JSON_ONLY") == "1"
        _ph_t0 = _time.time() if _phase_prof else None

        def _emit_phase(_name, _wall_s, **_kw):
            """Emit phase marker — human-readable + JSON event for IPC parsing."""
            if not _phase_prof:
                return
            rec = {"_basiwan_event": True, "event": "phase",
                   "name": _name, "wall_s": round(_wall_s, 3), **_kw}
            line = _json.dumps(rec, separators=(",", ":"))
            if _json_only:
                print(line, flush=True)
            else:
                print(f"[BASIWAN-PHASE] {_name} {_wall_s:.2f}s"
                      + (f" ({_kw})" if _kw else ""), flush=True)
                print(f"[BASIWAN-EVENT] {line}", flush=True)

        _t5_no_cache = os.environ.get("BASIWAN_NO_T5_CACHE") == "1"
        _t5_no_batch = os.environ.get("BASIWAN_NO_T5_BATCH") == "1"
        _t5_cache_key = hashlib.sha256(
            f"{input_prompt}|||{n_prompt}|||{self.text_encoder.text_len}".encode()
        ).hexdigest()
        _t5_cache_file = _t5_cache_dir / f"{_t5_cache_key}.pt"

        if not _t5_no_cache and _t5_cache_file.exists():
            _blob = torch.load(_t5_cache_file, map_location=self.device)
            context, context_null = _blob['context'], _blob['context_null']
            logging.info(f"[P21] T5 cache HIT {_t5_cache_key[:8]}")
        else:
            t5_device = self.device if not self.t5_cpu else torch.device('cpu')
            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
            if _t5_no_batch:
                context = self.text_encoder([input_prompt], t5_device)
                context_null = self.text_encoder([n_prompt], t5_device)
            else:
                _both = self.text_encoder([input_prompt, n_prompt], t5_device)
                context, context_null = [_both[0]], [_both[1]]
            if not self.t5_cpu and offload_model:
                self.text_encoder.model.cpu()
            if self.t5_cpu:
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]
            if not _t5_no_cache:
                _t5_cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save({'context': [t.cpu() for t in context],
                            'context_null': [t.cpu() for t in context_null]},
                           _t5_cache_file)
                logging.info(f"[P21] T5 cache MISS, saved {_t5_cache_key[:8]}")
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]

        if _phase_prof:
            _emit_phase("t5_encode", _time.time() - _ph_t0)
            _ph_t1 = _time.time()

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        # [2026-06-09 #370] Optional I2V conditioning, grafted from
        # wan/image2video.py:259-323 so the optimized pipeline (block-swap,
        # chunked FFN, V2 kernel, T5 cache, phase events) is reused instead
        # of the unoptimized WanI2V class. Requires I2V expert weights
        # (in_dim=36) loaded by the runner's --model-type i2v; passing img
        # to T2V experts (in_dim=16) fails at the patch_embedding concat.
        # img=None leaves the T2V path byte-identical.
        y_cond = None
        if img is not None:
            import torchvision.transforms.functional as _TF
            _img_t = _TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
            _lat_h, _lat_w = target_shape[2], target_shape[3]
            _h, _w = size[1], size[0]
            # 4-channel temporal mask: 1 for conditioning frame 0, 0 after.
            # repeat_interleave ×4 on frame 0 matches the VAE's 4× temporal
            # compression of the first pixel frame.
            msk = torch.ones(1, F, _lat_h, _lat_w, device=self.device)
            msk[:, 1:] = 0
            msk = torch.concat([
                torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1),
                msk[:, 1:]
            ], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, _lat_h, _lat_w)
            msk = msk.transpose(1, 2)[0]
            # [P33-aware] The VAE is parked on CPU until decode; pull it to
            # GPU for this one encode, then park it again (same pattern as
            # the decode block below).
            _decoder_cpu = os.environ.get("BASIWAN_NO_DECODER_CPU") != "1"
            if _decoder_cpu:
                self.vae.model.to(self.device)
            y_lat = self.vae.encode([
                torch.concat([
                    torch.nn.functional.interpolate(
                        _img_t[None].cpu(), size=(_h, _w),
                        mode='bicubic').transpose(0, 1),
                    torch.zeros(3, F - 1, _h, _w)
                ], dim=1).to(self.device)
            ])[0]
            if _decoder_cpu:
                self.vae.model.cpu()
                torch.cuda.empty_cache()
            y_cond = torch.concat([msk, y_lat])

        # [2026-06-11 #385] SDEdit V2V restyle: encode the source video into
        # the denoiser's latent space. At denoise_strength<1 the scheduler
        # (below) starts mid-trajectory from this latent + partial noise, so
        # structure/motion carry from the source while the LoRA restyles. The
        # T2V path needs no `y` conditioning (in_dim=16) — content rides in
        # the latent init. video is a (3, F, H, W) pixel tensor in [-1,1].
        # video=None or denoise_strength>=1.0 leaves the path byte-identical.
        video_latent = None
        if video is not None and denoise_strength < 1.0:
            assert video.shape[1] == F, (
                f"video frames {video.shape[1]} != frame_num {F}")
            _decoder_cpu_v = os.environ.get("BASIWAN_NO_DECODER_CPU") != "1"
            if _decoder_cpu_v:
                self.vae.model.to(self.device)
            video_latent = self.vae.encode([video.to(self.device)])[0]
            if _decoder_cpu_v:
                self.vae.model.cpu()
                torch.cuda.empty_cache()
            if _phase_prof:
                _emit_phase("vae_encode_video", _time.time() - _ph_t1)

        # [#387] Sliding-window overlap injection: encode the previous window's
        # STYLED tail (overlap_pixels, (3, n_overlap, H, W) in [-1,1]) into the
        # leading latent frames we re-anchor to each denoise step, so the kept
        # frames stay continuous in style/motion with the prior window across the
        # seam. Same VAE (mean/std) as the main latents. overlap_pixels=None
        # leaves the path byte-identical.
        overlap_latents = None
        if overlap_pixels is not None:
            _dec_cpu_ov = os.environ.get("BASIWAN_NO_DECODER_CPU") != "1"
            if _dec_cpu_ov:
                self.vae.model.to(self.device)
            overlap_latents = self.vae.encode([overlap_pixels.to(self.device)])[0]
            if _dec_cpu_ov:
                self.vae.model.cpu()
                torch.cuda.empty_cache()
            if _phase_prof:
                _emit_phase("vae_encode_overlap", _time.time() - _ph_t1)

        # [#388] VACE depth-control: build the 96-ch control latent ONCE (it is
        # constant across all denoising steps — verified ali-vilab semantics).
        # vace_video is the depth control pixel video (3,F,H,W) in [-1,1]. We use
        # an explicit ALL-ONES mask => inactive = VAE(zeros), reactive = VAE(depth),
        # + 64ch ones-mask => the 96ch [32+64] layout the checkpoint was trained
        # on (masks=None would give 80ch, which the vace_in_dim=96 patch-embed
        # rejects). inactive/reactive MUST pass through the SAME self.vae (with
        # its mean/std scaling) as the main latents — zeroing the latent directly
        # is wrong because the VAE mean-shift offsets zeros. vace_video=None leaves
        # the path byte-identical (vace_context stays None below).
        vace_context = None
        _anchor_latents = None    # [#389] keyframe hard-lock injection targets
        _anchor_lat_idx = None
        if vace_video is not None:
            from basi import vace as _vmod
            # [#388] LOUD guard: VACE depth-lock collapses below the validated
            # regime (50 steps / guide 5.0). Warn so a weak run can't silently
            # look "broken" (the exact trap that cost a debug cycle).
            _vwarn = _vmod.check_vace_regime(sampling_steps, guide_scale)
            if _vwarn:
                import warnings as _warnings
                _warnings.warn(_vwarn, RuntimeWarning)
                print(f"[BASIWAN-WARN] {_vwarn}", flush=True)
            assert vace_video.shape[1] == F, (
                f"vace_video frames {vace_video.shape[1]} != frame_num {F}")
            _dec_cpu_vc = os.environ.get("BASIWAN_NO_DECODER_CPU") != "1"
            if _dec_cpu_vc:
                self.vae.model.to(self.device)
            _vv = vace_video.to(self.device)
            # [#389] vace_mask given => keyframe-anchored editing (0=keep anchor,
            # 1=generate); None => depth-control / full-regen (all-ones mask).
            _mask = (vace_mask.to(self.device) if vace_mask is not None
                     else _vmod.ones_mask(_vv))
            _inact_px, _react_px = _vmod.split_inactive_reactive(_vv, _mask)
            _inact = self.vae.encode([_inact_px])[0]      # 16ch VAE(zeros)
            _react = self.vae.encode([_react_px])[0]      # 16ch VAE(depth)
            if _dec_cpu_vc:
                self.vae.model.cpu()
                torch.cuda.empty_cache()
            _Flat, _Hlat, _Wlat = _inact.shape[1], _inact.shape[2], _inact.shape[3]
            # [#389] sparse keyframe masks need the causal VAE-group min-pool or
            # single-frame anchors get dropped by nearest-exact; depth (None mask)
            # keeps the byte-identical reference nearest-exact path.
            _treduce = "vae_min" if vace_mask is not None else "nearest"
            _m64 = _vmod.build_mask_channels(_mask, _Flat, _Hlat, _Wlat,
                                             temporal_reduce=_treduce)
            _z = _vmod.assemble_vace_latent(_inact, _react, _m64).to(_inact.dtype)
            vace_context = [_z]                            # list[ [96,Fl,Hl,Wl] ]
            # [#389] Keyframe HARD-LOCK: the Fun checkpoint's VACE conditioning
            # only SOFTLY preserves sparse RGB anchors (research-confirmed). So
            # ALSO RePaint-inject the anchor latent frames each denoise step — the
            # anchor latents come from the inactive channel (= VAE of the anchor
            # frames), mapped to the causal VAE latent grid. This enforces the
            # edit at the anchor positions regardless of the model's weak
            # propagation. Depth (vace_mask=None) skips this entirely.
            if vace_mask is not None:
                _pf = (_mask.reshape(_mask.shape[0], -1).amin(dim=1) < 0.5)  # anchor frames
                _idx = sorted({0 if f == 0 else (f - 1) // 4 + 1
                               for f in range(_pf.shape[0]) if bool(_pf[f])})
                _anchor_lat_idx = [i for i in _idx if i < _Flat]
                _anchor_latents = (_inact[:, _anchor_lat_idx].clone()
                                   if _anchor_lat_idx else None)
            if _phase_prof:
                _emit_phase("vace_encode", _time.time() - _ph_t1)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync_low_noise = getattr(self.low_noise_model, 'no_sync',
                                    noop_no_sync)
        no_sync_high_noise = getattr(self.high_noise_model, 'no_sync',
                                     noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync_low_noise(),
                no_sync_high_noise(),
        ):
            boundary = self.boundary * self.num_train_timesteps

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # [2026-06-11 #385] SDEdit V2V: enter the schedule mid-trajectory.
            # kijai-parity start step; our solvers ship set_begin_index +
            # begin-index-aware add_noise (x_t = (1-sigma)*x0 + sigma*noise,
            # exact float sigma), so no array slicing of sigmas — just set the
            # begin index and iterate the tail timesteps. video_latent=None
            # leaves timesteps + latents untouched (byte-identical T2V).
            if video_latent is not None:
                start_idx = max(
                    sampling_steps - int(sampling_steps * denoise_strength) - 1,
                    0)
                sample_scheduler.set_begin_index(start_idx)
                timesteps = timesteps[start_idx:]
                latents = [
                    sample_scheduler.add_noise(
                        video_latent.unsqueeze(0),
                        noise[0].unsqueeze(0),
                        timesteps[0:1].to(video_latent.device)
                    ).squeeze(0)
                ]
            else:
                # sample videos
                latents = noise

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}
            if y_cond is not None:
                arg_c['y'] = [y_cond]
                arg_null['y'] = [y_cond]

            # [Faster-Wan2.2 P10] Opt-in CUDA-event wall breakdown.
            # BASIWAN_PROFILE=1 prints per-phase ms at end of generation.
            # Costs ~2 µs/event — negligible at our shape but disabled by default.
            _profile = os.environ.get("BASIWAN_PROFILE") == "1"
            _ev = (lambda: torch.cuda.Event(enable_timing=True)) if _profile else None
            _phase_ms = {'swap': 0.0, 'fwd_cond': 0.0, 'fwd_uncond': 0.0,
                         'sched': 0.0} if _profile else None
            # [Faster-Wan2.2 P25] Fixed-shape CUDA Graphs at Lightning shape.
            # Captures the per-step WanModel.forward once per expert after warmup, replays
            # for remaining steps. Amortizes kernel launch overhead (~5700 launches/fwd → 1).
            # Auto-disables on: distributed/SP, CFG-mixed steps, TeaCache active. The CFG-skip
            # gate (P22) is required because graph capture binds context tensors at capture time.
            # [#388] VACE injects per-step vace_context via arg_c/arg_null; the
            # captured graph replay path can't carry it, so VACE forwards must
            # run eager. Disable CUDA graphs when a control latent is present.
            _cuda_graph_requested = (os.environ.get("BASIWAN_CUDA_GRAPHS") == "1"
                                     and vace_context is None)
            _cuda_graph_skip_reason = None
            # [P29 update 2026-05-28] We previously force-skipped CUDA Graphs whenever
            # Float8 weights were detected. Bisect proved torchao FP8 is graph-safe;
            # the actual blocker was flash_attn 2.8.3's internal workspace going stale
            # at N≥5 captured calls. attention.py:flash_attention auto-routes to SDPA
            # when torch.cuda.is_current_stream_capturing() returns True — so the FP8
            # auto-skip is no longer needed. Keep the rest of the guard logic
            # (distributed/SP, TeaCache, CFG-mixed) below.
            _all_cfg_skip = (os.environ.get("BASIWAN_NO_CFG_SKIP") != "1")
            if _all_cfg_skip:
                _all_cfg_skip = all(
                    abs((guide_scale[1] if t.item() >= boundary else guide_scale[0]) - 1.0) < 1e-6
                    for t in timesteps)
            if _cuda_graph_requested:
                if self.device.type != 'cuda' or not torch.cuda.is_available():
                    _cuda_graph_skip_reason = "CUDA unavailable"
                elif dist.is_initialized() or self.sp_size != 1:
                    _cuda_graph_skip_reason = "distributed / sequence-parallel path not supported"
                elif not _all_cfg_skip:
                    _cuda_graph_skip_reason = "CUDA graphs are currently wired only for CFG-free steps"
                elif getattr(self.low_noise_model, 'enable_teacache', False) or getattr(
                        self.high_noise_model, 'enable_teacache', False):
                    _cuda_graph_skip_reason = "TeaCache changes per-step control flow inside forward"
            if _cuda_graph_requested and _cuda_graph_skip_reason is not None:
                logging.info("[P25] CUDA graphs disabled: %s", _cuda_graph_skip_reason)
            _cuda_graph_active = _cuda_graph_requested and _cuda_graph_skip_reason is None
            _graph_runner = None
            _graph_model_name = None

            for _step_i, t in enumerate(tqdm(timesteps)):
                # [2026-06-10] Per-step worker event so the UI can show live
                # step progress + per-step wall. Without this the worker-mode
                # progress bar froze for the whole step loop — at 20 steps ×
                # guide>1 (CFG doubles forwards) that's many opaque minutes.
                _emit_phase(f"step", _time.time() - _ph_t1,
                            i=_step_i, n=len(timesteps)) if _phase_prof else None
                latent_model_input = latents
                timestep = [t]
                # [#388] VACE: pass the (constant) control latent to BOTH the
                # cond and uncond forwards via arg_c/arg_null — the official
                # code passes vace_context on both; dropping it on uncond
                # corrupts the CFG subtraction. The vace_end_percent gate
                # (kijai semantics) STOPS passing the control once we're past
                # that fraction of steps, so structure is depth-locked early
                # then released for late-step style freedom — set None to skip
                # the vace stack entirely those steps.
                if vace_context is not None:
                    _inject_vace = (_step_i / max(len(timesteps), 1)) < vace_end_percent
                    _vc = vace_context if _inject_vace else None
                    arg_c['vace_context'] = _vc
                    arg_c['vace_context_scale'] = vace_context_scale
                    arg_null['vace_context'] = _vc
                    arg_null['vace_context_scale'] = vace_context_scale

                timestep = torch.stack(timestep)
                required_model_name, offload_model_name = self._model_names_for_timestep(
                    t, boundary)
                sample_guide_scale = guide_scale[1] if t.item(
                ) >= boundary else guide_scale[0]

                if _profile: s0 = _ev(); e0 = _ev(); s0.record()
                if _cuda_graph_active:
                    if _graph_model_name != required_model_name:
                        if _graph_runner is not None:
                            del _graph_runner
                            _graph_runner = None
                        model = self._prepare_model_by_name(
                            required_model_name,
                            offload_model_name=offload_model_name,
                            offload_previous=(offload_model or self.init_on_cpu))
                        try:
                            _graph_runner = WanForwardCudaGraph.capture(
                                model,
                                latent=latent_model_input[0],
                                timestep=timestep,
                                context=context,
                                seq_len=seq_len,
                                amp_dtype=self.param_dtype,
                            )
                            _graph_model_name = required_model_name
                            logging.info(
                                "[P25] Captured CUDA graph for %s at latent=%s seq_len=%d",
                                required_model_name,
                                tuple(latent_model_input[0].shape),
                                seq_len,
                            )
                        except Exception as exc:
                            _cuda_graph_active = False
                            _cuda_graph_skip_reason = f"capture failed for {required_model_name}: {exc}"
                            logging.warning("[P25] %s; falling back to eager path.",
                                            _cuda_graph_skip_reason)
                            _graph_model_name = None
                    else:
                        model = getattr(self, required_model_name)
                if not _cuda_graph_active:
                    model = self._prepare_model_for_timestep(
                        t, boundary, offload_model)
                if _profile: e0.record()

                if _profile: s1 = _ev(); e1 = _ev(); s1.record()
                if _cuda_graph_active and _graph_runner is not None:
                    noise_pred_cond = _graph_runner.replay(
                        latent_model_input[0], timestep)[0]
                else:
                    noise_pred_cond = model(
                        latent_model_input, t=timestep, **arg_c)[0]
                if _profile: e1.record()

                # [Faster-Wan2.2 P22] CFG-skip when guide_scale==1.0 (Lightning mode).
                # At guide==1: noise_pred = noise_pred_uncond + 1*(noise_pred_cond - noise_pred_uncond)
                # = noise_pred_cond. The uncond forward is wasted — skipping halves per-step
                # compute. Set BASIWAN_NO_CFG_SKIP=1 to force-run uncond (e.g. for debugging).
                _cfg_skip = (abs(sample_guide_scale - 1.0) < 1e-6
                             and os.environ.get("BASIWAN_NO_CFG_SKIP") != "1")
                if _cfg_skip:
                    if _profile: s2 = _ev(); e2 = _ev(); s2.record(); e2.record()
                    noise_pred = noise_pred_cond
                else:
                    if _profile: s2 = _ev(); e2 = _ev(); s2.record()
                    noise_pred_uncond = model(
                        latent_model_input, t=timestep, **arg_null)[0]
                    if _profile: e2.record()
                    noise_pred = noise_pred_uncond + sample_guide_scale * (
                        noise_pred_cond - noise_pred_uncond)

                if _profile: s3 = _ev(); e3 = _ev(); s3.record()
                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
                # [#387] Sliding-window overlap injection (RePaint-style). After
                # the step produces latents at the NEXT noise level, overwrite the
                # leading n_ov latent frames with the KNOWN previous-window tail
                # forward-diffused to that same level. The model's temporal
                # attention then carries the prior window's style/motion into the
                # frames we keep. Skip the final step (no next timestep) — and the
                # overlap region is dropped at stitch anyway (emit_start=overlap),
                # so its only job is to guide the kept frames mid-trajectory.
                if overlap_latents is not None and _step_i < len(timesteps) - 1:
                    _n_ov = overlap_latents.shape[1]
                    _t_next = timesteps[_step_i + 1]
                    _known = sample_scheduler.add_noise(
                        overlap_latents.unsqueeze(0),
                        noise[0][:, :_n_ov].unsqueeze(0),
                        _t_next.unsqueeze(0).to(overlap_latents.device),
                    ).squeeze(0)
                    latents[0][:, :_n_ov] = _known.to(latents[0].dtype)
                # [#389] Keyframe hard-lock: RePaint-inject the anchor latent
                # frames toward their known (inactive-channel) latents each step,
                # so the edited keyframes are enforced even though the Fun
                # checkpoint only softly conditions on them.
                if _anchor_latents is not None and _step_i < len(timesteps) - 1:
                    _t_next = timesteps[_step_i + 1]
                    _ak = sample_scheduler.add_noise(
                        _anchor_latents.unsqueeze(0),
                        noise[0][:, :_anchor_latents.shape[1]].unsqueeze(0),
                        _t_next.unsqueeze(0).to(_anchor_latents.device),
                    ).squeeze(0)
                    for _j, _li in enumerate(_anchor_lat_idx):
                        latents[0][:, _li] = _ak[:, _j].to(latents[0].dtype)
                if _profile:
                    e3.record(); torch.cuda.synchronize()
                    _phase_ms['swap'] += s0.elapsed_time(e0)
                    _phase_ms['fwd_cond'] += s1.elapsed_time(e1)
                    _phase_ms['fwd_uncond'] += s2.elapsed_time(e2)
                    _phase_ms['sched'] += s3.elapsed_time(e3)

            if _profile:
                total = sum(_phase_ms.values())
                n = len(timesteps)
                logging.info("Faster-Wan2.2 wall breakdown (%d steps):", n)
                for k, v in _phase_ms.items():
                    logging.info("  %-10s %8.1f ms total | %6.1f ms/step | %5.1f%%",
                                 k, v, v / n, 100 * v / max(total, 1e-9))
                logging.info("  %-10s %8.1f ms total | %6.1f ms/step | %5.1f%%",
                             "TOTAL", total, total / n, 100.0)

            if _graph_runner is not None:
                del _graph_runner

            if _phase_prof:
                torch.cuda.synchronize()
                _emit_phase("step_loop", _time.time() - _ph_t1, n_steps=len(timesteps))
                _ph_t2 = _time.time()

            x0 = latents
            if offload_model:
                # 2026-06-08 Windows port: residency probe + (optional) meta-tensor
                # offload. Default path (.to('cpu')) leaves the CUDA allocator
                # fragmented on Windows where expandable_segments is non-functional,
                # causing VAE conv-workspace OOM. BASIWAN_META_OFFLOAD=1 replaces
                # every cuda parameter and _basiwan_packed tensor with a meta
                # empty — zero bytes held — mirroring kijai's offload_transformer
                # pattern. Single-shot use only: after meta offload, the model
                # cannot run again without re-loading weights from cache.
                _a0 = torch.cuda.memory_allocated() / 1024**3
                _r0 = torch.cuda.memory_reserved() / 1024**3
                print(f"[BASIWAN-DIAG] post-step alloc={_a0:.2f}GB resv={_r0:.2f}GB", flush=True)

                if os.environ.get("BASIWAN_META_OFFLOAD") == "1":
                    self._meta_offload_basiwan_dit(self.low_noise_model)
                    self._meta_offload_basiwan_dit(self.high_noise_model)
                else:
                    self.low_noise_model.cpu()
                    self.high_noise_model.cpu()

                import gc as _gc
                _gc.collect()
                torch.cuda.empty_cache()
                _a1 = torch.cuda.memory_allocated() / 1024**3
                _r1 = torch.cuda.memory_reserved() / 1024**3
                print(f"[BASIWAN-DIAG] post-offload alloc={_a1:.2f}GB resv={_r1:.2f}GB", flush=True)
            if _phase_prof:
                torch.cuda.synchronize()
                _emit_phase("offload", _time.time() - _ph_t2)
                _ph_t3 = _time.time()

            if self.rank == 0:
                # [P33] Pull the decoder back to GPU for the actual decode step.
                _decoder_cpu = os.environ.get("BASIWAN_NO_DECODER_CPU") != "1"
                if self._taehv_decoder is not None:
                    if _decoder_cpu:
                        self._taehv_decoder.to(self.device)
                    # [P26] TAEHV decode. x0 is list[Tensor(C, T, H, W)]; TAEHV expects
                    # (B, T, C, H, W). Output: (B, T, 3, H_full, W_full) in [0, 1].
                    # Wan's downstream expects list[Tensor(3, T_out, H_full, W_full)] in [-1, 1].
                    videos = []
                    for z in x0:  # z: (C, T, H, W)
                        t_in = z.permute(1, 0, 2, 3).unsqueeze(0).to(self.device)  # (1, T, C, H, W)
                        with torch.no_grad():
                            # P34: parallel=True requires holding all (T, H, W)
                            # conv outputs simultaneously — at p720_49f (T=13, H=720,
                            # W=1280) the parallel mode OOMs in the early conv layers.
                            # Sequential mode processes one chunk at a time. Same end
                            # result, ~2-3× slower (~5s vs ~2s at p720_49f), zero OOM risk.
                            _parallel = os.environ.get("BASIWAN_TAEHV_PARALLEL", "auto")
                            if _parallel == "auto":
                                # Auto: parallel for small T*H*W, sequential for large.
                                _vox = t_in.shape[1] * t_in.shape[3] * t_in.shape[4]
                                _parallel = _vox < 8_000_000
                            else:
                                _parallel = _parallel == "1"
                            out = self._taehv_decoder.decode_video(
                                t_in, parallel=_parallel, show_progress_bar=False)
                        # out: (1, T_out, 3, H_full, W_full) in [0, 1]
                        # Wan downstream expects list[Tensor(3, T, H, W)] in [-1,1] float32.
                        v = out.squeeze(0).permute(1, 0, 2, 3).float()
                        v = v.mul_(2).sub_(1).clamp_(-1, 1)
                        videos.append(v)
                    if _decoder_cpu:
                        self._taehv_decoder.cpu()
                else:
                    if _decoder_cpu:
                        self.vae.model.to(self.device)
                    videos = self.vae.decode(x0)
                    if _decoder_cpu:
                        self.vae.model.cpu()
                if _decoder_cpu:
                    torch.cuda.empty_cache()

        if _phase_prof:
            torch.cuda.synchronize()
            _emit_phase("vae_decode", _time.time() - _ph_t3)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
