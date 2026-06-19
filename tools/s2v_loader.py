"""Wan2.2-S2V GGUF loading for the optimized stack (S2).

Importable (no argparse/CLI side effects, unlike run_one_video_gguf.py) so both
the runner (S4) and the CPU smoke (S2) can use it. S2V is SINGLE-expert (one 14B
WanModel_S2V, not an A14B high/low MoE pair), so this is one GGUF, not two.

The load mirrors the per-expert path in run_one_video_gguf._build_pipe_gguf:
meta-build from the s2v config.json -> to_empty(cpu) -> swap nn.Linear for
BareGGMLLinear -> assign every GGUF tensor by name -> rematerialize the computed
`freqs`. The extra S2V submodules (audio_injector, casual_audio_encoder,
frame_packer, cond_encoder, trainable_cond_mask) are built straight from the
config and populated by the same generic by-name loop (S1 verified the GGUF
carries all those families).
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn as nn

from gguf_vendor.bare_gguf import gguf_sd_loader, swap_linear_with_ggml


def load_s2v_model_from_gguf(gguf_path, config_json_path):
    """Return (model, missing, mismatched).

    `missing`   = GGUF keys with no target module (model built too small).
    `mismatched`= GGUF keys that hit a target but could not be assigned.
    A correct load has both empty.
    """
    from wan.modules.s2v.model_s2v import WanModel_S2V
    from wan.modules.model import rope_params

    with open(config_json_path) as f:
        mcfg = json.load(f)
    mcfg = {k: v for k, v in mcfg.items() if not k.startswith("_")}
    with torch.device("meta"):
        model = WanModel_S2V(**mcfg)
    model = model.to_empty(device="cpu")

    n_swapped = swap_linear_with_ggml(model)
    print(f"[s2v-load] swapped {n_swapped} Linears for GGML dequant", flush=True)

    sd, extra = gguf_sd_loader(str(gguf_path), handle_prefix=None)
    arch = extra.get("arch_str")
    if arch is not None and arch != "wan":
        print(f"[s2v-load] WARN: expected arch 'wan', got {arch!r}", flush=True)

    missing, mismatched = [], []
    for key, tensor in sd.items():
        mod = model
        attrs = key.split(".")
        try:
            for a in attrs[:-1]:
                mod = getattr(mod, a) if not a.isdigit() else mod[int(a)]
        except (AttributeError, IndexError):
            missing.append(key)
            continue
        leaf = attrs[-1]
        try:
            if leaf in mod._parameters:
                mod._parameters[leaf] = nn.Parameter(
                    tensor.detach() if hasattr(tensor, "detach") else tensor,
                    requires_grad=False)
            elif hasattr(mod, leaf):
                setattr(mod, leaf, tensor)
            else:
                mismatched.append(key)
        except Exception as e:  # noqa: BLE001
            mismatched.append(f"{key}: {type(e).__name__}: {e}")

    # The meta-build leaves COMPUTED plain-attr tensors (not params, not
    # registered buffers) on the meta device — to_empty only materializes
    # params/buffers. WanModel_S2V.freqs and the FramePackMotioner's
    # zip_frame_buckets + freqs are all such constants. Rematerialize them.
    if hasattr(model, "freqs") and getattr(model.freqs, "is_meta", False):
        d = model.dim // model.num_heads
        model.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))], dim=1)

    fp = getattr(model, "frame_packer", None)
    if fp is not None and any(
            getattr(getattr(fp, a, None), "is_meta", False)
            for a in ("zip_frame_buckets", "freqs")):
        # Rebuild a fresh tiny FramePackMotioner on CPU and steal its computed
        # constants (uses the real init logic — no formula duplication). Its
        # conv weights come from the GGUF; only these two constants are needed.
        from wan.modules.s2v.motioner import FramePackMotioner
        fresh = FramePackMotioner(
            inner_dim=model.dim, num_heads=model.num_heads,
            zip_frame_buckets=[1, 2, 16],
            drop_mode=mcfg.get("framepack_drop_mode", "drop"))
        fp.zip_frame_buckets = fresh.zip_frame_buckets
        fp.freqs = fresh.freqs

    # GATE: no computed tensor may remain on meta — that would crash mid-generate
    # (as zip_frame_buckets.item() did). Scan every module's plain-attr tensors.
    meta_leftovers = []
    for mod_name, mod in model.named_modules():
        for attr, val in list(mod.__dict__.items()):
            if isinstance(val, torch.Tensor) and getattr(val, "is_meta", False):
                meta_leftovers.append(f"{mod_name}.{attr}" if mod_name else attr)

    return model, missing, mismatched, meta_leftovers


def build_s2v_pipe_gguf(gguf_path, config_json_path, checkpoint_dir,
                        base_dir=None, device_id=0, force_dequant=False,
                        resident=True, block_swap_n=None, ffn_chunk=0):
    """[S4] Build the WanS2V pipeline with our GGUF-loaded noise_model.

    Reuses the vendored WanS2V wrapper (so encode_audio + the 80-frame chunked
    `generate` come for free) but swaps its DiT for the GGUF model by stubbing
    WanModel_S2V.from_pretrained. T5 stays on CPU, the wav2vec2 audio encoder on
    CPU (cuBLASLt dodge), the Wan2.1 VAE on the cuda device. The DiT is either
    BASIWAN-kernel prepacked (default) or left raw for the pure-dequant fallback
    (force_dequant / kernel-unavailable). With `resident=True` the whole DiT is
    moved to the GPU (BareGGMLLinear._apply moves the packed weight); block-swap
    is layered on by the caller when VRAM is tight.
    """
    import os
    # [S10] Force bf16 norms for S2V. model.py auto-gates RMS/LayerNorm to bf16 only at
    # seq>50000 (to dodge fp32 norm transients that blow the 24GB wall). But 720p with a
    # short window sits in the 40-50k seq gap (48f -> seq 45760) where the gate stays OFF
    # and the fp32 norm copies (+ rope's .float()) OOM the DiT step. bf16 norms are
    # validated-equivalent (that's what the gate uses at high seq); forcing them for S2V
    # frees ~1-2GB/step so 720p multi-chunk fits. Set in-process so it survives the
    # worker env scrub (_studio_env pops these) and the forward reads it live.
    os.environ["BASIWAN_RMS_BF16"] = "1"
    os.environ["BASIWAN_LN_BF16"] = "1"
    from wan.configs import WAN_CONFIGS
    from wan.modules.s2v.model_s2v import WanModel_S2V
    from wan.speech2video import WanS2V
    from gguf_vendor.bare_gguf import prepack_basiwan_weights, basiwan_kernel_available

    _orig = WanModel_S2V.from_pretrained.__func__

    def _stub(cls, *a, **k):
        model, missing, mismatched, meta_left = load_s2v_model_from_gguf(
            gguf_path, config_json_path)
        if missing or mismatched or meta_left:
            raise RuntimeError(
                f"s2v GGUF load incomplete: missing={len(missing)} "
                f"mismatched={len(mismatched)} meta_leftovers={meta_left[:5]}")
        return model

    WanModel_S2V.from_pretrained = classmethod(_stub)
    try:
        import copy
        cfg = copy.deepcopy(WAN_CONFIGS["s2v-14B"])
        # T5 (11GB), VAE, and the umt5 tokenizer are shared with the T2V base —
        # don't duplicate them into the S2V dir. WanS2V does
        # os.path.join(checkpoint_dir, <cfg path>), and os.path.join with an
        # ABSOLUTE second arg returns that absolute path, so we point these at the
        # base while keeping checkpoint_dir = the S2V dir (which holds config.json
        # + the wav2vec2-large-xlsr-53-english/ subdir the audio encoder needs).
        _base = (Path(base_dir) if base_dir is not None
                 else Path(checkpoint_dir).parent / "Wan2.2-T2V-A14B")
        cfg.t5_checkpoint = str(_base / "models_t5_umt5-xxl-enc-bf16.pth")
        cfg.t5_tokenizer = str(_base / "google" / "umt5-xxl")
        cfg.vae_checkpoint = str(_base / "Wan2.1_VAE.pth")
        pipe = WanS2V(
            config=cfg, checkpoint_dir=str(checkpoint_dir), device_id=device_id,
            rank=0, t5_fsdp=False, dit_fsdp=False, use_sp=False, t5_cpu=True,
            init_on_cpu=True, convert_model_dtype=False)
    finally:
        WanModel_S2V.from_pretrained = classmethod(_orig)

    nm = pipe.noise_model
    nm.eval().requires_grad_(False)

    _force = force_dequant or os.environ.get("BASIWAN_FORCE_DEQUANT") == "1"
    if _force or not basiwan_kernel_available():
        why = ("BASIWAN_FORCE_DEQUANT" if _force else
               "kernel unavailable -> pure-torch dequant")
        print(f"[s2v-build] dequant path, no prepack ({why}); raw Q4 retained",
              flush=True)
    else:
        # Pack cache: prepack is ~280s; cache the packed layout (keyed on the
        # GGUF path) so re-runs are fast. Windows python writes the cache to a
        # NATIVE D:/ path (the 9p-corruption caveat was WSL-side torch.save only).
        os.environ.setdefault(
            "BASIWAN_PACK_CACHE_DIR",
            str(Path(__file__).resolve().parent.parent / "cache" / "marlin_packs"))
        try:
            from gguf_vendor.basiwan_pack_cache import cached_prepack_basiwan_weights
            n, from_cache = cached_prepack_basiwan_weights(nm, str(gguf_path))
            print(f"[s2v-build] prepacked {n} BASIWAN weights "
                  f"({'from cache' if from_cache else 'fresh; cached'})", flush=True)
        except Exception as e:  # noqa: BLE001
            n = prepack_basiwan_weights(nm)
            print(f"[s2v-build] prepacked {n} BASIWAN weights "
                  f"(cache unavailable: {e})", flush=True)

    if ffn_chunk and ffn_chunk > 0:
        install_s2v_ffn_chunk(nm, chunk_size=ffn_chunk)

    if block_swap_n is not None and 0 <= block_swap_n < len(nm.blocks):
        install_s2v_block_swap(nm, block_swap_n, device_id=device_id)
    elif resident:
        nm.to(torch.device(f"cuda:{device_id}"))
    return pipe


def install_s2v_block_swap(model, n, device_id=0):
    """Block-swap the single-expert S2V DiT so the 14B doesn't sit pinned against
    the 24GB wall (full-resident at real sizes thrashed: 2101 s/step from WDDM
    spill). Mirrors run_one_video_gguf._install_block_swap_gguf, minus the MoE
    boundary hook (S2V is single-expert): pin every NON-block submodule to GPU
    (audio_injector, casual_audio_encoder, frame_packer, cond_encoder, the
    embeddings, head — all small + on the hot path), keep blocks[:n] resident, and
    swap blocks[n:] CPU<->GPU around their forward. Sync only (the async ring is a
    WSL-fragile optimization; sync is the safe cross-platform default).

    BareGGMLLinear._apply moves the packed/raw GGML weight with .to(); the
    computed plain-attr tensors (freqs, zip_frame_buckets) are device-moved by the
    model's own forward, so they need no special handling here.
    """
    swap_dev = torch.device(f"cuda:{device_id}")
    off_dev = torch.device("cpu")
    # Flag so speech2video.generate skips its full-model .to(device)/.cpu() (which
    # would defeat block-swap by making ALL blocks resident at the loop peak).
    model._basiwan_block_swap = True
    for child_name, child in model.named_children():
        if child_name != "blocks":
            child.to(swap_dev)
    nb = len(model.blocks)
    for i, block in enumerate(model.blocks):
        block.to(swap_dev if i < n else off_dev)
    torch.cuda.synchronize()

    def _make_wrap(orig_fwd, idx, blk):
        # capture by arg (no late-binding); n is a stable free var. Mirrors the
        # proven run_one_video_gguf._basiwan_swap_forward SYNC path EXACTLY: the
        # synchronize() after each .to() + empty_cache() per block are LOAD-BEARING
        # -- without them the swapped-out blocks' GPU packed weights are not
        # released and ALL 38 swapped blocks accumulate on GPU (~9.5GB; profiled
        # via _s2v_mem_snapshot -> bare_gguf.py:_apply was the peak hog). The
        # per-block empty_cache costs wall time but is what frees the swap memory.
        def _swapped(*a, **kw):
            if idx >= n:
                blk.to(swap_dev, non_blocking=False)
                torch.cuda.synchronize()
            out = orig_fwd(*a, **kw)
            if idx >= n:
                blk.to(off_dev, non_blocking=False)
                torch.cuda.synchronize()
            else:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
            return out
        return _swapped

    for i, block in enumerate(model.blocks):
        block.forward = _make_wrap(block.forward, i, block)
    torch.cuda.synchronize()
    print(f"[s2v-build] block-swap: {n}/{nb} resident, {nb - n} swapped", flush=True)
    return nb - n


def install_s2v_ffn_chunk(model, chunk_size=4096, threshold=8000):
    """Chunk each block's FFN over the sequence dim. S2V's motion_frames=73
    inflate every chunk to ~60k tokens at 480p, so the FFN intermediate
    (seq x ffn_dim 13824) is the dominant activation transient (~1.7 GB/block at
    62k tokens) -> spills the 24GB wall even with block-swap. Splitting the seq
    into chunk_size pieces cuts that peak ~(seq/chunk_size)x. Mirrors
    run_one_video_gguf._install_chunked_ffn_gguf's _ChunkedFFN, with a shared
    output-buffer pool keyed by (shape,dtype,device) since Wan inference is
    Python-serialized (no concurrent block forwards)."""
    import torch.nn as nn
    pool = {}

    def _out_buf(x):
        key = (tuple(x.shape), str(x.dtype), str(x.device))
        buf = pool.get(key)
        if buf is None or buf.shape != x.shape:
            buf = torch.empty_like(x)
            pool[key] = buf
        return buf

    class _ChunkedFFN(nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig

        def forward(self, x):
            if x.dim() >= 2 and x.shape[-2] > threshold:
                n = x.shape[-2]
                out = _out_buf(x)
                for s in range(0, n, chunk_size):
                    e = min(s + chunk_size, n)
                    ch = x[..., s:e, :]
                    for layer in self.orig:
                        ch = layer(ch)
                    out[..., s:e, :].copy_(ch)
                return out
            return self.orig(x)

    n_wrap = 0
    for blk in model.blocks:
        if hasattr(blk, "ffn") and not isinstance(blk.ffn, _ChunkedFFN):
            blk.ffn = _ChunkedFFN(blk.ffn)
            n_wrap += 1
    print(f"[s2v-build] chunked-FFN: wrapped {n_wrap} blocks "
          f"(chunk={chunk_size}, threshold={threshold})", flush=True)
    return n_wrap
