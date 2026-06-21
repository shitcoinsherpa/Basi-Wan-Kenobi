#!/usr/bin/env python3
"""GGUF-loaded subprocess wrapper.

Status: SCAFFOLD. The full GGUF loader integration
is multi-hour work to adapt city96's ComfyUI nodes (vendored at
`tools/gguf_vendor/`) to a bare PyTorch context. This file documents
the integration path and provides argparse / env scaffolding so the
remaining work is just wiring up the loader call. It is NOT yet
end-to-end runnable.

To complete (the work that remains):

1. Implement `_build_pipe_gguf(gguf_path)` that:
   a. Reads the .gguf file via gguf.GGUFReader (vendored loader.py
      `gguf_sd_loader`).
   b. Maps GGUF tensor names → WanModel state_dict keys. QuantStack's
      naming follows the diffusers convention; our wan/modules/model.py
      WanModel uses the same naming, so this should be 1:1 for most
      tensors with a possible "model." prefix strip.
   c. For each WanModel module: instantiate empty, then load the
      state_dict, falling back to per-tensor dequant_tensor() for any
      quantized tensor that doesn't have a direct fp16/bf16 equivalent.
   d. Wrap nn.Linear weights with the vendored `GGMLTensor` so dequant
      happens at compute time, not load time.

2. Adapt the existing _build_pipe meta-shell pattern to skip BF16
   weight load (because GGUF IS the weight source). Mirror what
   prod_shape_bench.py:_build_pipe does for the FP8 path, but with
   GGUF as the source.

3. Reuse all P34 patches from `run_one_video.py` unchanged — they're
   dtype-agnostic and operate on nn.Module / nn.Linear surfaces.

4. Test with the QuantStack/Wan2.2-T2V-A14B-GGUF Q4_K_M and Q8_0
   variants. Expected outcomes:
   - Q4_K_M (~9.6 GB / expert) should fit p480_17f on 10-12 GB GPUs.
   - Q8_0 (~15 GB / expert) should match FP8 quality at higher VRAM cost.

5. Bench against the FP8 P34 baseline via tools/compare_bench.py.
   Quality bar per T2.1-B: Q8 within ±1% CLIP-T, Q5 ±2%, Q4 ±4%.

References:
  - vendored loader: tools/gguf_vendor/{loader.py, dequant.py, ops.py}
  - upstream reference: https://github.com/city96/ComfyUI-GGUF
  - weights: https://huggingface.co/QuantStack/Wan2.2-T2V-A14B-GGUF
"""
import argparse, json, os, sys, time
from pathlib import Path
_PROC_T0 = time.time()  # process-start reference for [startup-phase] prints

# Match run_one_video.py's CUDA env setup before any CUDA-touching import.
os.environ.setdefault("CUBLASLT_WORKSPACE_SIZE", "268435456")  # 256 MB
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16384:16")
# Windows port: `expandable_segments:True` is silently NON-FUNCTIONAL
# on Windows pip wheels — gated behind PYTORCH_C10_DRIVER_API_SUPPORTED, absent
# in pip builds (cu128 distributions included). We saw the warning at startup
# and ignored it. On Windows we instead use garbage_collection_threshold to
# reclaim cached blocks before OOM. DO NOT add max_split_size_mb — measured
# at p720_33f to add catastrophic per-Linear overhead (step 1: 798s
# vs 109s baseline; the cap forces per-call allocator bookkeeping that scales
# badly at large M).
# On Linux/WSL2: expandable_segments works correctly; keep the documented recipe.
if os.name == "nt":
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "garbage_collection_threshold:0.8",
    )
else:
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        "expandable_segments:True",
    )

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--model-type", choices=("t2v", "i2v", "vace", "s2v"), default="t2v",
                help="t2v (default) or i2v. i2v selects WAN_CONFIGS['i2v-A14B'] "
                     "(boundary 0.900, in_dim=36 experts) and expects --gguf-high/"
                     "--gguf-low to point at the I2V GGUF pair; expert config.json "
                     "is read from the GGUF's grandparent dir (e.g. "
                     "Wan2.2-I2V-A14B-GGUF/<name>/config.json) with base_dir fallback. "
                     "s2v (Wan2.2-S2V) is SINGLE-expert: pass the one S2V GGUF as "
                     "--gguf-high (--gguf-low is ignored), config.json from "
                     "Wan-AI/Wan2.2-S2V-14B staged next to it; audio-driven, needs "
                     "--audio + --image (reference).")
ap.add_argument("--gguf-high", required=True, type=Path,
                help="path to HighNoise expert .gguf file")
ap.add_argument("--gguf-low", required=True, type=Path,
                help="path to LowNoise expert .gguf file")
ap.add_argument("--vae", required=True, type=Path,
                help="path to Wan2.1 VAE safetensors (shared across quant levels)")
ap.add_argument("--base-dir", required=True, type=Path,
                help="path to the Wan2.2 base checkpoint dir (has high/low_noise_model/config.json + VAE)")
ap.add_argument("--lora-dir", type=Path, default=None,
                help="optional path to Wan2.2-Lightning LoRA dir (has high/low_noise_model.safetensors). "
                     "When set, the LoRA delta is merged onto the dequantized GGUF weights at load time "
                     "so the 4-step + CFG=1 recipe produces comparable quality to the FP8-prequant path. "
                     "Cost: weights become BF16 (~4× the Q4 size), so only viable on 24GB+ cards. "
                     "Future work: forward-time delta keeps Q4 footprint for 12GB-tier cards.")
ap.add_argument("--lora-strength", type=float, default=1.0,
                help="LoRA strength multiplier (default 1.0, matches FP8 prequant)")
ap.add_argument("--lora-mode", choices=("merge", "forward"), default="merge",
                help="merge: dequant Q4→BF16 at load + bake LoRA in (fast forward, ~4× the VRAM); "
                     "forward: keep Q4 weight + apply LoRA delta in BareGGMLLinear.forward "
                     "(preserves Q4 footprint — required for 12GB-tier cards).")
ap.add_argument("--ffn-chunk-size", type=int, default=0,
                help="when seq > 8000 tokens, chunk FFN along seq dim with this "
                     "chunk size (typical: 2048 or 4096). 0 = no chunking. "
                     "Required at p720_33f+ to avoid Q4-dequant peak OOM.")
ap.add_argument("--block-swap-n", type=int, default=-1,
                help="number of expert blocks to keep GPU-resident at all times. The "
                     "remaining 40-N blocks live on CPU and swap GPU↔CPU around each "
                     "forward call. Mirrors the run_one_video.py block-swap pattern but "
                     "works with BareGGMLLinear (relies on the _apply override that "
                     "follows weight/bias/LoRA across .to()). -1 (default) = no swap. "
                     "0 = aggressive; 4 = typical 24GB-card config; pick based on "
                     "free VRAM at shape.")
# These six become per-request inputs in --serve mode. Keep
# required at CLI time for the legacy path; validate manually after parsing.
ap.add_argument("--prompt", default=None)
ap.add_argument("--width", type=int, default=None)
ap.add_argument("--height", type=int, default=None)
ap.add_argument("--frames", type=int, default=None)
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--guide", type=float, default=1.0)
ap.add_argument("--out", default=None)
ap.add_argument("--meta", default=None)
ap.add_argument("--clip-t", action="store_true")
ap.add_argument("--image", default=None,
                help="[#373] I2V conditioning image (single-shot); the serve path "
                     "uses the JSON 'image' field. Use with --model-type i2v. For "
                     "--model-type s2v this is the reference character image.")
ap.add_argument("--audio", default=None,
                help="[S2V] driving audio (.wav) for model_type=s2v; serve path uses "
                     "the JSON 'audio' field.")
ap.add_argument("--s2v-dir", default=None,
                help="[S2V] checkpoint dir holding config.json + "
                     "wav2vec2-large-xlsr-53-english/ (T5/VAE come from --base-dir).")
ap.add_argument("--serve", action="store_true",
                help="Persistent-worker mode: after pipe build, emit "
                     "{\"event\":\"ready\"} and loop reading JSON request "
                     "lines from stdin. See _serve_loop() for protocol.")
args = ap.parse_args()
if not args.serve:
    _missing = [n for n in ("prompt", "width", "height", "frames", "out", "meta")
                if getattr(args, n) is None]
    if _missing:
        ap.error(f"required args missing (or pass --serve): "
                 f"{', '.join('--' + a for a in _missing)}")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "gguf_vendor"))

# Default-stack env for stacked-equivalent path
os.environ.setdefault("BASIWAN_SAGEATTN", "0")
os.environ.setdefault("BASIWAN_TAEHV_PARALLEL", "0")
os.environ.setdefault("BASIWAN_RMS_BF16", "1")
os.environ.setdefault("BASIWAN_LN_BF16", "1")

# Snapshot the user/start.js-provided VAE config
# BEFORE _do_one_generation mutates the env for shape-specific auto-gating.
# Restored on every request so a prior p720 request can't silently change
# the VAE path for a subsequent p480 request.
_INITIAL_BASIWAN_TAEHV_VAE = os.environ.get("BASIWAN_TAEHV_VAE")
_INITIAL_BASIWAN_VAE_TILING = os.environ.get("BASIWAN_VAE_TILING")
os.environ.setdefault("BASIWAN_NO_VAE_COMPILE", "1")
os.environ.setdefault("BASIWAN_NO_FP16_RECAST", "1")

import gc, json, time, random  # noqa: E402
import torch  # noqa: E402
import _basiwan_deep_profile as _dp  # noqa: E402  (no-op when env gate off)

# TF32 + cudnn.benchmark
# kept opt-in. Multi-prompt verification (5 prompts × {notf32, tf32}):
#   cat_boxing  +0.0040 composite (single-prompt anomaly that drove initial GG)
#   cat_surf    +0.0007
#   nyc_chase   -0.0015
#   dappled     +0.0035
#   ironman     -0.0046
# Avg delta +0.0004 — essentially zero. Wall regression +5% remains real.
# Initial cat_boxing +0.026 imaging boost did not generalize. Single-prompt
# benches are misleading; require multi-prompt to verify generalization.
# Opt-in via BASIWAN_TF32=1 or BASIWAN_CUDNN_BENCH=1.
if os.environ.get("BASIWAN_TF32") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True
if os.environ.get("BASIWAN_CUDNN_BENCH") == "1":
    torch.backends.cudnn.benchmark = True
# Windows port: cudnn.deterministic forces cuDNN to pick smaller-
# workspace conv plans (excludes high-workspace Winograd / FFT variants).
# This is the per-op equivalent of `expandable_segments` — instead of growing
# segments to fit large plans (which Windows can't do), shrink the plans to
# fit the fragmented allocator. Output stays within bf16 rounding tolerance.
# Opt-in via BASIWAN_CUDNN_DETERMINISTIC=1.
if os.environ.get("BASIWAN_CUDNN_DETERMINISTIC") == "1":
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from gguf_vendor.bare_gguf import (  # noqa: E402
    gguf_sd_loader, swap_linear_with_ggml, dequantize_tensor, is_quantized,
)


def _parse_lightning_lora(lora_path: Path) -> dict[str, dict[str, torch.Tensor]]:
    """Group a Lightning LoRA safetensors file into {target_name: {down, up, alpha}}.
    Strips the "diffusion_model." prefix so target_name matches WanModel paths."""
    from safetensors.torch import load_file
    sd_lora = load_file(str(lora_path))
    targets: dict[str, dict[str, torch.Tensor]] = {}
    for k, v in sd_lora.items():
        tail = k[len("diffusion_model."):] if k.startswith("diffusion_model.") else k
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if tail.endswith(suffix):
                name = tail[: -len(suffix)]
                kind = suffix.lstrip(".").replace(".weight", "")
                targets.setdefault(name, {})[kind] = v
                break
    return targets


def _walk_module(expert, dotted_name: str):
    """Walk `expert.{dotted_name}`, supporting numeric attrs for nn.Sequential.

    FFN-chunk wrappers (_ChunkedFFN / fused) store the
    original Sequential as `.orig`, so after pipe build `blocks.N.ffn` is a
    wrapper, not subscriptable — a numeric component like `ffn.0` would raise
    `TypeError: '_ChunkedFFN' object is not subscriptable`. Boot-time attach
    runs BEFORE wrapping so it never hits this; runtime set_lora re-attach
    does. Descend through `.orig` until we reach the subscriptable Sequential
    (the same Linear objects boot attached to, by reference)."""
    mod = expert
    for a in dotted_name.split("."):
        try:
            if a.isdigit():
                # Unwrap chunk/fused FFN wrappers to reach the Sequential.
                guard = 0
                while not hasattr(mod, "__getitem__") and hasattr(mod, "orig"):
                    mod = mod.orig
                    guard += 1
                    if guard > 4:
                        return None
                mod = mod[int(a)]
            else:
                mod = getattr(mod, a)
        except (AttributeError, IndexError, TypeError, KeyError):
            return None
    return mod


def _merge_lightning_into_gguf_expert(expert, lora_path: Path, strength: float = 1.0) -> int:
    """MERGE mode: dequant Q4 → BF16 + bake LoRA into the weight. Fast at
    forward time but ~4× the VRAM (BF16 weights replace Q4 weights)."""
    targets = _parse_lightning_lora(lora_path)
    merged = 0
    for name, parts in targets.items():
        down, up, alpha = parts.get("lora_down"), parts.get("lora_up"), parts.get("alpha")
        if down is None or up is None: continue
        mod = _walk_module(expert, name)
        if mod is None: continue
        w = getattr(mod, "weight", None)
        if w is None: continue
        if is_quantized(w):
            w_bf16 = dequantize_tensor(w, dtype=torch.bfloat16)
        else:
            w_bf16 = w.to(torch.bfloat16)
        rank = down.shape[0]
        scale = float(alpha.item() if alpha is not None else rank) / rank
        delta = (up.to(torch.float32) @ down.to(torch.float32)) * (scale * strength)
        merged_w = (w_bf16.to(torch.float32) + delta).to(torch.bfloat16)
        setattr(mod, "weight", merged_w)
        merged += 1
    return merged


def _is_lora_safetensors(path: Path) -> bool:
    """Detect whether a safetensors file holds LoRA deltas (`lora_down`/
    `lora_up`/`alpha` keys) or full model weights. Required for I-1.1 to
    distinguish Lightning-4 LoRAs (250928, Seko-V2.0) from full-model
    high-noise variants (250928-dyno-high)."""
    try:
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            for k in f.keys():
                if ".lora_down.weight" in k or ".lora_up.weight" in k:
                    return True
                # Only need to inspect a handful of keys to decide.
                break
        # Wider check: scan more keys for the lora_ pattern
        with safe_open(str(path), framework="pt") as f:
            for k in list(f.keys())[:50]:
                if ".lora_down.weight" in k or ".lora_up.weight" in k:
                    return True
        return False
    except Exception:
        return False


def _load_full_model_weights(expert, model_path: Path) -> int:
    """Load a FULL safetensors model into a
    BareGGMLLinear-wrapped expert, replacing the Q4_K weight with the
    full BF16 weights. Used for the Lightning-4 250928-dyno-high variant
    which ships as a full-parameter checkpoint (28.6 GB) instead of a
    LoRA delta.

    Pre-condition: `expert` was loaded from GGUF and has BareGGMLLinear
    on every Linear position. The full-model weights REPLACE the GGUF
    weights — we discard the Q4_K data for layers covered by the full
    model.

    Returns: count of weights replaced."""
    from safetensors import safe_open
    replaced = 0
    with safe_open(str(model_path), framework="pt") as f:
        keys = list(f.keys())
        for key in keys:
            if not key.endswith(".weight") and not key.endswith(".bias"):
                continue
            module_dotted = key.rsplit(".", 1)[0]
            param_name = key.rsplit(".", 1)[1]
            mod = _walk_module(expert, module_dotted)
            if mod is None:
                continue
            tensor = f.get_tensor(key).to(torch.bfloat16)
            if param_name == "weight":
                # For BareGGMLLinear: drop the GGUF quantized weight,
                # install a plain BF16 weight. Also drop any pre-packed
                # BASIWAN tensors (they're invalidated by the swap).
                # PyTorch Module requires nn.Parameter for `.weight`; bare
                # tensor assignment raises TypeError.
                if hasattr(mod, "weight"):
                    mod.weight = torch.nn.Parameter(tensor.detach(),
                                                    requires_grad=False)
                if hasattr(mod, "_basiwan_packed"):
                    mod._basiwan_packed = None
                replaced += 1
            elif param_name == "bias":
                if hasattr(mod, "bias"):
                    mod.bias = torch.nn.Parameter(tensor.detach(),
                                                  requires_grad=False)
                replaced += 1
    return replaced


def _attach_lightning_forward_lora(expert, lora_path: Path, strength: float = 1.0) -> int:
    """FORWARD mode: keep Q4 weight; attach (down, up, scale) to each matching
    BareGGMLLinear so the LoRA delta is computed at forward time. Preserves
    the Q4 VRAM footprint (~9.65GB per expert) — required for 12GB-tier cards.
    Cost: rank-128 matmul per Linear per forward call (~0.1% of main FLOPs)."""
    targets = _parse_lightning_lora(lora_path)
    attached = 0
    for name, parts in targets.items():
        down, up, alpha = parts.get("lora_down"), parts.get("lora_up"), parts.get("alpha")
        if down is None or up is None: continue
        mod = _walk_module(expert, name)
        if mod is None: continue
        # BareGGMLLinear holds these as plain attrs; nn.Linear etc. don't
        # have the slots — skip non-BareGGML targets gracefully.
        if not hasattr(mod, "lora_scale"): continue
        rank = down.shape[0]
        scale = float(alpha.item() if alpha is not None else rank) / rank
        # Store as BF16. Originally FP32 to preserve
        # precision through an FP32 delta matmul; but the delta matmul is
        # now BF16 (with cuBLAS internal FP32 accumulation) to keep the
        # peak (M, N) transient at 211 MB instead of 423 MB — necessary
        # to fit the 24 GB card alongside the block-swap residency.
        # BF16 LoRA storage halves total resident LoRA footprint from
        # ~3.6 GB to ~1.8 GB across both experts, which is the difference
        # between OOM-at-FFN[0] and steady-state diffusion.
        # BASIWAN_LORA_FP32=1 keeps LoRA tensors
        # in FP32 instead of casting to BF16. Tests whether BF16 LoRA precision
        # is the source of the today-vs-June2 imaging_quality regression
        # (-0.137 imaging across all today's variants vs June 2 ref).
        if os.environ.get("BASIWAN_LORA_FP32") == "1":
            mod.lora_down = down.to(torch.float32).detach()
            mod.lora_up = up.to(torch.float32).detach()
        else:
            mod.lora_down = down.to(torch.bfloat16).detach()
            mod.lora_up = up.to(torch.bfloat16).detach()
        # Store the rank-derived base scale separately so
        # _apply_lora_strength_to_pipe() can re-multiply at request time.
        # Without this, lora_scale fuses the user-tunable strength into the
        # immutable intrinsic scale, blocking per-request strength control.
        mod.lora_base_scale = scale
        mod.lora_scale = scale * strength
        attached += 1
    return attached


def _apply_lora_strength_to_pipe(pipe, strength: float) -> int:
    """Update forward-LoRA strength on both experts of a built pipe. Used in
    --serve mode so each request's lora_strength UI value is honored without
    rebuilding the pipe. Returns the number of modules updated."""
    n = 0
    for expert_name in ("low_noise_model", "high_noise_model"):
        expert = getattr(pipe, expert_name, None)
        if expert is None:
            continue
        for mod in expert.modules():
            if hasattr(mod, "lora_base_scale") and hasattr(mod, "lora_scale"):
                mod.lora_scale = mod.lora_base_scale * float(strength)
                n += 1
    return n


def _detach_all_forward_lora(pipe) -> int:
    """Clear forward-attached LoRA from EVERY module on both
    experts. Required before set_lora swaps in a new combo: re-attach only
    overwrites modules the NEW file targets, so modules the OLD combo touched
    but the new one doesn't would keep stale deltas. The forward gate is
    `lora_down is not None` (bare_gguf.py) — setting scale to 0 still pays the
    matmul, so we null the tensors. Returns count cleared."""
    n = 0
    for expert_name in ("low_noise_model", "high_noise_model"):
        expert = getattr(pipe, expert_name, None)
        if expert is None:
            continue
        for mod in expert.modules():
            if hasattr(mod, "lora_scale"):
                mod.lora_down = None
                mod.lora_up = None
                mod.lora_scale = 0.0
                if hasattr(mod, "lora_base_scale"):
                    mod.lora_base_scale = 0.0
                n += 1
    return n


def _fixup_lora_device(pipe) -> int:
    """Move each attached LoRA tensor onto its module's
    current device. Blocks [:block_swap_n] and non-block children are pinned
    GPU-resident at install and never `.to()` again, so a CPU-side attach
    leaves their LoRA on CPU — the forward self-heals per call (an H2D every
    step). One post-attach pass eliminates that. Truth source for device:
    the module's live weight (or its packed weight if the GGUF blob was
    freed). Returns count moved."""
    n = 0
    for expert_name in ("low_noise_model", "high_noise_model"):
        expert = getattr(pipe, expert_name, None)
        if expert is None:
            continue
        for mod in expert.modules():
            if getattr(mod, "lora_down", None) is None:
                continue
            dev = None
            w = getattr(mod, "weight", None)
            if w is not None:
                dev = w.device
            else:
                packed = getattr(mod, "_basiwan_packed", None)
                pw = getattr(packed, "weight", None) if packed is not None else None
                if pw is not None:
                    dev = pw.device
            if dev is None:
                continue
            if mod.lora_down.device != dev:
                mod.lora_down = mod.lora_down.to(dev)
                mod.lora_up = mod.lora_up.to(dev)
                n += 1
    return n


def _set_lora_on_pipe(pipe, lora_dir, strength: float = 1.0) -> dict:
    """Hot-swap the forward LoRA on a live serve worker
    without a restart. lora_dir=None clears all LoRA (revert to plain GGUF);
    otherwise attaches lora_dir/{low,high}_noise_model.safetensors. The combo
    has user strength baked in (alpha=rank → intrinsic scale 1.0), so
    `strength` here is normally 1.0. Returns per-expert attach counts."""
    from pathlib import Path as _P
    cleared = _detach_all_forward_lora(pipe)
    result = {"cleared": cleared, "low": 0, "high": 0}
    if lora_dir is None:
        return result
    ld = _P(lora_dir)
    for expert_name, key in (("low_noise_model", "low"),
                             ("high_noise_model", "high")):
        expert = getattr(pipe, expert_name, None)
        f = ld / f"{expert_name}.safetensors"
        if expert is None or not f.exists():
            continue
        result[key] = _attach_lightning_forward_lora(expert, f, strength)
    _fixup_lora_device(pipe)
    return result


def _build_pipe_gguf(gguf_high: Path, gguf_low: Path, base_dir: Path,
                     model_type: str = "t2v"):
    """Build WanT2V with GGUF experts. Mirrors prod_shape_bench._build_pipe
    structure but replaces the FP8 safetensors load with GGUF state-dict
    + BareGGMLLinear wrapping for runtime dequant."""
    # S2V is a SEPARATE engine (WanS2V, single-expert) with its own generate
    # signature. build_s2v_pipe_gguf does the full setup (prepack+cache, ffn-chunk,
    # block-swap, tiled VAE), so main() must SKIP its own ffn-chunk/block-swap
    # installs for s2v (gated below). Widescreen needs tiled enc+dec -> force it on.
    if model_type == "s2v":
        os.environ["BASIWAN_VAE_TILING"] = "1"   # widescreen needs tiled enc+dec
        os.environ["BASIWAN_TAEHV_VAE"] = "0"    # s2v uses the real tiled Wan2.1 VAE
        from s2v_loader import build_s2v_pipe_gguf
        from wan.configs import SIZE_CONFIGS
        if not args.s2v_dir:
            raise RuntimeError("--model-type s2v requires --s2v-dir (config.json + "
                               "wav2vec2 dir)")
        _s2vd = Path(args.s2v_dir)
        _bsn = args.block_swap_n if args.block_swap_n >= 0 else 2
        _ffn = args.ffn_chunk_size if args.ffn_chunk_size > 0 else 4096
        pipe = build_s2v_pipe_gguf(
            gguf_path=gguf_high, config_json_path=_s2vd / "config.json",
            checkpoint_dir=_s2vd, base_dir=base_dir, device_id=0,
            block_swap_n=_bsn, ffn_chunk=_ffn)
        print(f"[runner-gguf] S2V pipe built (block_swap_n={_bsn}, ffn_chunk={_ffn}, "
              f"tiled VAE on)", flush=True)
        return pipe, SIZE_CONFIGS
    import wan
    from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
    from wan.modules.model import WanModel
    from wan.modules.model import rope_params

    # 1. Stub WanModel.from_pretrained so the WanT2V wrapper instantiates
    #    each expert on meta device — no BF16 load.
    _orig = WanModel.from_pretrained.__func__
    def _meta_from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        root = Path(pretrained_model_name_or_path)
        sub = kwargs.get("subfolder")
        if sub:
            root = root / sub
        cfg_path = root / "config.json"
        with cfg_path.open() as f:
            cfg = json.load(f)
        cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
        with torch.device("meta"):
            m = cls(**cfg)
        return m
    WanModel.from_pretrained = classmethod(_meta_from_pretrained)
    try:
        # i2v-A14B differs only in boundary (0.900), sample_shift (5.0) and
        # guide defaults — same T5/VAE/dims; WanT2V consumes it generically.
        # VACE-Fun is a T2V-derived model (same T5/VAE/boundary, +control
        # branch) — use the t2v pipe config; the vace submodules are injected
        # into the per-expert mcfg below.
        _cfg_key = "t2v" if model_type == "vace" else model_type
        cfg = WAN_CONFIGS[f"{_cfg_key}-A14B"]
        print(f"[startup-phase] imports+wan_module {time.time() - _PROC_T0:.1f}s "
              f"since process start (model_type={model_type})", flush=True)
        _t_pipe = time.time()
        pipe = wan.WanT2V(
            config=cfg, checkpoint_dir=str(base_dir), device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_sp=False,
            t5_cpu=True, convert_model_dtype=True, init_on_cpu=True)
        print(f"[startup-phase] wan_t2v_ctor {time.time() - _t_pipe:.1f}s "
              f"(T5+VAE+compile inside)", flush=True)
    finally:
        WanModel.from_pretrained = classmethod(_orig)

    # 2. Replace meta-shell experts with GGUF state dicts.
    for name, gguf_path in [("high_noise_model", gguf_high),
                            ("low_noise_model",  gguf_low)]:
        setattr(pipe, name, None)
        gc.collect()
        # QuantStack/Wan2.2-T2V-A14B-GGUF uses bare WanModel naming (no
        # "model." or "model.diffusion_model." prefix) — verified,
        # 1095/1095 key match. handle_prefix=None is the right value.
        _t_sd = time.time()
        sd, extra = gguf_sd_loader(str(gguf_path), handle_prefix=None)
        print(f"[startup-phase] {name}: gguf_sd_loader {time.time() - _t_sd:.1f}s",
              flush=True)
        arch = extra.get("arch_str")
        if arch is not None and arch != "wan":
            print(f"[gguf] WARN: expected arch 'wan', got {arch!r}", flush=True)

        # Instantiate a fresh WanModel on meta and materialize structure.
        # For i2v, the expert config (in_dim=36, model_type=i2v) lives next
        # to the I2V GGUFs (Wan2.2-I2V-A14B-GGUF/<name>/config.json — staged
        # from Wan-AI/Wan2.2-I2V-A14B); base_dir holds the T2V configs
        # (in_dim=16) which would build a skeleton the I2V tensors don't fit.
        cfg_path = base_dir / name / "config.json"
        if model_type == "i2v":
            _i2v_cfg = Path(gguf_path).parent.parent / name / "config.json"
            if _i2v_cfg.exists():
                cfg_path = _i2v_cfg
            else:
                raise FileNotFoundError(
                    f"i2v expert config not found at {_i2v_cfg}; stage the "
                    f"config.json files from Wan-AI/Wan2.2-I2V-A14B next to "
                    f"the GGUFs (in_dim=36 — the T2V config cannot be used)")
        with cfg_path.open() as f:
            mcfg = json.load(f)
        mcfg = {k: v for k, v in mcfg.items() if not k.startswith("_")}
        # VACE-Fun: the merged QuantStack GGUF carries 8 vace_blocks +
        # vace_patch_embedding inline (verified) but ships no config.json, so the
        # base T2V mcfg is missing the vace config. Inject vace_layers (official
        # Wan2.2-Fun-VACE-A14B config: [0,5,10,15,20,25,30,35]) + vace_in_dim=96
        # so WanModel builds the vace submodules; the generic assignment loop
        # below then loads every vace_*/vace_patch_embedding tensor by name.
        if model_type == "vace":
            mcfg["vace_layers"] = [0, 5, 10, 15, 20, 25, 30, 35]
            mcfg["vace_in_dim"] = 96
        with torch.device("meta"):
            expert = WanModel(**mcfg)
        # Move structure off meta (placeholders) — gives every parameter a
        # real (empty) storage so attribute access works.
        expert = expert.to_empty(device="cpu")

        # Swap nn.Linear → BareGGMLLinear BEFORE assigning GGUF tensors.
        # nn.Parameter requires floating-point dtype and would reject the
        # packed uint8 Q4_K blocks. BareGGMLLinear stores the GGMLTensor as
        # a plain attribute and dequantizes on every forward.
        n_swapped = swap_linear_with_ggml(expert)
        print(f"[gguf] {name}: swapped {n_swapped} Linears for GGML dequant", flush=True)

        # Manual assignment loop: bypass load_state_dict (which would try
        # to put GGMLTensor into Parameter slots and assert dtype/shape).
        # Strategy:
        #   - For BareGGMLLinear targets (.weight is NOT a Parameter slot):
        #     direct setattr; the dequant happens at forward time.
        #   - For Parameter slots (LayerNorm/RMSNorm .weight, biases, etc.):
        #     replace _parameters[leaf] with nn.Parameter(tensor, requires_grad=False).
        #     The GGUF loader has already dequantized 1D BF16 tensors to F32.
        #   - For Buffer slots (none in WanModel except .freqs which we
        #     rematerialize manually): not handled by this loop.
        import torch.nn as _nn
        missing, mismatched = [], []
        for key, tensor in sd.items():
            mod = expert
            attrs = key.split(".")
            try:
                for a in attrs[:-1]:
                    mod = getattr(mod, a) if not a.isdigit() else mod[int(a)]
            except (AttributeError, IndexError):
                missing.append(key); continue
            leaf = attrs[-1]
            try:
                if leaf in mod._parameters:
                    # The leaf is registered as a Parameter slot. Wrap the
                    # (already-dequantized) tensor in an inference Parameter.
                    mod._parameters[leaf] = _nn.Parameter(
                        tensor.detach() if hasattr(tensor, "detach") else tensor,
                        requires_grad=False,
                    )
                elif hasattr(mod, leaf):
                    # Plain attribute (BareGGMLLinear.weight/.bias case) —
                    # assign the GGMLTensor directly; dequant at forward.
                    setattr(mod, leaf, tensor)
                else:
                    mismatched.append(key)
            except Exception as e:
                mismatched.append(f"{key}: {type(e).__name__}: {e}")
        if missing:
            print(f"[gguf] {name}: {len(missing)} keys had no target module "
                  f"(showing 3): {missing[:3]}", flush=True)
        if mismatched:
            print(f"[gguf] {name}: {len(mismatched)} keys could not assign "
                  f"(showing 3): {mismatched[:3]}", flush=True)

        # WanModel.freqs is computed not stored; rematerialize on CPU.
        if hasattr(expert, "freqs") and getattr(expert.freqs, "is_meta", False):
            d = expert.dim // expert.num_heads
            expert.freqs = torch.cat([
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ], dim=1)
        # Optionally apply Lightning LoRA OR a full-model dyno variant.
        # Three modes:
        #   - full safetensors (e.g. 250928-dyno-high) → _load_full_model_weights
        #   - LoRA safetensors merge → _merge_lightning_into_gguf_expert
        #   - LoRA safetensors forward → _attach_lightning_forward_lora
        if args.lora_dir is not None:
            lora_path = args.lora_dir / f"{name}.safetensors"
            if not lora_path.exists():
                print(f"[gguf] {name}: LoRA path {lora_path} missing — skipping",
                      flush=True)
            elif not _is_lora_safetensors(lora_path):
                # Full-model checkpoint (e.g. 250928-dyno high-noise).
                t0 = time.time()
                n_replaced = _load_full_model_weights(expert, lora_path)
                print(f"[gguf] {name}: LOADED {n_replaced} full-model weights from "
                      f"{lora_path.name} in {time.time()-t0:.1f}s (replaces Q4)",
                      flush=True)
            elif args.lora_mode == "merge":
                t0 = time.time()
                n_merged = _merge_lightning_into_gguf_expert(
                    expert, lora_path, strength=args.lora_strength)
                print(f"[gguf] {name}: MERGED {n_merged} LoRA targets into dequant "
                      f"weights in {time.time()-t0:.1f}s (strength={args.lora_strength}, "
                      f"BF16 weights ~4× Q4 size)", flush=True)
            else:  # forward
                t0 = time.time()
                n_att = _attach_lightning_forward_lora(
                    expert, lora_path, strength=args.lora_strength)
                print(f"[gguf] {name}: ATTACHED {n_att} forward-time LoRA targets "
                      f"in {time.time()-t0:.1f}s (strength={args.lora_strength}, "
                      f"Q4 weights preserved for low-VRAM cards)", flush=True)

        expert.eval().requires_grad_(False)

        # Eagerly pre-pack BASIWAN Q4_K/Q6_K weights while the
        # expert is still entirely on CPU. The lazy-pack-on-first-forward
        # path pays a heavy D2H staging cost once block-swap has put the
        # block on GPU under VRAM pressure — measured: OOM at FFN[0] of
        # block 0 on RTX 4090 24 GB at p480_17f after only 8 successful
        # packs. CPU-side packing is bandwidth-trivial and avoids the
        # transient CUDA workspace need.
        # The pack cache writes to a local disk path (~/.cache/marlin_packs,
        # overridable via BASIWAN_PACK_CACHE_DIR in _env.sh) and produces
        # bit-identical output to a fresh pack. Avoid a network- or 9p-mounted
        # cache dir: writing packed tensors there can corrupt in-memory tensor
        # state during torch.save. Safe to enable by default.
        from gguf_vendor.bare_gguf import prepack_basiwan_weights, basiwan_kernel_available
        t0 = time.time()
        # Prepack produces the CUDA kernel's packed layout AND frees the
        # raw Q4/Q6 weight. When the kernel is UNAVAILABLE (AMD ROCm / Apple MPS /
        # no build toolchain) or BASIWAN_FORCE_DEQUANT=1, we must NOT prepack — the
        # pure-torch dequant fallback needs the raw weight retained.
        _force_dq = os.environ.get("BASIWAN_FORCE_DEQUANT") == "1"
        if os.environ.get("BASIWAN_SKIP_PREPACK") == "1" or _force_dq or not basiwan_kernel_available():
            _why = ("BASIWAN_SKIP_PREPACK=1" if os.environ.get("BASIWAN_SKIP_PREPACK") == "1"
                    else "BASIWAN_FORCE_DEQUANT=1" if _force_dq
                    else "kernel unavailable -> pure-torch dequant fallback (W14)")
            print(f"[gguf] {name}: SKIPPING BASIWAN prepack ({_why}); raw Q4/Q6 "
                  f"weights retained for the dequant path", flush=True)
            n_packed = 0
            cache_tag = f"skipped ({_why})"
        elif os.environ.get("BASIWAN_USE_PACK_CACHE") == "1":
            from gguf_vendor.basiwan_pack_cache import cached_prepack_basiwan_weights
            gguf_path = args.gguf_high if name == "high_noise_model" else args.gguf_low
            n_packed, from_cache = cached_prepack_basiwan_weights(expert, gguf_path)
            cache_tag = "from cache" if from_cache else "freshly packed; cache written"
        else:
            n_packed = prepack_basiwan_weights(expert)
            cache_tag = "freshly packed (cache disabled)"
        print(f"[gguf] {name}: {n_packed} BASIWAN weights {cache_tag} in "
              f"{time.time() - t0:.1f}s", flush=True)

        setattr(pipe, name, expert)
        gc.collect()
    return pipe, SIZE_CONFIGS


def _install_chunked_ffn_gguf(pipe, chunk_size: int, threshold: int = 8000) -> int:
    """Wrap each block's `ffn` (nn.Sequential of Linear+GELU+Linear) so that
    at seq > threshold tokens, the seq dim is split into chunk_size pieces
    and processed one at a time. Reduces per-call peak by ~(seq/chunk_size).
    Required at p720_33f+ on the GGUF runner to avoid Q4-dequant + activation
    matmul OOM. Mirrors run_one_video.py's _ChunkedFFN pattern."""
    import torch.nn as _nn
    fused_wrapper_cls = None
    use_fused = os.environ.get("BASIWAN_FUSED_FFN", "").strip().lower() not in {
        "",
        "0",
        "false",
        "no",
        "off",
    }
    if use_fused:
        try:
            from gguf_vendor.fused_ffn_triton import FusedChunkedQ4FFN

            fused_wrapper_cls = FusedChunkedQ4FFN
        except Exception as e:
            print(f"[runner-gguf] fused-FFN unavailable ({e}); falling back to chunked-FFN",
                  flush=True)

    # Process-singleton pool of FFN output buffers, keyed by (shape, dtype, device).
    # Replaces the per-instance _out_buf which became the dominant pile at
    # p720_17f: 80 ChunkedFFN instances × ~489 MB each (M=18000, N=13568, bf16).
    # With the pool, all instances that emit the SAME shape share one backing
    # tensor — serially reused since Wan inference is Python-serialized
    # (no concurrent block forwards). Mirrors the BasiwanRuntimePool pattern.
    _ffn_out_pool: dict = {}

    def _get_pooled_out_buf(x: torch.Tensor) -> torch.Tensor:
        key = (tuple(x.shape), str(x.dtype), str(x.device))
        buf = _ffn_out_pool.get(key)
        if buf is None or buf.shape != x.shape:
            buf = torch.empty_like(x)
            _ffn_out_pool[key] = buf
        return buf

    class _ChunkedFFN(_nn.Module):
        def __init__(self, orig):
            super().__init__()
            self.orig = orig
            # Per-instance sticky buffer eliminated; the pool covers reuse
            # across ALL ChunkedFFN instances + ALL forward calls of the same
            # shape. At p720_17f: 80 instances × 489 MB → 1 buffer × 489 MB.

        def forward(self, x):
            if x.dim() >= 2 and x.shape[-2] > threshold:
                n = x.shape[-2]
                out = _get_pooled_out_buf(x)
                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    ch = x[..., start:end, :]
                    for layer in self.orig:
                        ch = layer(ch)
                    out[..., start:end, :].copy_(ch)
                return out
            return self.orig(x)

    n_wrap = 0
    already_wrapped_types = (_ChunkedFFN,)
    if fused_wrapper_cls is not None:
        already_wrapped_types = (_ChunkedFFN, fused_wrapper_cls)
    for expert_name in ("low_noise_model", "high_noise_model"):
        expert = getattr(pipe, expert_name, None)
        if expert is None or not hasattr(expert, "blocks"):
            continue
        for blk in expert.blocks:
            if hasattr(blk, "ffn") and not isinstance(blk.ffn, already_wrapped_types):
                if fused_wrapper_cls is not None:
                    blk.ffn = fused_wrapper_cls(blk.ffn, chunk_size=chunk_size, threshold=threshold)
                else:
                    blk.ffn = _ChunkedFFN(blk.ffn)
                n_wrap += 1
    pipe._gguf_ffn_wrapper_label = "fused-FFN" if fused_wrapper_cls is not None else "chunked-FFN"
    return n_wrap


def _setup_async_ring_buffer(pipe, swap_n):
    """Probe pin budget and create a rolling-pin RingBuffer for async H2D.

    Called from main() AFTER _install_block_swap_gguf so the install
    function's bytecode stays clean of ring-buffer references (which on
    PyTorch 2.11+cu130/WSL2 trigger a bogus install-time CUDA OOM).
    """
    # Lazy imports — these modules import only torch + stdlib.
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(__file__))
    from _basiwan_pin_probe import probe_pin_budget
    from _basiwan_ring_buffer import RingBuffer, measure_block_bytes

    # Find a sample swap block to measure size.
    sample_block = None
    for name in ("low_noise_model", "high_noise_model"):
        exp = getattr(pipe, name, None)
        if exp is None or not hasattr(exp, 'blocks'):
            continue
        if len(exp.blocks) > swap_n:
            sample_block = exp.blocks[swap_n]
            break
    if sample_block is None:
        print("[runner-gguf] ring-buffer: no swap blocks found — async disabled", flush=True)
        return
    slot_bytes = measure_block_bytes(sample_block)
    # Round up to nearest MB for stable measurement.
    slot_bytes = ((slot_bytes + (1 << 20) - 1) // (1 << 20)) * (1 << 20)
    # Probe how many slots we can safely hold. Cap default varies by
    # platform: WSL2 dxgkrnl has a count-based descriptor-table cap that
    # trips with ~50 small cudaHostRegister calls per block even at K=1.
    # On native Linux /
    # native Windows / macOS this cap doesn't exist, so K=4 (~1 GB
    # pinned) is fine. Pinokio targets all three native platforms so
    # the ring path is the ship; WSL2 is a local-dev fallback.
    _is_wsl2 = False
    try:
        with open("/proc/version") as _f:
            _v = _f.read().lower()
        _is_wsl2 = ("microsoft" in _v) or ("wsl" in _v)
    except FileNotFoundError:
        pass
    _default_k_max = "0" if _is_wsl2 else "4"
    max_k = int(os.environ.get("BASIWAN_BLOCK_SWAP_RING_K_MAX",
                                _default_k_max))
    k = probe_pin_budget(slot_bytes, max_slots=max_k)
    if k == 0:
        print(f"[runner-gguf] ring-buffer: probe K=0 — async pinning unavailable, "
              f"slot={slot_bytes/(1<<20):.0f}MB", flush=True)
        return
    cudart = torch.cuda.cudart()
    pipe._swap_ring = RingBuffer(k, cudart)
    print(f"[runner-gguf] ring-buffer: K={k} slots × {slot_bytes/(1<<20):.0f}MB = "
          f"{k*slot_bytes/(1<<30):.2f}GB pinned", flush=True)
    _dp.assert_cuda_clean("after-ring-setup")
    _dp.report_pinned("after-ring-setup")


def _pin_block_to_cpu_call(block):
    """Page-lock CPU-resident leaf tensors in `block` via cudaHostRegister.
    No copy, no new allocation — just kernel page-flag flips on existing
    storage. NOTE: BareGGMLLinear plain attrs are walked via duck-typing
    (hasattr checks) to avoid a from-import that triggered an install-time
    OOM trigger on PyTorch 2.11+cu130/WSL2."""
    _cudart = torch.cuda.cudart()
    def _try_register(t):
        if t is None or t.device.type != 'cpu' or t.is_pinned():
            return
        try:
            s = t.untyped_storage()
            _cudart.cudaHostRegister(s.data_ptr(), s.nbytes(), 0)
        except Exception:
            pass
    for p in block.parameters(recurse=True):
        _try_register(p.data)
    for b in block.buffers(recurse=True):
        _try_register(b.data)
    for mod in block.modules():
        # Duck-typed BareGGMLLinear detection — avoids module-level import.
        if hasattr(mod, '_basiwan_packed'):
            _try_register(getattr(mod, 'weight', None))
            _try_register(getattr(mod, 'bias', None))
            _try_register(getattr(mod, 'lora_down', None))
            _try_register(getattr(mod, 'lora_up', None))
            mp = mod._basiwan_packed
            if mp is not None:
                _try_register(getattr(mp, 'weight', None))
                _try_register(getattr(mp, 'weight_hi', None))
                _try_register(getattr(mp, 'scales', None))
                _try_register(getattr(mp, 'mins', None))


def _basiwan_swap_forward(pipe, blk, idx, n, n_blocks, exp_name, blocks_ref,
                          swap_dev, off_dev, orig_fwd, args, kwargs, async_mode):
    """Unified block-swap forward — dispatched from the make_wrap closure.
    Module-level so the closure stays minimal (avoids bytecode-
    density install-time OOM trigger on PyTorch 2.11+cu130/WSL2)."""
    if not async_mode:
        if idx >= n:
            blk.to(swap_dev, non_blocking=False)
            torch.cuda.synchronize()
        out = orig_fwd(*args, **kwargs)
        if idx >= n:
            blk.to(off_dev, non_blocking=False)
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize()
        torch.cuda.empty_cache()
        return out

    cur_stream = torch.cuda.current_stream()
    swap_stream = pipe._swap_stream
    prefetch_ev = pipe._swap_prefetch_events.pop((exp_name, idx), None)
    if prefetch_ev is not None:
        cur_stream.wait_event(prefetch_ev)
    elif idx >= n and pipe._swap_block_state.get((exp_name, idx)) != 'gpu':
        blk.to(swap_dev, non_blocking=False)
    pipe._swap_block_state[(exp_name, idx)] = 'gpu'

    next_idx = idx + 1
    if next_idx < n_blocks and next_idx >= n:
        if pipe._swap_block_state.get((exp_name, next_idx)) == 'cpu':
            d2h_ev = pipe._swap_pending_d2h_events.pop((exp_name, next_idx), None)
            # Pin the source block via the rolling-window ring buffer so the
            # cudaMemcpyAsync becomes a true overlapped DMA copy. If ring is
            # absent (probe found K=0, e.g., on memlock-restricted hosts),
            # the copy falls back to PyTorch's staged path — still correct,
            # just no overlap.
            ring = getattr(pipe, "_swap_ring", None)
            if ring is not None:
                _dp_pre_pin = _dp.scan_pinned_bytes() if _dp.deep_enabled() else -1
                ring.ensure_pinned((exp_name, next_idx), blocks_ref[next_idx])
                _dp.assert_cuda_clean(f"after-pin-{exp_name}-{next_idx}")
                if _dp.deep_enabled() and _dp_pre_pin >= 0:
                    _dp_post_pin = _dp.scan_pinned_bytes()
                    print(f"[basiwan-prof] pinned delta blk={next_idx}: "
                          f"{(_dp_post_pin - _dp_pre_pin) / (1 << 20):.0f}MB "
                          f"total={_dp_post_pin / (1 << 30):.2f}GB", flush=True)
            with torch.cuda.stream(swap_stream):
                if d2h_ev is not None:
                    swap_stream.wait_event(d2h_ev)
                def _do_h2d(_b=blocks_ref[next_idx], _d=swap_dev):
                    return _b.to(_d, non_blocking=True)
                _dp.detect_sync_degradation(
                    f"H2D-{exp_name}-{next_idx}", _do_h2d)
                ev = torch.cuda.Event()
                ev.record(swap_stream)
            _dp.assert_cuda_clean(f"after-prefetch-{exp_name}-{next_idx}")
            pipe._swap_prefetch_events[(exp_name, next_idx)] = ev
            pipe._swap_block_state[(exp_name, next_idx)] = 'pending_h2d'

    out = orig_fwd(*args, **kwargs)

    if idx >= n:
        fwd_done = torch.cuda.Event()
        fwd_done.record(cur_stream)
        with torch.cuda.stream(swap_stream):
            swap_stream.wait_event(fwd_done)
            blk.to(off_dev, non_blocking=True)
            d2h_done = torch.cuda.Event()
            d2h_done.record(swap_stream)
        pipe._swap_pending_d2h_events[(exp_name, idx)] = d2h_done
        pipe._swap_block_state[(exp_name, idx)] = 'cpu'

    # After the LAST swap block of this expert's forward, drain the
    # PyTorch caching host allocator. Without this, .to(cpu)'s stage
    # pages leak into the cache and host RSS grows unbounded — Linux
    # OOM-kills at step 3 on 36 GB host RAM. Drain returns pages to OS.
    # Cost: forces a sync of in-flight D2H, but those would block the
    # next expert anyway at the MoE boundary.
    if idx == n_blocks - 1 and idx >= n:
        try:
            swap_stream.synchronize()
            torch._C._host_emptyCache()
        except Exception:
            pass
    return out


def _install_block_swap_gguf(pipe, n: int) -> None:
    """Install block-swap on the GGUF experts. Mirrors run_one_video.py's
    pattern: pin non-block submodules (text_emb, time_emb, head, etc.) to
    GPU permanently; keep blocks[:n] resident; offload blocks[n:] to CPU
    and swap them in/out around each forward call.

    Adding the case-B bytecode trigger (LOAD_GLOBAL on
    _pin_block_to_cpu_call inside a local def) to see if the function is
    actually invoked before the OOM, or if its mere bytecode reference
    triggers something else. Standalone repro (without Wan/GGUF) does NOT
    show the OOM — so the bug requires GGUF-specific runtime."""
    swap_dev = torch.device("cuda:0")
    off_dev = torch.device("cpu")

    # Async mode gate. cudaHostRegister-pinned async block-swap measured
    # 24% per-step gain. Keep make_wrap closures SMALL to avoid the
    # bytecode-density trigger that produced bogus CUDA OOM
    # on the SECOND expert's first non-block submodule .to(cuda) when
    # closures held inline async stream/event references.
    _async_mode = (os.environ.get("BASIWAN_BLOCK_SWAP_ASYNC") == "1") and torch.cuda.is_available()

    # Initialize per-pipe async state ALWAYS (even in sync mode) so the
    # MoE-boundary hook doesn't AttributeError. swap_stream is created
    # lazily after the first expert's residency is set, to avoid the
    # cublasLt-workspace OOM during initial submodule migration.
    pipe._swap_stream = None
    pipe._swap_prefetch_events = {}
    pipe._swap_pending_d2h_events = {}
    pipe._swap_block_state = {}

    counts = {}
    for name in ("low_noise_model", "high_noise_model"):
        expert = getattr(pipe, name, None)
        if expert is None or not hasattr(expert, "blocks"):
            continue
        for child_name, child in expert.named_children():
            if child_name != "blocks":
                child.to(swap_dev, non_blocking=False)
        # BISECT: globals lookup still seems to trigger. Skip pinning,
        # see if full _basiwan_swap_forward body alone OOMs.
        for i, block in enumerate(expert.blocks):
            block.to(swap_dev if i < n else off_dev, non_blocking=False)
            pipe._swap_block_state[(name, i)] = 'gpu' if i < n else 'cpu'
        torch.cuda.synchronize()
        # Lazy stream alloc — only after both experts done.
        if _async_mode and pipe._swap_stream is None:
            torch.cuda.empty_cache()
            pipe._swap_stream = torch.cuda.Stream(device=swap_dev)
        blocks_ref = expert.blocks
        n_blocks = len(blocks_ref)
        for i, block in enumerate(expert.blocks):
            orig = block.forward
            block._orig_forward_pre_swap = orig
            block._block_idx = i
            # Closure body is intentionally minimal — dispatches to a
            # module-level function. Keeps make_wrap closures small (avoids
            # the bytecode-density trigger that produced bogus
            # CUDA OOM at install on PyTorch 2.11+cu130/WSL2).
            # Default-arg capture preserves the closure-bug fix for MoE
            # boundary (Ruff B023 / pylint W0640 would catch the pattern).
            def make_wrap(blk, idx, orig_fwd,
                          exp_name=name,
                          blocks_ref=blocks_ref,
                          n_blocks=n_blocks):
                def _swapped_forward(*a, **kw):
                    return _basiwan_swap_forward(
                        pipe, blk, idx, n, n_blocks, exp_name, blocks_ref,
                        swap_dev, off_dev, orig_fwd, a, kw, _async_mode)
                return _swapped_forward
            block.forward = make_wrap(block, i, orig)
        counts[name] = len(expert.blocks)
    print(f"[runner-gguf] block-swap installed: "
          f"N={n} resident, "
          f"{counts.get('low_noise_model', 0) - n}+{counts.get('high_noise_model', 0) - n} "
          f"swap blocks", flush=True)

    # Patch _prepare_model_by_name to re-enforce block-swap
    # residency after Wan's boundary-crossing `to(self.device)`. The pipe moves
    # the entire incoming expert to GPU at MoE boundary (step 2→3 for
    # Lightning-4), defeating block-swap for one step. Each step thereafter
    # migrates blocks[N:] off via the swap wrapper, but the boundary PEAK is
    # 40 blocks resident = ~3-4 GB extra residency at p720_33f → OOM in the
    # transient pile that survives 2 steps elsewhere.
    #
    # Fix: after `.to(self.device)`, immediately migrate blocks[N:] back to
    # CPU. Same invariant block-swap installed. Costs one extra D2H burst at
    # boundary (saves multiple smaller D2H's that the wrapper would do).
    _orig_prepare = pipe._prepare_model_by_name

    def _prepare_then_reapply_swap(required_name, offload_name=None, offload_previous=False):
        model = _orig_prepare(required_name, offload_name, offload_previous=offload_previous)
        if hasattr(model, "blocks"):
            if _async_mode and getattr(pipe, "_swap_stream", None) is not None:
                pipe._swap_stream.synchronize()
                pipe._swap_prefetch_events.clear()
                pipe._swap_pending_d2h_events.clear()
                ring = getattr(pipe, "_swap_ring", None)
                if ring is not None:
                    ring.drain()
            # BISECT: simplified MoE hook — no pin reference
            for i, block in enumerate(model.blocks):
                if i >= n:
                    block.to(off_dev, non_blocking=False)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if _async_mode:
                for i in range(len(model.blocks)):
                    pipe._swap_block_state[(required_name, i)] = 'gpu' if i < n else 'cpu'
        return model

    pipe._prepare_model_by_name = _prepare_then_reapply_swap
    print(f"[runner-gguf] boundary-crossing residency hook installed", flush=True)


_DEPTH_MODEL_CACHE = {}


def _get_depth_model():
    """Lazy-load Depth-Anything-V2-Small once per worker, kept on CPU and
    run on CPU. The Dinov2 backbone uses raw nn.Linear; once the BASIWAN GGUF
    pipe is resident it has set deterministic/cuBLAS state that makes those
    F.linear calls trip the cuBLASLt workspace-heuristic bug (bogus 256 GiB
    alloc) on this Windows torch build — bisected to depth.py:extract_depth, NOT
    the VACE forward (see memory/cublaslt_flinear_bug). Extracting on CPU
    sidesteps the GPU heuristic entirely; the model is ~100MB and 17 frames cost
    ~tens of seconds, negligible vs the ~7-min 50-step generate."""
    if "m" not in _DEPTH_MODEL_CACHE:
        from basi import depth as _dep
        _DEPTH_MODEL_CACHE["m"], _DEPTH_MODEL_CACHE["p"] = _dep.load_depth_model(
            cache_dir=os.environ.get("BASIWAN_DEPTH_CACHE_DIR"), device="cpu")
    return _DEPTH_MODEL_CACHE["m"], _DEPTH_MODEL_CACHE["p"]


def _plain_tensor(t):
    """Coerce a generated video tensor to a vanilla torch.Tensor before torch.save.

    The GGUF weight path wraps every tensor in gguf_vendor.ops.GGMLTensor, and that
    subclass type rides __torch_function__ all the way to the decoded pixel output —
    so a naive torch.save pickles a `gguf_vendor.ops.GGMLTensor` class reference into
    the .pt. Any consumer without tools/ on sys.path (e.g. the app.py gradio process)
    then dies on torch.load with "No module named 'gguf_vendor'". as_subclass drops the
    type sharing storage; .clone() gives the artifact its own clean storage so it is a
    fully portable, dependency-free tensor. Costs one CPU copy of an already-CPU tensor.
    """
    try:
        if type(t) is not torch.Tensor and isinstance(t, torch.Tensor):
            return t.as_subclass(torch.Tensor).clone()
    except Exception:
        pass
    return t


def _do_sliding_generation(pipe, size_configs, *, prompt, width, height,
                           window, steps, guide, out_pt, out_meta, video,
                           denoise_strength, overlap=9, discard=4,
                           color_match=1.0):
    """Any-length restyle via overlapping windows (Wan2GP sequential).
    Loads the FULL source, tiles into <=`window`-frame windows sharing an
    `overlap` seam, SDEdit-restyles each with the previous window's styled tail
    injected into the denoise loop (continuity), LAB color-matches the seam, and
    stitches. The window arithmetic + stitch are basi.sliding (CPU-unit-tested);
    here we only supply the per-window pipe.generate. Returns (wall, shape) and
    writes meta, exactly like _do_one_generation (serve loop emits the result)."""
    import torchvision
    from basi import sliding as _sl
    t0 = time.time()
    _vp = Path(video)
    if _vp.suffix == ".pt":
        src = torch.load(_vp, map_location="cpu").float()
    else:
        _fr, _, _ = torchvision.io.read_video(str(_vp), pts_unit="sec")
        src = _fr.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
    if src.shape[-2:] != (height, width):
        src = torch.nn.functional.interpolate(
            src.transpose(0, 1), size=(height, width),
            mode="bicubic", align_corners=False).transpose(0, 1)
    src = src.clamp_(-1, 1)
    total = src.shape[1]
    plan = _sl.plan_windows(total, window=window, overlap=overlap, discard=discard)
    _emit_runner_event("phase", name="sliding_plan", total=total, windows=len(plan))
    _shift = float(os.environ.get("BASIWAN_SHIFT", "5.0"))

    def gen_window(src_slice, overlap_pixels, w):
        # Wan VAE round-trips only 4k+1 frame counts (decode of L latents = 4*(L-1)
        # +1 pixels). The final window of an arbitrary-length upload is usually NOT
        # 4k+1, so pad it (repeat last frame) up to the next valid count, generate,
        # and trim back — otherwise the decode length wouldn't match the plan slice.
        rl = int(src_slice.shape[1])
        rl_valid = ((rl - 1 + 3) // 4) * 4 + 1
        if rl_valid != rl:
            src_in = torch.cat(
                [src_slice, src_slice[:, -1:].repeat(1, rl_valid - rl, 1, 1)], dim=1)
        else:
            src_in = src_slice
        _emit_runner_event("phase", name="sliding_window",
                           index=w.index, n=len(plan), frames=rl)
        vid = pipe.generate(
            input_prompt=prompt, size=(width, height),
            frame_num=rl_valid, sampling_steps=steps, seed=0,
            shift=_shift, guide_scale=(guide, guide), offload_model=True,
            video=src_in, denoise_strength=float(denoise_strength),
            overlap_pixels=overlap_pixels)
        return vid[:, :rl].clamp(-1, 1).float().cpu()   # trim pad back to plan length

    _vp = _VramPeak().start()
    out = _sl.orchestrate(src, plan, gen_window, overlap=overlap,
                          color_match=float(color_match)).clamp(-1, 1).float().cpu()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.time() - t0
    _vram = _vp.stop()
    _out_path = Path(out_pt)
    _tmp = _out_path.with_suffix(_out_path.suffix + ".tmp")
    torch.save(_plain_tensor(out), str(_tmp))
    try:
        with _tmp.open("rb") as _f:
            os.fsync(_f.fileno())
    except OSError:
        pass
    os.replace(str(_tmp), str(_out_path))
    meta = {"video_path": str(out_pt), "wall_s": wall, "vram": _vram,
            "shape": [width, height, total], "steps": steps, "guide": guide,
            "scheme": "sliding", "windows": len(plan), "overlap": overlap,
            "discard": discard}
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    print(f"[runner-gguf] sliding {len(plan)}w generated in {wall:.1f}s "
          f"shape={tuple(out.shape)}", flush=True)
    return wall, tuple(out.shape)


# First-class peak-VRAM telemetry. ONE source for the whole stack: every generation
# path (_do_one_generation / _s2v / _sliding) brackets its generate with this, writes
# the numbers into its meta.json, and stores them in _LAST_GEN_VRAM so the worker's
# `result` event reports the SAME figures the meta does. Records the torch allocator
# peak (deterministic + reproducible) AND polls device-level used memory in a daemon
# thread, so `device_peak_gb` matches what nvidia-smi shows (allocator-reserved + CUDA
# context + any non-torch device memory). All GB. No-op without CUDA.
_LAST_GEN_VRAM: dict = {}


class _VramPeak:
    __slots__ = ("_poll", "_stop", "_th", "_dev_peak")

    def __init__(self, poll_s: float = 0.05):
        self._poll = poll_s
        self._stop = None
        self._th = None
        self._dev_peak = 0

    def start(self):
        if not torch.cuda.is_available():
            return self
        torch.cuda.reset_peak_memory_stats()
        import threading
        _free0, _total = torch.cuda.mem_get_info()
        self._dev_peak = _total - _free0          # baseline (resident weights + others)
        self._stop = threading.Event()

        def _loop():
            while not self._stop.is_set():
                try:
                    _free, _tot = torch.cuda.mem_get_info()
                    used = _tot - _free
                    if used > self._dev_peak:
                        self._dev_peak = used
                except Exception:
                    pass
                self._stop.wait(self._poll)
        self._th = threading.Thread(target=_loop, daemon=True)
        self._th.start()
        return self

    def stop(self) -> dict:
        global _LAST_GEN_VRAM
        if not torch.cuda.is_available():
            _LAST_GEN_VRAM = {}
            return {}
        if self._stop is not None:
            self._stop.set()
        if self._th is not None:
            self._th.join(timeout=1.0)
        _LAST_GEN_VRAM = {
            "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 2),
            "device_peak_gb": round(self._dev_peak / 1024**3, 2),
        }
        return _LAST_GEN_VRAM


def _do_one_generation_s2v(pipe, *, prompt, width, height, frames, out_pt, out_meta,
                           image, audio):
    """One S2V (audio-driven talking character) generation via WanS2V.generate.
    Enforces the validated S2V_RECIPE (steps 40 / guide 4.5 / shift 3.0 — not the
    t2v Lightning defaults). width*height is the max pixel AREA (get_gen_size honors
    the reference aspect). image = reference character, audio = driving wav.
    num_repeat=None lets encode_audio set the clip length from the audio."""
    from basi import s2v as _S
    if image is None or audio is None:
        raise RuntimeError("s2v requires both 'image' (reference) and 'audio' (wav)")
    _rec = _S.S2V_RECIPE
    _infer = frames if frames and frames > 0 else _rec["infer_frames"]
    # RESIDENCY-AWARE per-chunk window cap (the rule-#9-correct design). The
    # DiT-step activation scales ~linearly with tokens = infer_frames * megapixels
    # (K_ACT ~= 0.169 GB/(frame*Mpix), sweep tools/_s2v_720p_window_sweep.py) and must
    # fit in the FREE VRAM left after this process's resident weights AND anything else
    # on the card.
    #
    # Measured on a CLEAN 24GiB 4090 (12-step sweep): 720p 56f->17.5GB/54s-it,
    # 72f->20.0GB/60s-it (both healthy), 80f->21.7GB/158s-it (THRASH -- crosses
    # garbage_collection_threshold:0.8 ~= 20.6GB and the allocator GCs every step). So
    # the healthy 720p ceiling is ~72f on a clean card. The catastrophic thrash seen
    # earlier at EVERY window was CONTAMINATION: a concurrent ~10GB worker left only
    # ~14GB free, so even tiny windows hit the wall (rule #9 -- diagnose residency
    # before transients; verified by re-running clean). Basing the cap on FREE (not
    # total) makes it self-correct: ~full window on a clean card, automatically
    # conservative if memory is occupied -- it cannot thrash. GRID_OVERSHOOT 1.08
    # (ref-upscale + /64 rounding); FRAG_MARGIN 4.0GB keeps the peak below the GC
    # trigger. Gate: only >0.6MP grids risk it; 480p-class is PROVEN at the full window
    # (S9) and is never capped. Length unaffected: chunks chain (motion carry + S8).
    # FRAG_MARGIN 5.0: keeps the per-chunk window in the MEASURED fast zone (clean
    # sweep: <=72f runs 54-60 s/it; 80f GC-thrashes to 158 s/it). With the per-chunk
    # empty_cache reclaim in speech2video.generate preventing OOM, this margin targets
    # ~52f on a clean 24GB card -- comfortably fast, chains for length.
    _K_ACT, _GRID_OVERSHOOT, _FRAG_MARGIN, _CAP_MPIX = 0.169, 1.08, 5.0, 0.60
    if torch.cuda.is_available() and (int(width) * int(height)) / 1e6 > _CAP_MPIX:
        torch.cuda.synchronize()
        _free_gb = torch.cuda.mem_get_info()[0] / 1e9   # free AFTER resident weights
        _mpix = (int(width) * int(height)) / 1e6 * _GRID_OVERSHOOT
        _act_budget = _free_gb - _FRAG_MARGIN           # room for step activation + frag
        _safe = max(16, int(_act_budget / (_K_ACT * max(_mpix, 1e-3))))
        if _infer > _safe:
            print(f"[runner-gguf] S2V window capped {_infer}->{_safe} for {_mpix:.2f}MP "
                  f"(residency-aware: {_free_gb:.1f}GB free, {_act_budget:.1f}GB for "
                  f"activation; chunks chain for length)", flush=True)
            _infer = _safe
    # S2V output resolution is gated by the REFERENCE image:
    # get_size_less_than_area keeps a ref SMALLER than max_area at its own (small)
    # size and only pads -- so a small reference yields a small video no matter the
    # selected resolution. To honor the user's resolution choice, upscale a too-small
    # ref (Lanczos, aspect-preserved) just past max_area so the engine's size math
    # scales it DOWN to the full target grid; the DiT then generates at that grid
    # (synthesizing detail a plain upscale can't). No-op when the ref already meets
    # the budget. (Apache-2.0 change note.)
    _ref_path = str(image)
    try:
        from PIL import Image as _PILImage
        _ri = _PILImage.open(_ref_path).convert("RGB")
        _tgt = int(width) * int(height)
        if _ri.width * _ri.height < _tgt:
            _sc = (1.05 * _tgt / (_ri.width * _ri.height)) ** 0.5
            _nw, _nh = max(64, round(_ri.width * _sc)), max(64, round(_ri.height * _sc))
            _ref_up = Path(out_pt).with_suffix(".ref_up.png")
            _ri.resize((_nw, _nh), _PILImage.LANCZOS).save(_ref_up)
            _ref_path = str(_ref_up)
            print(f"[runner-gguf] S2V ref upscaled {_ri.width}x{_ri.height}->{_nw}x{_nh} "
                  f"to unlock the {int(width)}x{int(height)} grid", flush=True)
    except Exception as _e:  # noqa: BLE001
        print(f"[runner-gguf] S2V ref upscale skipped: {_e}", flush=True)
    _vp = _VramPeak().start()
    t0 = time.time()
    video = pipe.generate(
        input_prompt=prompt, ref_image_path=_ref_path, audio_path=str(audio),
        enable_tts=False, tts_prompt_audio=None, tts_prompt_text=None, tts_text=None,
        num_repeat=None, pose_video=None, max_area=int(width) * int(height),
        infer_frames=_infer, shift=_rec["shift"], sampling_steps=_rec["sampling_steps"],
        guide_scale=_rec["guide_scale"], seed=0, offload_model=False,
        color_anchor_strength=_rec.get("color_anchor_strength", 0.0),
        color_anchor_min_chunks=_rec.get("color_anchor_min_chunks", 4))
    torch.cuda.synchronize()
    wall = time.time() - t0
    _vram = _vp.stop()
    video = video.clamp(-1, 1).float().cpu()
    _out = Path(out_pt); _tmp = _out.with_suffix(_out.suffix + ".tmp")
    torch.save(_plain_tensor(video), str(_tmp))
    try:
        with _tmp.open("rb") as _f:
            os.fsync(_f.fileno())
    except OSError:
        pass
    os.replace(str(_tmp), str(_out))
    meta = {"video_path": str(out_pt), "wall_s": wall, "vram": _vram,
            "shape": list(tuple(video.shape)), "model_type": "s2v",
            "recipe": _rec, "infer_frames": _infer, "scheme": "gguf-s2v"}
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    print(f"[runner-gguf] S2V generated in {wall:.1f}s shape={tuple(video.shape)}",
          flush=True)
    return wall, tuple(video.shape)


def _do_one_generation(pipe, size_configs, *, prompt, width, height, frames,
                       steps, guide, out_pt, out_meta, clip_t=False, seed=0,
                       lora_strength=None, image=None,
                       video=None, denoise_strength=1.0,
                       vace_depth=False, vace_context_scale=1.0,
                       vace_end_percent=1.0,
                       vace_edit=False, anchor_images=None, anchor_positions=None,
                       audio=None):
    """Run one generation. Auto-gates VAE per shape, calls pipe.generate, writes
    atomic .pt + meta.json. Returns (wall_s, video_shape_tuple).

    Used by both legacy CLI path and --serve worker loop.
    """
    # S2V is a separate engine with a different generate signature -> dedicated
    # path. image = reference character, audio = driving track.
    if args.model_type == "s2v":
        return _do_one_generation_s2v(
            pipe, prompt=prompt, width=width, height=height, frames=frames,
            out_pt=out_pt, out_meta=out_meta, image=image, audio=audio)
    # Reset VAE env vars to their worker-start values BEFORE the
    # auto-gate re-evaluates. Without this, a prior p720 request leaves
    # BASIWAN_TAEHV_VAE="0" and BASIWAN_VAE_TILING="1" set, which then blocks
    # the auto-gate guard on a subsequent p480 request and silently uses the
    # wrong VAE path. We preserve user explicit overrides by snapshotting the
    # ORIGINAL values once (at module load below) and restoring them here.
    def _restore(name, val):
        if val is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = val
    _restore("BASIWAN_TAEHV_VAE", _INITIAL_BASIWAN_TAEHV_VAE)
    _restore("BASIWAN_VAE_TILING", _INITIAL_BASIWAN_VAE_TILING)

    # Per-request LoRA strength. None ⇒ leave whatever was
    # baked at pipe-build time (legacy CLI path uses args.lora_strength as
    # the baked value). When called via the worker --serve path with a
    # per-request lora_strength, override here so the UI slider is honored.
    if lora_strength is not None and args.lora_mode == "forward":
        _n = _apply_lora_strength_to_pipe(pipe, lora_strength)
        if _n:
            print(f"[runner-gguf] LoRA strength set to {lora_strength} on {_n} modules", flush=True)

    # Auto-gate at request time (per-shape). The same logic that lived inline
    # in main() prior to; moving it here means a worker serving
    # multiple shapes will re-evaluate on each request.
    if (torch.cuda.is_available()
            and not os.environ.get("BASIWAN_TAEHV_VAE")  # respect explicit override
            and width == 1280 and height == 720
            and frames <= 33):
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gb >= 22.0:
            os.environ["BASIWAN_VAE_TILING"] = "1"
            os.environ["BASIWAN_TAEHV_VAE"] = "0"
            print(f"[runner-gguf] auto-gate: tiled-full-VAE ON for "
                  f"{width}x{height} {frames}f on "
                  f"{total_vram_gb:.1f} GB card", flush=True)
        else:
            print(f"[runner-gguf] auto-gate: {total_vram_gb:.1f} GB < 22 GB; "
                  f"keeping TAEHV at p720", flush=True)

    size = size_configs.get(f"{width}*{height}", (width, height))
    # I2V conditioning image (continuation). Only valid
    # when the worker was started with --model-type i2v (in_dim=36 experts);
    # on a T2V worker the patch_embedding concat would shape-error, so fail
    # loud and early with an actionable message instead.
    _img = None
    if image is not None:
        if args.model_type != "i2v":
            raise RuntimeError(
                "request has 'image' but worker is --model-type t2v; "
                "restart the worker with the I2V GGUF pair")
        from PIL import Image as _PILImage
        _img = _PILImage.open(image).convert("RGB")
    # SDEdit V2V restyle source. Decode the upload to a
    # (3, frames, H, W) pixel tensor in [-1,1], resized to the request shape
    # and fit to `frames` (fail loud if shorter — caller should pre-trim).
    # T2V worker only; restyle rides content in the latent, needs no I2V
    # experts. _vid_src=None / denoise_strength>=1.0 = plain T2V.
    _vid_src = None
    if video is not None and float(denoise_strength) < 1.0:
        _vp = Path(video)
        if _vp.suffix == ".pt":
            _vid_src = torch.load(_vp, map_location="cpu").float()
        else:
            import torchvision
            _fr, _, _ = torchvision.io.read_video(str(_vp), pts_unit="sec")
            _vid_src = _fr.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
        if _vid_src.shape[1] < frames:
            raise RuntimeError(
                f"source video has {_vid_src.shape[1]} frames < requested "
                f"{frames}; trim the request or supply a longer clip")
        _vid_src = _vid_src[:, :frames]
        if _vid_src.shape[-2:] != (height, width):
            _vid_src = torch.nn.functional.interpolate(
                _vid_src.transpose(0, 1), size=(height, width),
                mode="bicubic", align_corners=False).transpose(0, 1)
        _vid_src = _vid_src.clamp_(-1, 1)

    # VACE depth-control restyle: load the source, extract its depth as
    # the control video, and ENFORCE the validated quality recipe (steps/guide/
    # shift pulled from basi.vace.VACE_DEPTH_RECIPE — the single source of truth
    # so the 0.973-depth-lock settings can't silently drift to the broken fast
    # regime). Full denoise (no SDEdit init); structure comes from the depth.
    _vace_video = None
    _vace_shift = None
    _vace_mask = None
    if vace_edit:
        # Keyframe-anchored editing: build the VACE (guide, mask) from the
        # edited anchor frames at their positions; generate() hard-locks those
        # anchor latents each step so the edits are enforced + propagate. Enforce
        # the same quality recipe as depth (50/guide5/shift3). anchors-only (no
        # depth) per the v1 scope (anchors+dense control is a Wan2GP bug).
        from basi import vace as _vmod
        import imageio.v3 as _iio
        _anchors = []
        for _ap in (anchor_images or []):
            _im = _iio.imread(str(_ap)).astype("float32")
            _anchors.append(torch.from_numpy(_im[..., :3]).permute(2, 0, 1) / 127.5 - 1.0)
        _positions = [int(p) for p in (anchor_positions or [])]
        _vace_video, _vace_mask = _vmod.build_keyframe_guide_and_mask(
            _anchors, _positions, frames, height, width)
        steps = _vmod.VACE_DEPTH_RECIPE["sampling_steps"]
        guide = _vmod.VACE_DEPTH_RECIPE["guide_scale"]
        _vace_shift = _vmod.VACE_DEPTH_RECIPE["shift"]
        _emit_runner_event("phase", name="vace_edit",
                           anchors=len(_anchors), positions=_positions)
    if vace_depth:
        from basi import vace as _vmod, depth as _dep
        _vp = Path(video)
        if _vp.suffix == ".pt":
            _vsrc = torch.load(_vp, map_location="cpu").float()
        else:
            import torchvision
            _fr, _, _ = torchvision.io.read_video(str(_vp), pts_unit="sec")
            _vsrc = _fr.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0)
        if _vsrc.shape[1] < frames:
            raise RuntimeError(
                f"vace source has {_vsrc.shape[1]} frames < requested {frames}")
        _vsrc = _vsrc[:, :frames]
        if _vsrc.shape[-2:] != (height, width):
            _vsrc = torch.nn.functional.interpolate(
                _vsrc.transpose(0, 1), size=(height, width),
                mode="bicubic", align_corners=False).transpose(0, 1)
        _vsrc = _vsrc.clamp_(-1, 1)
        _emit_runner_event("phase", name="vace_depth_extract")
        # Extract depth on CPU. The Depth-Anything Dinov2 backbone uses raw
        # nn.Linear (F.linear); once the BASIWAN GGUF pipe is built it has set
        # deterministic/cuBLAS state that makes those F.linear calls trip the
        # cuBLASLt workspace-heuristic bug (bogus 256 GiB alloc) on this Windows
        # torch build — bisected to depth.py:extract_depth, NOT the VACE forward.
        # The depth model is ~100MB and 17 frames on CPU costs ~tens of seconds,
        # negligible vs the ~7-min 50-step generate, and it sidesteps the bug
        # robustly regardless of the resident pipe's backend state.
        _dm, _dp = _get_depth_model()   # kept on CPU; extract runs on CPU
        _vace_video = _dep.extract_depth(_vsrc, _dm, _dp).cpu()
        steps = _vmod.VACE_DEPTH_RECIPE["sampling_steps"]
        guide = _vmod.VACE_DEPTH_RECIPE["guide_scale"]
        _vace_shift = _vmod.VACE_DEPTH_RECIPE["shift"]

    # Seed: a >=0 value is reproducible; <0 (the UI's "-1 = random") draws a fresh seed per
    # call so repeated clicks vary (a fixed seed would make every generation identical).
    _seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**31 - 1)
    torch.manual_seed(_seed)
    _vp = _VramPeak().start()
    t0 = time.time()
    _shift = (_vace_shift if _vace_shift is not None
              else float(os.environ.get("BASIWAN_SHIFT", "5.0")))
    # VACE runs offload_model=False (everything resident). The validated
    # path used False and fit on 24GB at p480_17f; offload_model=True swaps
    # experts CPU<->GPU mid-generate, fragmenting the allocator enough to trip
    # the cuBLASLt F.linear workspace-heuristic bug (the bogus 256 GiB alloc).
    # Non-VACE paths keep the proven offload=True memory behaviour.
    _offload = False if (vace_depth or vace_edit) else True
    video = pipe.generate(
        input_prompt=prompt, size=size,
        frame_num=frames, sampling_steps=steps, seed=_seed,
        shift=_shift,
        guide_scale=(guide, guide), offload_model=_offload,
        img=_img, video=_vid_src, denoise_strength=float(denoise_strength),
        vace_video=_vace_video, vace_context_scale=float(vace_context_scale),
        vace_end_percent=float(vace_end_percent), vace_mask=_vace_mask)
    torch.cuda.synchronize()
    wall = time.time() - t0
    _vram = _vp.stop()
    video = video.clamp(-1, 1).float().cpu()

    # Atomic write: torch.save → fsync → rename.
    _out_path = Path(out_pt)
    _tmp_path = _out_path.with_suffix(_out_path.suffix + ".tmp")
    torch.save(_plain_tensor(video), str(_tmp_path))
    try:
        with _tmp_path.open("rb") as _f:
            os.fsync(_f.fileno())
    except OSError:
        pass
    os.replace(str(_tmp_path), str(_out_path))
    meta = {
        "video_path": str(out_pt),
        "wall_s": wall,
        "vram": _vram,
        "shape": [width, height, frames],
        "steps": steps,
        "guide": guide,
        "scheme": "gguf",
        "gguf_high": str(args.gguf_high),
        "gguf_low": str(args.gguf_low),
    }
    if clip_t:
        # --clip-t is a dev-only quality metric; prod_shape_bench is a dev bench (not shipped).
        # Degrade gracefully so the flag never crashes a generation if the bench is absent.
        try:
            from prod_shape_bench import _clip_t
            meta["clip_t"] = _clip_t(video, prompt)
        except Exception as _e:  # noqa: BLE001
            print(f"[runner-gguf] --clip-t skipped (metric unavailable: {_e})", flush=True)
    Path(out_meta).write_text(json.dumps(meta, indent=2))
    print(f"[runner-gguf] generated in {wall:.1f}s shape={tuple(video.shape)}", flush=True)
    return wall, tuple(video.shape)


def _emit_runner_event(event, **kw):
    """JSON event emission (paired with human-readable lines in non-JSON mode)."""
    rec = {"_basiwan_event": True, "event": event, **kw}
    line = json.dumps(rec, separators=(",", ":"))
    if os.environ.get("BASIWAN_RUNNER_JSON_ONLY") == "1":
        print(line, flush=True)
    else:
        print(f"[BASIWAN-EVENT] {line}", flush=True)


def _serve_loop(pipe, size_configs):
    """Persistent-worker loop. Reads JSON request lines from stdin, runs
    pipe.generate() per request, emits JSON events to stdout. Catches OOM as
    non-fatal (worker continues); catches other CUDA errors as fatal (worker
    emits then exits, so the parent supervisor respawns).
    """
    _resident = (round(torch.cuda.memory_reserved() / 1024**3, 2)
                 if torch.cuda.is_available() else None)
    _emit_runner_event("ready", resident_gb=_resident)
    _prewarm_joined = False  # block the first gen on the mmap pre-warm
    _combo_active = [False]   # mutable flag: a user-LoRA combo is loaded
    # runtime-scalable flag: the active LoRA is USER-ONLY (no Lightning
    # entangled in the rank-concat), so request-time lora_strength can scale the
    # single LoRA exactly. For a Lightning+user combo this must stay False —
    # one lora_scale can't separate the two, so strength is baked at build and
    # pinned. Measured (#388 strength sweep, depth-control): style peaks at 1.0,
    # degrades by 1.5, depth-lock stays 0.969-0.980 across all -> adjustment is
    # meaningful AND safe, so user-only LoRAs honor the request slider.
    _combo_runtime_scalable = [False]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _emit_runner_event("error", id=None, kind="BAD_JSON",
                               msg=str(e), fatal=False)
            continue
        cmd = req.get("cmd")
        req_id = req.get("id")
        if cmd == "shutdown":
            _emit_runner_event("shutdown", id=req_id, ok=True)
            return 0
        if cmd == "ping":
            _r = (round(torch.cuda.memory_reserved() / 1024**3, 2)
                  if torch.cuda.is_available() else None)
            _emit_runner_event("pong", id=req_id, resident_gb=_r)
            continue
        if cmd == "set_lora":
            # Hot-swap the user LoRA combo on the live worker. No
            # restart; ~0.3 s (load_file + attach + device fixup). When a
            # combo is active, request-time lora_strength is pinned to 1.0
            # (user strength is baked into the combo) — _combo_active tells
            # _do_one_generation to ignore stray non-1.0 strengths.
            _a = req.get("args", {})
            _dir = _a.get("dir")
            try:
                import time as _t
                _t0 = _t.time()
                _res = _set_lora_on_pipe(pipe, _dir,
                                         float(_a.get("strength", 1.0)))
                _combo_active[0] = _dir is not None
                # user-only LoRA -> request-time strength is honored, not
                # pinned (no Lightning to entangle). Default False = the safe
                # baked/pinned behaviour for Lightning+user combos.
                _combo_runtime_scalable[0] = bool(
                    _dir is not None and _a.get("runtime_scalable", False))
                _res["wall_s"] = round(_t.time() - _t0, 3)
                _emit_runner_event("lora_set", id=req_id, **_res)
            except Exception as _e:
                _emit_runner_event("error", id=req_id, kind="SET_LORA_FAILED",
                                   msg=f"{type(_e).__name__}: {_e}", fatal=False)
            continue
        if cmd != "generate":
            _emit_runner_event("error", id=req_id, kind="BAD_CMD",
                               msg=f"unknown cmd: {cmd!r}", fatal=False)
            continue
        a = req.get("args", {})
        try:
            # The mmap pre-warm scan (started at pack-load, fire-and-
            # forget) is still streaming the ~19 GB pack off disk when `ready`
            # fires. Joining it BEFORE the first forward pass converts a
            # hidden random-page-fault storm (observed 1000 s+ when a duplicate
            # worker — now blocked by #378 — starved the page cache) into one
            # visible sequential wait. No-op when already complete.
            if not _prewarm_joined:
                _prewarm_joined = True
                try:
                    from gguf_vendor.basiwan_pack_cache import join_prewarm
                    _pw_wall = join_prewarm(
                        progress_cb=lambda el: _emit_runner_event(
                            "phase", id=req_id, name="weights_page_in",
                            elapsed_s=round(el, 1)))
                    if _pw_wall > 1.0:
                        _emit_runner_event("phase", id=req_id,
                                           name="weights_page_in_done",
                                           wall_s=round(_pw_wall, 1))
                except Exception as _pe:
                    _emit_runner_event("phase", id=req_id,
                                       name="weights_page_in_skipped",
                                       msg=f"{type(_pe).__name__}: {_pe}")
            _emit_runner_event("started", id=req_id,
                               prompt_first40=(a.get("prompt", "") or "")[:40])
            if a.get("sliding"):
                # Any-length restyle: tile into overlapping windows.
                # SDEdit defaults (8-step mid-entry, denoise 0.6) match the #385
                # single-window restyle recipe.
                wall, shape = _do_sliding_generation(
                    pipe, size_configs,
                    prompt=a["prompt"], width=a["width"], height=a["height"],
                    window=a.get("frames", 81), steps=a.get("steps", 8),
                    guide=a.get("guide", 1.0),
                    out_pt=a["out"], out_meta=a["meta"],
                    video=a.get("video"),
                    denoise_strength=a.get("denoise_strength", 0.6),
                    overlap=a.get("sliding_overlap", 9),
                    discard=a.get("sliding_discard", 4),
                    color_match=a.get("sliding_color_match", 1.0))
                _emit_runner_event("result", id=req_id, wall_s=round(wall, 3),
                                   shape=list(shape), out_pt=a["out"],
                                   out_meta=a["meta"], vram=_LAST_GEN_VRAM, ok=True)
                continue
            wall, shape = _do_one_generation(
                pipe, size_configs,
                prompt=a["prompt"], width=a["width"], height=a["height"],
                frames=a["frames"], steps=a.get("steps", 4),
                guide=a.get("guide", 1.0), seed=a.get("seed", 0),
                out_pt=a["out"], out_meta=a["meta"],
                clip_t=a.get("clip_t", False),
                # A Lightning+user combo bakes user strength in
                # (alpha=rank → scale 1.0) and one lora_scale can't separate the
                # two, so pin to 1.0. A USER-ONLY LoRA (#388, runtime_scalable)
                # is a single LoRA — request strength scales it exactly, so honor
                # the slider. No active combo → honor the request value as before.
                lora_strength=(1.0 if (_combo_active[0]
                                       and not _combo_runtime_scalable[0])
                               else a.get("lora_strength")),
                image=a.get("image"),
                audio=a.get("audio"),
                video=a.get("video"),
                denoise_strength=a.get("denoise_strength", 1.0),
                # VACE depth-control restyle fields.
                vace_depth=a.get("vace_depth", False),
                vace_context_scale=a.get("vace_context_scale", 1.0),
                vace_end_percent=a.get("vace_end_percent", 1.0),
                # VACE keyframe-anchored editing fields.
                vace_edit=a.get("vace_edit", False),
                anchor_images=a.get("anchor_images"),
                anchor_positions=a.get("anchor_positions"))
            _emit_runner_event("result", id=req_id,
                               wall_s=round(wall, 3),
                               shape=list(shape),
                               out_pt=a["out"], out_meta=a["meta"],
                               vram=_LAST_GEN_VRAM, ok=True)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            import gc as _gc
            _gc.collect()
            _emit_runner_event("error", id=req_id, kind="CUDA_OOM",
                               msg=str(e), fatal=False)
        except KeyError as e:
            _emit_runner_event("error", id=req_id, kind="MISSING_FIELD",
                               msg=f"required arg missing: {e}", fatal=False)
        except RuntimeError as e:
            # CUDA runtime errors that aren't OOM — context state may be
            # corrupted, can't safely continue. Emit then exit(2) so the
            # supervisor restarts the worker.
            _msg = str(e)
            _fatal = "CUDA" in _msg or "cuda" in _msg
            _emit_runner_event("error", id=req_id, kind="RUNTIME_ERROR",
                               msg=_msg, fatal=_fatal)
            if _fatal:
                return 2
    return 0


def main():
    # Simulate a smaller card: cap the caching allocator to a fraction of
    # total VRAM so per-tier OOM walls can be probed on one GPU. FIT/NO-FIT gate
    # only — does NOT reproduce WDDM shared-RAM spill or real-card step times, so
    # treat a pass as "fits (soft)" and any speed as estimated. No-op when unset.
    _vram_frac = os.environ.get("BASIWAN_VRAM_FRACTION")
    if _vram_frac and torch.cuda.is_available():
        try:
            _f = float(_vram_frac)
            torch.cuda.set_per_process_memory_fraction(_f, 0)
            _tot = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"[runner-gguf] BASIWAN_VRAM_FRACTION={_f} -> simulating "
                  f"~{_f * _tot:.1f}GB of {_tot:.1f}GB (FIT gate; not perf-accurate)",
                  flush=True)
        except Exception as _e:
            print(f"[runner-gguf] VRAM-fraction cap failed (non-fatal): {_e}", flush=True)
    pipe, size_configs = _build_pipe_gguf(args.gguf_high, args.gguf_low,
                                          args.base_dir, args.model_type)
    print(f"[runner-gguf] pipe ready in {time.time() - _t_start:.1f}s", flush=True)

    # VACE runs offload_model=False (everything resident, fits in ~23 GB
    # on 24 GB at p480_17f) so it never needs chunked-FFN's memory savings — and
    # chunking MEASURES catastrophically slow on the VACE model: 44.8 s/it with
    # chunk=4096 (80 blocks wrapped) vs 7.9 s/it unchunked at 832x480x17f, a 5.7x
    # regression (memory-bound small-GEMM, 164 W / 100% util). Skip it for VACE.
    if args.ffn_chunk_size > 0 and args.model_type not in ("vace", "s2v"):
        n_wrap = _install_chunked_ffn_gguf(pipe, args.ffn_chunk_size)
        label = getattr(pipe, "_gguf_ffn_wrapper_label", "chunked-FFN")
        print(f"[runner-gguf] {label} installed: {n_wrap} blocks "
              f"(threshold=8000 chunk={args.ffn_chunk_size})", flush=True)
    elif args.model_type == "s2v":
        print("[runner-gguf] S2V: ffn-chunk + block-swap installed inside "
              "build_s2v_pipe_gguf; skipping main()'s installs", flush=True)
    elif args.model_type == "vace":
        print("[runner-gguf] VACE: chunked-FFN DISABLED (offload=False resident; "
              "chunking measured 5.7x slower — 44.8 vs 7.9 s/it at 832x480)",
              flush=True)

    if args.block_swap_n >= 0 and args.model_type != "s2v":
        _install_block_swap_gguf(pipe, args.block_swap_n)
        # Async ring-buffer pin setup. Probes the platform's pinned-memory
        # budget at runtime, allocates a rolling pin window of K blocks,
        # and exposes it on pipe._swap_ring for use by the async forward
        # wrap. K is bounded so total pinned memory stays under the cap
        # (WSL2 dxgkrnl pinned-page route + tmpfs sizing).
        if (os.environ.get("BASIWAN_BLOCK_SWAP_ASYNC") == "1"
                and torch.cuda.is_available()):
            _setup_async_ring_buffer(pipe, args.block_swap_n)

    # Auto-on tiled-full-VAE for 24GB+ cards at
    # p720_17f / p720_33f. p720_17f measured 16.31 GB peak; p720_33f predicted
    # ~20.9 GB via linear-frame scaling. p720_49f predicted ~25 GB (OOM on 24GB
    # leaving headroom); p720_81f FALSIFIED. Sub-720p shapes keep
    # TAEHV default — preview shapes don't need full-VAE imaging quality.
    if (torch.cuda.is_available()
            and not os.environ.get("BASIWAN_TAEHV_VAE")  # respect explicit override
            and args.width == 1280 and args.height == 720
            and args.frames <= 33):
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if total_vram_gb >= 22.0:
            os.environ["BASIWAN_VAE_TILING"] = "1"
            os.environ["BASIWAN_TAEHV_VAE"] = "0"
            print(f"[runner-gguf] auto-gate: tiled-full-VAE ON for "
                  f"{args.width}x{args.height} {args.frames}f on "
                  f"{total_vram_gb:.1f} GB card", flush=True)
        else:
            print(f"[runner-gguf] auto-gate: {total_vram_gb:.1f} GB < 22 GB; "
                  f"keeping TAEHV at p720", flush=True)

    # Inject TAEHV tiny VAE. The full Wan2.1 VAE's F.conv3d at 1280×720 OOMs
    # the leftover GPU after the experts return to CPU under block-swap;
    # TAEHV is ~50× smaller and decodes the same latents 4-6× faster at
    # slight quality cost. Same pattern as run_one_video.py:P33.
    if os.environ.get("BASIWAN_TAEHV_VAE", "1") != "0":
        # Portable default under BASIWAN_CKPT_DIR (else <repo>/checkpoints), where
        # smoke_test.py auto-downloads taew2_1.pth. Optional: if absent we fall
        # back to the full tiled VAE, so a missing file is harmless.
        _taehv_base = os.environ.get("BASIWAN_CKPT_DIR", str(REPO / "checkpoints"))
        ckpt = Path(os.environ.get("BASIWAN_TAEHV_CKPT",
                                    str(Path(_taehv_base) / "taehv" / "taew2_1.pth")))
        if ckpt.exists():
            try:
                # REPO is module-level (line ~137); do NOT reassign locally here —
                # a local REPO= shadows it across this whole function and makes the
                # earlier _taehv_base reference (which uses REPO) an UnboundLocalError
                # (caught by #394's 12GB-taehv cell — the TAEHV default path).
                sys.path.insert(0, str(REPO / "tools"))
                from taehv import TAEHV
                decoder = TAEHV(checkpoint_path=str(ckpt)).cpu().eval()
                decoder.requires_grad_(False)
                pipe._taehv_decoder = decoder
                print(f"[runner-gguf] TAEHV decoder injected (CPU until decode)",
                      flush=True)
            except Exception as e:
                print(f"[runner-gguf] TAEHV inject failed: {e}", flush=True)

    if args.serve:
        # Persistent-worker loop. Reads JSON request lines from stdin.
        return _serve_loop(pipe, size_configs)

    # Legacy one-shot CLI path. Run a single generation from argparse values.
    wall, shape = _do_one_generation(
        pipe, size_configs,
        prompt=args.prompt, width=args.width, height=args.height,
        frames=args.frames, steps=args.steps, guide=args.guide,
        out_pt=args.out, out_meta=args.meta, clip_t=args.clip_t,
        image=args.image, audio=args.audio)
    # Structured result event for legacy CLI consumers (matches the
    # one-shot subprocess path used by app.py prior to the worker refactor).
    _emit_runner_event("result", wall_s=round(wall, 3), shape=list(shape),
                       out_pt=str(args.out), out_meta=str(args.meta),
                       vram=_LAST_GEN_VRAM, ok=True)


_t_start = time.time()
if __name__ == "__main__":
    main()
