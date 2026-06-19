"""Bare-PyTorch wrapper around city96's GGUF loader.

ComfyUI-GGUF's `ops.py` hard-imports `comfy.ops`, `comfy.lora`, and
`comfy.model_management`, none of which are available in a bare PyTorch
context. This wrapper:

  1. Stubs the three `comfy.*` modules in `sys.modules` BEFORE importing
     the vendored `ops.py` (which is transitively pulled by `loader.py`).
     The stubs only need to satisfy attribute lookup at import time and
     a couple of API surfaces our forward path actually exercises.

  2. Re-exports the bits of the loader we need (`gguf_sd_loader`,
     `GGMLTensor`, `dequantize_tensor`).

  3. Provides `BareGGMLLinear`, a drop-in replacement for `nn.Linear`
     that holds a quantized `GGMLTensor` weight and dequantizes it on
     each forward call. Avoids the full `GGMLLayer` / `GGMLOps.Linear`
     class hierarchy that depends on `comfy.ops.manual_cast.Linear`.

Usage:
    from gguf_vendor.bare_gguf import (
        gguf_sd_loader, GGMLTensor, BareGGMLLinear, swap_linear_with_ggml,
    )
    sd, extra = gguf_sd_loader("/path/to/wan-q4_k_m.gguf", handle_prefix="model.")
    model.load_state_dict(sd, strict=False)
    swap_linear_with_ggml(model)
    model.eval()
"""
from __future__ import annotations

import os
import sys
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 2026-06-08 ROOT-CAUSE: F.linear on Windows torch 2.7.0+cu128 mis-sizes its
# cuBLASLt workspace and requests 256 GiB CUDA allocations for sizes used by
# Wan2.2 (time_embedding 256→5120 at seq_len=7800 under autocast(fp32); QKV
# 5120×5120 bf16). Both fail before the kernel starts. Explicit matmul+bias
# uses a different dispatch (regular cuBLAS gemm) that works correctly.
# Repro: F.linear((1,7800,256) fp32, (5120,256), (5120,)) → OOM 256 GiB.
# `torch.matmul(x, w.T) + b` → succeeds in milliseconds.
def _safe_linear(x: torch.Tensor, w: torch.Tensor, b):
    """F.linear replacement that bypasses cuBLASLt workspace bug."""
    out = torch.matmul(x.contiguous(), w.contiguous().transpose(-2, -1))
    if b is not None:
        out = out + b
    return out


# ---------------------------------------------------------------------------
# Step 1: stub comfy.* before importing the vendored loader.
# These stubs satisfy ops.py:6-8 hard imports and the few attribute uses we
# can encounter via cast_bias_weight / get_weight. The bare path replaces
# all nn.Linear modules with BareGGMLLinear (below), so most of ops.py is
# never actually called — but the module-level inheritance from
# `comfy.ops.manual_cast.Linear` still requires the stubs to load.
# ---------------------------------------------------------------------------
def _install_comfy_stubs() -> None:
    if "comfy" in sys.modules:
        return

    comfy = types.ModuleType("comfy")

    # comfy.ops with .manual_cast having Linear/Conv2d/Embedding/LayerNorm/GroupNorm
    # base classes. GGMLOps inherits from manual_cast, but we never instantiate it
    # — we use BareGGMLLinear instead. So nn.* are sufficient placeholders.
    ops_mod = types.ModuleType("comfy.ops")
    class _ManualCast:
        Linear = nn.Linear
        Conv2d = nn.Conv2d
        Embedding = nn.Embedding
        LayerNorm = nn.LayerNorm
        GroupNorm = nn.GroupNorm
    ops_mod.manual_cast = _ManualCast
    def _cast_to(x, dtype, device, non_blocking=False, copy=False):
        return x.to(device=device, dtype=dtype, non_blocking=non_blocking, copy=copy)
    ops_mod.cast_to = _cast_to

    lora_mod = types.ModuleType("comfy.lora")
    def _calculate_weight(patches, weight, key, intermediate_dtype=None):
        # Bare context never has patches.
        return weight
    lora_mod.calculate_weight = _calculate_weight

    mm_mod = types.ModuleType("comfy.model_management")
    def _device_supports_non_blocking(device):
        return False  # safe default; bare loader stays on CPU until forward
    mm_mod.device_supports_non_blocking = _device_supports_non_blocking

    sys.modules["comfy"] = comfy
    sys.modules["comfy.ops"] = ops_mod
    sys.modules["comfy.lora"] = lora_mod
    sys.modules["comfy.model_management"] = mm_mod
    comfy.ops = ops_mod
    comfy.lora = lora_mod
    comfy.model_management = mm_mod


_install_comfy_stubs()


# Now safe to import the vendored loader chain.
from .loader import gguf_sd_loader  # noqa: E402
from .ops import GGMLTensor  # noqa: E402
from .dequant import dequantize_tensor, is_quantized  # noqa: E402


# [#400/W14] Pure-PyTorch dequant fallback gate. The Q4_K/Q6_K CUDA kernel is the
# fast path, but it needs a build toolchain + CUDA; AMD ROCm / Apple MPS / no-
# toolchain installs can't use it. When unavailable we fall back to the device-
# agnostic dequantize_tensor + _safe_linear path (PROVEN numerically == kernel:
# cos>0.9999, _smoke_dequant_fallback.py). Slower (dequant per call) but correct,
# and it lets those platforms run at all (vs the old hard raise).
_KERNEL_AVAIL = None


def basiwan_kernel_available() -> bool:
    """Lazy, cached: can the BASIWAN Q4_K/Q6_K CUDA kernel actually load? False on
    non-CUDA (incl. ROCm, whose .device is 'cuda' but the CUDA-C .pyd won't load),
    or when the build toolchain is absent. Drives both the load-time prepack gate
    and the forward dispatch."""
    global _KERNEL_AVAIL
    if _KERNEL_AVAIL is None:
        if not torch.cuda.is_available() or getattr(torch.version, "hip", None):
            _KERNEL_AVAIL = False
        else:
            try:
                from .basiwan_q4_kernel import _load_extension
                _load_extension()
                _KERNEL_AVAIL = True
            except Exception:
                _KERNEL_AVAIL = False
    return _KERNEL_AVAIL


def use_dequant_fallback(x) -> bool:
    """True when the forward should take the pure-torch dequant path: explicit
    BASIWAN_FORCE_DEQUANT, a non-CUDA device (mps/cpu), or no usable kernel."""
    if os.environ.get("BASIWAN_FORCE_DEQUANT") == "1":
        return True
    if getattr(x, "device", None) is not None and x.device.type != "cuda":
        return True
    return not basiwan_kernel_available()


class BareGGMLLinear(nn.Module):
    """nn.Linear-compatible wrapper that holds a quantized GGMLTensor
    weight and dequantizes on every forward. ~Q4 weight stays in VRAM at
    its compressed size; the fp16/bf16 copy is materialized in-place each
    call, used by F.linear, and freed.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Placeholders; real tensors get assigned post-load via __setattr__
        # bypass since we want to hold a GGMLTensor (not a Parameter).
        self.weight = None  # type: ignore[assignment]
        self.bias = None    # type: ignore[assignment]
        self._has_bias = bias
        # Forward-time LoRA delta (optional). When set, `forward` adds
        # `(x @ lora_down.T) @ lora_up.T * lora_scale` after the main
        # F.linear. Keeps the Q4 weight at its compressed footprint
        # (vs merge-at-load which dequant's to BF16 = ~4× the size).
        self.lora_down = None  # type: ignore[assignment]
        self.lora_up = None    # type: ignore[assignment]
        self.lora_scale: float = 0.0
        self._basiwan_packed = None
        self._basiwan_runtime = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [2026-06-02] One-shot residency probe. Print VRAM stats at the
        # very first BareGGMLLinear.forward call of a given process so any
        # subsequent OOM has a baseline to subtract from. Enabled via
        # BASIWAN_MEM_PROBE=1; off by default to keep the runner output
        # clean during measured benches.
        if (
            os.environ.get("BASIWAN_MEM_PROBE") == "1"
            and not getattr(BareGGMLLinear, "_mem_probe_fired", False)
            and torch.cuda.is_available()
        ):
            BareGGMLLinear._mem_probe_fired = True
            alloc = torch.cuda.memory_allocated() / 1e9
            res = torch.cuda.memory_reserved() / 1e9
            free, total = torch.cuda.mem_get_info()
            print(
                f"[mem-probe] first BareGGMLLinear.forward — "
                f"x.shape={tuple(x.shape)} dtype={x.dtype}: "
                f"allocated={alloc:.2f} GB  reserved={res:.2f} GB  "
                f"cuda_free={free/1e9:.2f} GB / total={total/1e9:.2f} GB",
                flush=True,
            )
        w = self.weight
        # When prepack_basiwan_weights ran during load, the original Q4/Q6
        # blob is freed and self.weight is None — _basiwan_packed owns
        # everything BASIWAN needs. Don't error on weight=None in that case.
        if w is None and self._basiwan_packed is None:
            raise RuntimeError("BareGGMLLinear.weight not set")
        b = self.bias
        if b is not None and is_quantized(b):
            b = dequantize_tensor(b, dtype=x.dtype)
        elif b is not None and b.dtype != x.dtype:
            b = b.to(x.dtype)
        if b is not None and b.device != x.device:
            b = b.to(x.device)

        out = None
        qtype_name = getattr(getattr(w, "tensor_type", None), "name",
                             None) if w is not None else (
            "Q4_K" if self._basiwan_packed is not None
                     and self._basiwan_packed.quant_type == 0 else
            "Q6_K" if self._basiwan_packed is not None else None
        )

        # [2026-06-04 drift-bisect] BASIWAN_BISECT_DEQUANT=1 forces the
        # dequant + F.linear path for Q4_K/Q6_K. This is EXPLICITLY ENABLED
        # forensic-only — diagnostic for the composite Q 0.7808 vs 0.842 drift.
        # Not a silent fallback. Will OOM at p720+ shapes; only use at p480/small.
        _bisect_dequant = os.environ.get("BASIWAN_BISECT_DEQUANT") == "1"
        if (w is not None and is_quantized(w)) or self._basiwan_packed is not None:
            # Production constraint (anti-laziness): NO silent fallback to the
            # baseline dequant+F.linear path for quantized weights. It allocates
            # a full BF16 weight per call which fragments the allocator and OOMs
            # at p720+ shapes. Every quantized path must succeed on its own kernel
            # OR raise loudly. See feedback_no_lazy_impl.md, feedback_supervise_codex.md.
            if _bisect_dequant and qtype_name in {"Q4_K", "Q6_K"}:
                # Diagnostic-only path. Dequant the Q4_K/Q6_K blob to BF16 and
                # F.linear. Verifies whether the BASIWAN kernel is contributing
                # to the today-vs-ship drift. NOT for production. (dequantize_tensor
                # is module-level imported at top — NO local import here: a local
                # `from .dequant import dequantize_tensor` would shadow it across the
                # whole forward and break the W14 fallback branch below — #400.)
                w_dq = dequantize_tensor(w, x.dtype).to(x.device)
                # 2026-06-08: F.linear has a cuBLASLt workspace bug on
                # Windows torch 2.7.0+cu128 that asks 256 GiB for QKV-sized
                # bf16 matmuls. Use explicit matmul+bias which works.
                out = _safe_linear(x, w_dq, b)
            elif qtype_name in {"Q4_K", "Q6_K"} and use_dequant_fallback(x):
                # [#400/W14] Pure-PyTorch dequant fallback for AMD ROCm / Apple MPS
                # / no-build-toolchain installs (and BASIWAN_FORCE_DEQUANT). PROVEN
                # numerically == the kernel: cos>0.9999, rel<0.007 vs the official
                # gguf dequant (tools/_smoke_dequant_fallback.py). Slower (dequant
                # per call) but correct + device-agnostic — replaces the old hard
                # raise so these platforms run AT ALL. Needs the raw Q4/Q6 weight,
                # so the loader skips prepack when basiwan_kernel_available()=False.
                if w is None:
                    raise RuntimeError(
                        f"GGUF {qtype_name} dequant fallback needs the raw weight but "
                        f"prepack freed it. The loader must skip prepack when "
                        f"basiwan_kernel_available() is False (W14 load-time gate).")
                out = _safe_linear(x, dequantize_tensor(w, x.dtype).to(x.device), b)
            elif qtype_name in {"Q4_K", "Q6_K"}:
                from .basiwan_q4_kernel import basiwan_q4k_linear, pack_basiwan_weight, BasiwanRuntimeBuffers

                if self._basiwan_packed is None:
                    # Last-resort lazy pack (prepack should have populated this).
                    self._basiwan_packed = pack_basiwan_weight(w)
                # BASIWAN kernel only accepts fp16 / bf16. If BASIWAN_NO_BF16_MOD=1
                # is active (used at p720+ to save the fp32→bf16 cast of e_block at
                # the modulation step), x can flow into here as fp32. Cast back to
                # bf16 for the BASIWAN call; output stays bf16.
                if x.dtype == torch.float32:
                    x = x.to(torch.bfloat16)
                # [2026-06-05 drift-bisect] BASIWAN_NO_POOL=1 forces a per-call
                # fresh BasiwanRuntimeBuffers() instead of the shared pool. Tests
                # whether pool buffer reuse is the source of the today-vs-ship
                # drift. Cost: extra (M, N) alloc per call.
                _basiwan_runtime = (
                    BasiwanRuntimeBuffers()
                    if os.environ.get("BASIWAN_NO_POOL") == "1"
                    else None
                )
                out = basiwan_q4k_linear(
                    x, self._basiwan_packed, bias=b, runtime=_basiwan_runtime
                )
                if out is None:
                    raise RuntimeError(
                        f"BASIWAN {qtype_name} path returned None; refusing silent fallback "
                        f"to dequant+F.linear (it OOMs at p720+ via fragmentation). "
                        f"Fix the BASIWAN kernel or pack adapter — do not paper over."
                    )
            else:
                # Non-Q4_K/Q6_K quantized (Q5_K, Q8_0, etc.). Our GGUF only
                # contains Q4_K + Q6_K (audited 2026-06-02: 280 Q4_K + 120 Q6_K
                # + 694 F16 + 1 F32). If this branch fires, the GGUF has an
                # unexpected quant — investigate, don't paper over.
                raise RuntimeError(
                    f"Unsupported quantized tensor type {qtype_name!r} in production path. "
                    f"Expected Q4_K or Q6_K. Refusing silent dequant fallback. "
                    f"If a new quant type was introduced, add an explicit kernel for it."
                )
        else:
            # Unquantized weight (F16 / F32 — norms, biases, embeddings). Safe to
            # go through F.linear: no allocator fragmentation, no dequant.
            # 2026-06-08 ROOT-CAUSE: GGMLTensor subclass has a
            # __torch_function__ override that mis-dispatches F.linear on
            # Windows torch 2.7.0+cu128 — it asks for 256 GiB output instead
            # of the correct 156 MB for time_embedding(256→5120) at
            # seq_len=7800. Strip the subclass via as_subclass(torch.Tensor)
            # so F.linear uses native dispatch.
            if type(w) is not torch.Tensor:
                w = w.as_subclass(torch.Tensor)
            if b is not None and type(b) is not torch.Tensor:
                b = b.as_subclass(torch.Tensor)
            if w.dtype != x.dtype:
                w = w.to(x.dtype)
            if w.device != x.device:
                w = w.to(x.device)
            if b is not None and b.dtype != x.dtype:
                b = b.to(x.dtype)
            if b is not None and b.device != x.device:
                b = b.to(x.device)
            out = _safe_linear(x, w, b)

        # Forward-time LoRA delta: (x @ down.T) @ up.T * scale.
        # Rank-128 LoRA adds ~0.1% of the main F.linear FLOPs.
        #
        # All-bf16 cuBLAS path (revised 2026-06-02 v2):
        #   Earlier we routed the second matmul through fp32 to preserve
        #   precision (−5.9% CLIP-T was the bf16-only cost measured
        #   2026-06-01). At p480_17f FFN[0] (M=7800, N=13568) the fp32
        #   output is 423 MB — alongside block-swap fragmentation that
        #   still OOMs on a 24 GB card.
        #
        #   cuBLAS bf16 GEMM uses fp32 internal accumulation, so the
        #   structural precision loss is the per-output bf16 ROUND, not
        #   accumulator truncation. Output stays bf16; peak transient
        #   drops from 423 MB → 211 MB at FFN[0]. The −5.9% CLIP-T is
        #   the price of fitting in 24 GB during GGUF+BASIWAN diffusion.
        if self.lora_down is not None and self.lora_up is not None:
            d = self.lora_down
            u = self.lora_up
            if d.device != x.device: d = d.to(x.device)
            if u.device != x.device: u = u.to(x.device)
            x2d = x.reshape(-1, x.shape[-1])
            inter = x2d @ d.T.to(x.dtype)            # (M, rank) bf16, small
            # [2026-06-02] In-place addmm_ instead of `out = out + delta`:
            #   out += alpha * inter @ u.T
            # Eliminates the (M, N) delta + scaled-delta + sum tensors
            # (each ~111 MB at FFN[0] p720_33f chunk = 333 MB transient).
            # `out` is a slice view of the BasiwanRuntimePool's pooled buffer,
            # so the mutation is on the pool slot — safe across calls because
            # the next forward overwrites the same slot before reading.
            uT = u.T.to(x.dtype)
            # Reshape out_2d to match (M_total, N) shape for the addmm.
            out_2d = out.reshape(-1, out.shape[-1])
            out_2d.addmm_(inter, uT, alpha=float(self.lora_scale))
        return out

    def _apply(self, fn, recurse=True):
        """Override nn.Module._apply so weight/bias/LoRA tensors follow
        device changes via .to(device), .cuda(), .cpu() etc.

        We hold these as plain attributes (not Parameter or Buffer) so the
        default _apply (which iterates _parameters and _buffers) misses
        them. Without this override, block-swap (which moves whole blocks
        between CPU and GPU) leaves the GGMLTensor weight on the original
        device — every forward then re-moves it implicitly, wasting wall
        time and breaking the swap-out semantics.
        """
        super()._apply(fn, recurse)
        if self.weight is not None:
            self.weight = fn(self.weight)
        # weight may have been freed after prepack_basiwan_weights — nothing to migrate.
        if self.bias is not None:
            self.bias = fn(self.bias)
        if self.lora_down is not None:
            self.lora_down = fn(self.lora_down)
        if self.lora_up is not None:
            self.lora_up = fn(self.lora_up)
        # [2026-06-02] BASIWAN-packed tensors MUST follow block-swap moves too.
        # Without this, the first forward call materializes ~12 MB packed
        # weight + scales/mins on GPU; subsequent block-swap-outs migrate
        # `self.weight` back to CPU but leave the packed tensors stranded
        # on GPU. After ~8-9 first-forward packs the 24 GB VRAM was
        # exhausted at p480_17f Q4_K (OOM at FFN[0] of block 0). Migrating
        # packed tensors with `fn` ties their lifetime to the block's
        # active device.
        if self._basiwan_packed is not None:
            mp = self._basiwan_packed
            mp.weight = fn(mp.weight)
            if mp.weight_hi is not None:
                mp.weight_hi = fn(mp.weight_hi)
            mp.scales = fn(mp.scales)
            if mp.mins is not None:
                mp.mins = fn(mp.mins)
        # Runtime scratch is device-local and must be rebuilt.
        self._basiwan_runtime = None
        return self


def prepack_basiwan_weights(module: nn.Module) -> int:
    """Eagerly pack all Q4_K/Q6_K weights for the BASIWAN path.

    Reason: the BareGGMLLinear.forward lazy-pack happens on first call.
    When that first call lands while block-swap has already placed the
    block on GPU and VRAM is tight, the `weight.cpu()` in pack_basiwan_weight
    fails with CUDA OOM (the D2H staging buffer can't be allocated).

    Call this immediately after the GGUF state dict has been loaded into
    the BareGGMLLinears and BEFORE block-swap is installed. At that point
    all weights are CPU-resident; packing is a pure CPU operation and
    incurs no GPU traffic.

    Returns the number of weights packed.
    """
    from .basiwan_q4_kernel import pack_basiwan_weight  # local import to avoid CUDA cost at import
    pack_basiwan_weight = pack_basiwan_weight  # local alias for the loop below
    n = 0
    for sub in module.modules():
        if not isinstance(sub, BareGGMLLinear):
            continue
        w = sub.weight
        if w is None or not is_quantized(w):
            continue
        qname = getattr(getattr(w, "tensor_type", None), "name", None)
        if qname not in {"Q4_K", "Q6_K"}:
            continue
        if sub._basiwan_packed is not None:
            continue
        sub._basiwan_packed = pack_basiwan_weight(w)
        # Free the original GGUF blob — BASIWAN owns dequant from here on. This
        # saves ~12.5 MB per attention Linear and ~34 MB per FFN Linear on the
        # active GPU device (block-swap was duplicating these alongside the
        # packed copy, hitting OOM at time_projection ~1:28 into step 0).
        sub.weight = None
        n += 1
    return n


def swap_linear_with_ggml(module: nn.Module) -> int:
    """Recursively replace `nn.Linear` children with `BareGGMLLinear`.
    Does NOT copy the source nn.Linear's weight/bias — those are
    nn.Parameter and would register as Parameter slots on the
    BareGGMLLinear, which would later reject the GGMLTensor (uint8)
    assignment. The caller is expected to fill `.weight` and `.bias`
    from the GGUF state dict after the swap. Returns the count of
    replacements made."""
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and not isinstance(child, BareGGMLLinear):
            bare = BareGGMLLinear(
                child.in_features, child.out_features,
                bias=(child.bias is not None),
            )
            # Defensive: ensure neither weight nor bias is in the
            # Parameter slot so later assignment of a GGMLTensor (uint8)
            # is accepted by nn.Module.__setattr__.
            bare._parameters.pop("weight", None)
            bare._parameters.pop("bias", None)
            setattr(module, name, bare)
            count += 1
        else:
            count += swap_linear_with_ggml(child)
    return count


__all__ = [
    "gguf_sd_loader",
    "GGMLTensor",
    "BareGGMLLinear",
    "prepack_basiwan_weights",
    "swap_linear_with_ggml",
    "dequantize_tensor",
    "is_quantized",
]
