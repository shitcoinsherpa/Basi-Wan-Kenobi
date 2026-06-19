# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified from Wan-Video/Wan2.2 (Apache-2.0) for BASI WAN K3N0B1: GGUF
# quantized path, block-swap offload, persistent worker, I2V graft, tiled
# VAE, profiling. See THIRD_PARTY_LICENSES.md.
import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    FLASH_ATTN_2_AVAILABLE = False

import warnings

# [Faster-Wan2.2 P28] Persistent cache of (q_lens, cu_seqlens) tensors keyed on
# (batch, seqlen, device). Stock built these per-call via torch.tensor([lq]*b).to(device)
# + torch.cat([zeros, lens]).cumsum().to(device); each call yielded freshly-allocated
# int32 tensors at different addresses. flash_attn_varlen_func captured those
# pointers inside a CUDA Graph; on replay the kernel read stale pointers and aborted
# with "illegal memory access" past 5 chained blocks. Persistent cache → stable
# addresses across all captured calls.
_LENS_CACHE: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}


def _cached_lens_and_cumsum(b: int, length: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (b, length, device.type, device.index)
    cached = _LENS_CACHE.get(key)
    if cached is not None:
        return cached
    q_lens = torch.tensor([length] * b, dtype=torch.int32, device=device)
    cu = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32)
    _LENS_CACHE[key] = (q_lens, cu)
    return q_lens, cu

__all__ = [
    'flash_attention',
    'attention',
]


def _scaled_dot_product_attention_fallback(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    causal=False,
    dtype=torch.bfloat16,
    prefer_cudnn=False,
):
    if q_lens is not None or k_lens is not None:
        warnings.warn(
            'Padding mask is disabled when using scaled_dot_product_attention. It can have a significant impact on performance.'
        )
    attn_mask = None

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    # [Faster-Wan2.2 P29] When called from a CUDA Graph capture (or anyone who
    # wants the kernel-fused attention path that is graph-replay-safe), force
    # cuDNN flash attention. cuDNN silently refuses non-contiguous inputs
    # (transpose returned a view) and falls through to MATH which is ~46×
    # slower at our shapes. So .contiguous() the (now-(B,H,L,D)) tensors before
    # invoking SDPA with the cuDNN backend pinned. Measured 2026-05-28:
    # contig+cuDNN-in-graph ≈ 8 ms per attn call vs 368 ms MATH.
    if prefer_cudnn:
        # cuDNN flash attention accepts the (B, H, L, D) view that .transpose(1, 2)
        # produces; no need to .contiguous() here (the contig copies were adding
        # 36GB/replay of memory traffic at Wan-Lightning scale, dominating the
        # captured graph wall time).
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel([SDPBackend.CUDNN_ATTENTION]):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)
        return out.transpose(1, 2).contiguous()

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, is_causal=causal, dropout_p=dropout_p)

    return out.transpose(1, 2).contiguous()


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    # [Faster-Wan2.2 P29] When a CUDA Graph capture is underway, flash_attn
    # 2.8.3 reliably aborts the captured graph past ~4 calls (illegal memory
    # access on RTX 4090; minimal repro 2026-05-28). flash_attn's internal
    # workspace/softmax_lse buffer becomes stale on the 5th captured call.
    # SDPA's attention kernels ARE graph-replay-safe. Route to SDPA when
    # capturing so the captured graph stays valid. Eager paths keep flash_attn
    # for full speed.
    if (q.device.type == 'cuda'
            and hasattr(torch.cuda, 'is_current_stream_capturing')
            and torch.cuda.is_current_stream_capturing()):
        return _scaled_dot_product_attention_fallback(
            q=q, k=k, v=v, q_lens=q_lens, k_lens=k_lens,
            dropout_p=dropout_p, causal=causal, dtype=dtype,
            prefer_cudnn=True,
        )

    # [Faster-Wan2.2 P32 v2 — 2026-05-31] SageAttention 2 (real v2.2.0+).
    # PRIOR (v1.0.6 from PyPI): INT8 QK + FP16 PV (Triton). On Wan2.2-A14B FP8
    # weights this produced black/collapsed frames on ~25% of prompts —
    # non-deterministically; thu-ml/SageAttention issues #93, #221, #273
    # describe the same failure mode. Our bench saw CLIP-T 0.31→0.15 on 2/8
    # prompts solo, recovered when stacked with TAEHV/tf32 (state-dependent).
    # NOW (v2.2.0 from source): on sm_89 the auto-dispatch routes to
    # `sageattn_qk_int8_pv_fp8_cuda(pv_accum_dtype="fp32+fp16")` — INT8 QK +
    # FP8 PV with a mixed FP32+FP16 accumulator engineered specifically for
    # Ada's 22-valid-bit FP8 MMA accumulator. v2 fixes the precision issue
    # that broke v1 under FP8 weights.
    #
    # Trigger conditions unchanged: uniform-batch self-attn, hd=128, noncausal
    # BF16, no window. Cross-attn (varlen) keeps flash_attn.
    # NaN/Inf + std-collapse safety nets retained as belt-and-suspenders.
    # BASIWAN_SAGEATTN_DEBUG=1 to log fallback reasons.
    import os as _os
    # [2026-06-05 audit JJ] Relaxed q_lens/k_lens gate. Original gate required
    # both None, which never fires in Wan (passes k_lens=seq_lens always). For
    # batch=1 production, seq_lens has 1 element == q.shape[1] (uniform). Allow
    # SageAttn when the lens are uniform (all equal to seq dim).
    def _is_uniform(lens, ref_len):
        if lens is None:
            return True
        try:
            return bool((lens == ref_len).all().item())
        except Exception:
            return False
    # Debug: print gate evaluation once
    if (_os.environ.get("BASIWAN_SAGEATTN_DEBUG") == "1"
            and not hasattr(flash_attention, "_sagegate_logged")):
        _g1 = _os.environ.get("BASIWAN_SAGEATTN") == "1"
        _g2 = q.device.type == 'cuda'
        _g3 = _is_uniform(q_lens, q.size(1))
        _g4 = _is_uniform(k_lens, k.size(1))
        _g5 = q.size(-1) == 128
        _g6 = q.dtype in (torch.bfloat16, torch.float16, torch.float32)
        _g7 = not causal
        _g8 = window_size == (-1, -1)
        _g9 = dropout_p == 0.0
        print(f"[P32-gate] env={_g1} cuda={_g2} q_uniform={_g3} k_uniform={_g4} "
              f"head128={_g5} (got{q.size(-1)}) dtype_ok={_g6} (got{q.dtype}) "
              f"non_causal={_g7} window_full={_g8} (got{window_size}) "
              f"no_dropout={_g9} | q_lens={q_lens} k_lens={k_lens} q.shape={tuple(q.shape)}",
              flush=True)
        flash_attention._sagegate_logged = True
    # [2026-06-05] Accept fp32 input too (FP32-norms ship default produces fp32
    # q/k). Cast q/k/v to bf16 before SageAttn since INT8 QK quant is dtype-
    # insensitive — the FP32 precision benefit lives in the upstream norm, not
    # in the QK matmul that gets quantized to INT8 anyway.
    if (_os.environ.get("BASIWAN_SAGEATTN") == "1"
            and q.device.type == 'cuda'
            and _is_uniform(q_lens, q.size(1)) and _is_uniform(k_lens, k.size(1))
            and q.size(-1) == 128
            and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and not causal
            and window_size == (-1, -1)
            and dropout_p == 0.0):
        try:
            import sageattention as _sa
            _q = q if q.dtype != torch.float32 else q.to(torch.bfloat16)
            _k = k if k.dtype != torch.float32 else k.to(torch.bfloat16)
            _v = v if v.dtype != torch.float32 else v.to(torch.bfloat16)
            x = _sa.sageattn(_q, _k, _v, tensor_layout="NHD",
                             is_causal=False, sm_scale=softmax_scale)
            if torch.isnan(x).any() or torch.isinf(x).any():
                raise RuntimeError("SageAttention produced NaN/Inf")
            # Black-output detector: any feature dim with effectively-zero variance
            # over the seq dimension was the v1 failure mode. v2 should not hit
            # this, but the check costs ~negligible and catches future regressions.
            _std = x.float().std(dim=1).mean().item()
            if _std < 1e-5:
                raise RuntimeError(f"SageAttention output collapsed (std={_std:.2e})")
            return x.type(out_dtype) if 'out_dtype' in dir() else x.type(q.dtype)
        except (ImportError, RuntimeError, AttributeError) as _e:
            if _os.environ.get("BASIWAN_SAGEATTN_DEBUG") == "1":
                warnings.warn(f"[P32] SageAttention fallback to flash_attn: {_e}")
            # Fall through to flash_attn path below.
    if q.device.type != 'cuda' or q.size(-1) > 256 or (
        not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE
    ):
        return _scaled_dot_product_attention_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            causal=causal,
            dtype=dtype,
        )

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # [Faster-Wan2.2 P28] Track whether the caller supplied per-sample lens or
    # whether we're in the uniform-batch case. Uniform case → fixed-batch
    # `flash_attn_func` (graph-safe). Varlen case → `flash_attn_varlen_func`
    # with cached cu_seqlens (stable addresses across captured calls).
    _uniform_q = q_lens is None
    _uniform_k = k_lens is None
    if _uniform_q:
        q = half(q.flatten(0, 1))
        q_lens, cu_q = _cached_lens_and_cumsum(b, lq, q.device)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))
        cu_q = None  # computed below
    if _uniform_k:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens, cu_k = _cached_lens_and_cumsum(b, lk, k.device)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))
        cu_k = None

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # [Faster-Wan2.2 P1] BF16→FP16 recast: FP16 FA-2 kernel uses FP16-accum
    # (2x tensor-core throughput) AND lets Br=128 fit without register spill.
    # Measured 1.60-1.70x speedup on Wan-typical shapes (b=1, hd=128, noncausal,
    # seq>=32k) on RTX 4090. Max abs diff vs BF16: 5e-4 — safely below diffusion
    # sampling noise floor. Triggers ONLY when conditions match FP16 kernel reqs.
    # Disable with env BASIWAN_NO_FP16_RECAST=1.
    import os
    _recast_to_fp16 = (
        q.dtype is torch.bfloat16
        and q.shape[-1] == 128
        and not causal
        and window_size == (-1, -1)
        and dropout_p == 0.0
        and os.environ.get("BASIWAN_NO_FP16_RECAST") != "1"
    )
    if _recast_to_fp16:
        orig_dtype = q.dtype
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)

    # P28: reuse cached cumsum when user didn't supply q_lens/k_lens, else compute.
    if cu_q is None:
        cu_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)
    if cu_k is None:
        cu_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
            0, dtype=torch.int32).to(q.device, non_blocking=True)

    # [Faster-Wan2.2 P28b] When the user didn't supply per-sample q_lens/k_lens
    # (the Wan case — uniform length per batch), use the fixed-batch
    # `flash_attn_func` instead of `flash_attn_varlen_func`. The varlen path
    # has internal workspace/state that breaks CUDA Graph replay past ~5
    # captured calls (silent abort / illegal memory access on RTX 4090, observed
    # 2026-05-28). The fixed-batch path uses a simpler call signature with no
    # cu_seqlens pointers, and the cuDNN/FA2 backend keeps its workspace shape
    # constant across calls of the same (b, lq, h, d) input — graph-replay safe.
    _use_uniform_fa = (
        FLASH_ATTN_2_AVAILABLE
        and not FLASH_ATTN_3_AVAILABLE
        and _uniform_q and _uniform_k  # P28: tracked at top, no runtime probe (sync-free)
    )
    if _use_uniform_fa:
        # Re-shape to (B, L, H, D) since flash_attn_func expects unflattened batch.
        q_u = q.view(b, lq, *q.shape[1:])
        k_u = k.view(b, lk, *k.shape[1:])
        v_u = v.view(b, lk, *v.shape[1:])
        x = flash_attn.flash_attn_func(
            q_u, k_u, v_u,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
        )
        if _recast_to_fp16:
            x = x.to(orig_dtype)
        return x.type(out_dtype)

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output
    return x.type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    if FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE:
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )
    else:
        return _scaled_dot_product_attention_fallback(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            causal=causal,
            dtype=dtype,
        )
