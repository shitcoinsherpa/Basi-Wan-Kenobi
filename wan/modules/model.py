# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified from Wan-Video/Wan2.2 (Apache-2.0) for BASI WAN K3N0B1: GGUF
# quantized path, block-swap offload, persistent worker, I2V graft, tiled
# VAE, profiling. See THIRD_PARTY_LICENSES.md.
import json
import math
import os as _os

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


_sinusoidal_freq_cache: dict[tuple, torch.Tensor] = {}


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    # [Faster-Wan2.2 P14] FP32 not FP64. Position values are timestep ints (max 1000),
    # max sinusoid angle ≈ 1000 rad. FP32 ε≈1e-7 — overhead trig accuracy unnecessary.
    # FP64 temp tensors were 2× bandwidth/memory. At freq_dim=256 + B*seq_len=151200,
    # FP64 sinusoid+cos+sin+cat was ~310 MB temp; FP32 is ~155 MB.
    position = position.to(torch.float32)

    # [Faster-Wan2.2 P27] Cache the frequency table per (half, device) so we don't
    # allocate `torch.arange(half).to(device)` every call. The allocation is illegal
    # inside `torch.cuda.graph` capture; caching makes the function CUDA-Graph-safe.
    freq_key = (half, position.device, position.dtype)
    freqs = _sinusoidal_freq_cache.get(freq_key)
    if freqs is None:
        freqs = torch.pow(
            10000, -torch.arange(half, device=position.device, dtype=position.dtype).div(half))
        _sinusoidal_freq_cache[freq_key] = freqs

    sinusoid = torch.outer(position, freqs)
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    # [Faster-Wan2.2 P3] FP32 is sufficient for trig precision at max_seq_len=1024,
    # theta=10000 → rel err < 1e-7. Stock used FP64 → complex128 tables, 8 bytes per
    # element instead of 4. Memory and bandwidth halved with no quality impact.
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float32).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)  # complex64 (was complex128)
    return freqs


@torch.amp.autocast('cuda', enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # [Faster-Wan2.2 P34 — 2026-05-31] Chunked rope path for high-seq inference.
    # At p720_49f (seq=46800) the .float() upcast allocates 958 MB transient
    # AND the complex multiply produces another 958 MB intermediate. Chunking
    # the per-sample work along seq dim caps the spike at ~80 MB per chunk
    # (chunk=4096). Threshold via BASIWAN_ROPE_CHUNK_THRESH (default 16000).
    import os as _os
    _rope_chunk_thresh = int(_os.environ.get("BASIWAN_ROPE_CHUNK_THRESH", "16000"))
    _rope_chunk_size = int(_os.environ.get("BASIWAN_ROPE_CHUNK_SIZE", "4096"))

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # Build freqs_i once for this sample (small ~few MB).
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        if seq_len > _rope_chunk_thresh:
            # Chunked path: pre-allocate output in x dtype, process slices
            x_out = torch.empty(
                seq_len, n, x.size(-1), dtype=x.dtype, device=x.device)
            for _s in range(0, seq_len, _rope_chunk_size):
                _e = min(_s + _rope_chunk_size, seq_len)
                _xi = torch.view_as_complex(
                    x[i, _s:_e].float().reshape(_e - _s, n, -1, 2))
                _xi = _xi * freqs_i[_s:_e]
                _xr = torch.view_as_real(_xi).flatten(2)
                x_out[_s:_e].copy_(_xr.to(x.dtype))
            if x.size(1) > seq_len:
                x_out = torch.cat([x_out, x[i, seq_len:]])
            output.append(x_out)
        else:
            # [Faster-Wan2.2 P3] FP32 (was FP64) — full path at low seq.
            x_i = torch.view_as_complex(x[i, :seq_len].float().reshape(
                seq_len, n, -1, 2))
            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            # append to collection
            output.append(x_i)
    # [Faster-Wan2.2 P3] return input dtype (was always FP32 → downstream had to cast back).
    return torch.stack(output).to(x.dtype)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        # [Faster-Wan2.2 P34] Skip the .float() upcast at extreme seq — saves
        # 2× input-bytes transient (e.g. 740 MB → no transient at p720_81f
        # k-norm on (1, 75600, 40, 128) bf16). RMSNorm's normalization is
        # numerically stable in bf16 for activations within reasonable range.
        import os as _os
        _rms_bf16 = _os.environ.get("BASIWAN_RMS_BF16")
        if _rms_bf16 is None:
            _rms_bf16_active = (x.dim() >= 2 and x.shape[-2] > 50000)
        else:
            _rms_bf16_active = _rms_bf16 == "1"
        # Debug: log once per process to surface dtype/shape at norm call
        if not hasattr(WanRMSNorm, "_debug_logged"):
            print(f"[P34-debug RMSNorm] x.dtype={x.dtype} x.shape={tuple(x.shape)} "
                  f"_rms_bf16_active={_rms_bf16_active}", flush=True)
            WanRMSNorm._debug_logged = True
        if _rms_bf16_active and x.dtype in (torch.bfloat16, torch.float16):
            # [P34] self.weight may be fp32 (RMSNorm params often kept high-precision).
            # bf16 * fp32 = fp32, promoting throughout downstream. Cast weight inline.
            # Also: PyTorch's bf16 autocast may promote rsqrt/mean to fp32 for stability,
            # so disable autocast in this stage too.
            with torch.amp.autocast('cuda', enabled=False):
                _w = self.weight if self.weight.dtype == x.dtype else self.weight.to(x.dtype)
                return self._norm(x) * _w
        # Force bf16 path even if dtype is fp32 — at extreme seq we can't afford the upcast
        if _rms_bf16_active:
            orig = x.dtype
            _w_bf = self.weight if self.weight.dtype == torch.bfloat16 else self.weight.to(torch.bfloat16)
            return (self._norm(x.bfloat16()) * _w_bf).to(orig)
        # [Faster-Wan2.2 P4 — REJECTED] torch.nn.functional.rms_norm in PyTorch 2.7
        # is NOT a fused CUDA kernel — it decomposed back to Python ops + measured
        # -17% on dim=5120 (the hot Wan path). Original Python loop is faster on Ada.
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        import torch.nn.functional as F
        # [Faster-Wan2.2 P34] At extreme seq (p720_81f: 75600 tokens × 5120
        # dim bf16 = 740 MB), the P11 fp32 upcast produces 1.5 GB transient
        # plus another 1.5 GB for the fp32 norm output. PyTorch 2.6+ supports
        # bf16 layer_norm natively (verified preserves dtype). Auto-trigger
        # at seq > 50000. Override via BASIWAN_LN_BF16=1/0.
        import os as _os
        _ln_bf16 = _os.environ.get("BASIWAN_LN_BF16")
        if _ln_bf16 is None:
            _ln_bf16_active = (x.dim() >= 2 and x.shape[-2] > 50000)
        else:
            _ln_bf16_active = _ln_bf16 == "1"
        if _ln_bf16_active and x.dtype in (torch.bfloat16, torch.float16):
            # [P34] PyTorch's bf16 autocast force-promotes layer_norm to fp32 for
            # stability, returning fp32 even with bf16 input. Disable autocast
            # locally to keep this stage in bf16.
            with torch.amp.autocast('cuda', enabled=False):
                return F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        # [Faster-Wan2.2 P11] FP8-safe: use explicit .to(torch.float32) instead of .float().
        # When upstream nn.Linear is FP8-quantized via torchao, intermediate tensors flowing
        # in carry quantized-tensor wrappers. .float() on those can silently no-op (keep
        # bfloat16), then F.layer_norm raises "expected Float but found BFloat16". .to(dtype)
        # always allocates a fresh tensor with the requested dtype. Identical to .float()
        # for plain tensors; correct for quantized wrappers.
        orig = x.dtype
        return F.layer_norm(x.to(torch.float32), self.normalized_shape,
                            None, None, self.eps).to(orig)


class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        # [Faster-Wan2.2 P2 — REJECTED] Tried fused QKV (1× dim→3*dim GEMM) hoping
        # for launch-overhead + L2-reuse win. Measured -2.2% on 14B 480p, -0.6% on
        # 720p — cuBLAS already saturates the 5120×5120 GEMM; widening to 5120×15360
        # gives no proportional benefit on Ada. Three separate launches stay.
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):
    # [Faster-Wan2.2 P12] context K/V projection cache.
    # Context (text embeddings) is constant across all denoising steps within a
    # single generation. Original code recomputes k=norm_k(self.k(context)) and
    # v=self.v(context) on every block × every step × cond+uncond = 40 × 50 × 2
    # = 4000 redundant GEMMs per video. With per-instance cache keyed on
    # id(context), we project once per generation per branch and reuse for the
    # remaining 3999 calls. Cache invalidates automatically when caller passes
    # a different context tensor object. Set BASIWAN_NO_CROSS_KV_CACHE=1 to
    # disable (e.g. for debugging / variable-context scenarios).
    _ctx_cache_disabled = _os.environ.get("BASIWAN_NO_CROSS_KV_CACHE") == "1"

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query (always — depends on x)
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        # compute key, value (cached on context id)
        if not self._ctx_cache_disabled:
            ctx_id = id(context)
            cache = getattr(self, "_ctx_kv_cache", None)
            if cache is not None and cache[0] == ctx_id:
                k, v = cache[1], cache[2]
            else:
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                self._ctx_kv_cache = (ctx_id, k, v)
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

    def reset_kv_cache(self):
        """Drop cached K/V tensors. Called by quality_gate.py between runs."""
        if hasattr(self, "_ctx_kv_cache"):
            del self._ctx_kv_cache


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        mod_scratch=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            mod_scratch(Tensor, optional): [Faster-Wan2.2 P31] Shared (B,L,6,C)
                scratch tensor provided by WanModel.forward so the per-block
                `(modulation + e)` computation can land in a single reused buffer
                instead of allocating a fresh tensor per block. Required for
                CUDA Graph replay at 40-block scale without blowing the graph
                private pool. When None we fall back to the original allocator
                path (correct, but pool-hungry under capture).
        """
        # [Faster-Wan2.2 P9] AdaLN-Zero modulation: drop FP32 autocast round-trips.
        # Stock: norm(x).float() + e (FP32) → ~750 MB allocator hit per block per step
        # at 14B 720p × 40 blocks × 50 steps × 2 (CFG) = many PB of pointless bandwidth.
        # Modulation values are randn/sqrt(dim) ≈ 0.014 — well within BF16 precision.
        # Disable with BASIWAN_NO_BF16_MOD=1 to fall back to stock FP32 path.
        # Debug: trace x dtype through block first time
        if not hasattr(WanAttentionBlock, "_p34_block_logged"):
            print(f"[P34-debug Block] entry x.dtype={x.dtype} e.dtype={e.dtype} "
                  f"mod_scratch={mod_scratch.dtype if mod_scratch is not None else None}", flush=True)
            WanAttentionBlock._p34_block_logged = True
        # [P34] Force-take bf16 path even if x somehow drifted to fp32, since
        # legacy path's `(modulation + e)` out-of-place add allocates 8 GB at
        # p720_81f. Set BASIWAN_FORCE_BF16_MOD=0 to restore strict check.
        _bf16_mod = (_os.environ.get("BASIWAN_NO_BF16_MOD") != "1") and (
            _os.environ.get("BASIWAN_FORCE_BF16_MOD", "1") == "1"
            or x.dtype == torch.bfloat16
        )

        if _bf16_mod:
            # [Faster-Wan2.2 P31] In-place modulation+e via shared scratch when
            # provided. mod_scratch is allocated once at the model level (P31 in
            # model.forward) and reused across all 40 blocks — captured graph
            # pool sees a single 480 MB allocation instead of 40 × 480 MB.
            mod = self.modulation.to(x.dtype)  # (1, 6, C) — small, deterministic
            if mod_scratch is not None:
                mod_scratch.copy_(e)
                mod_scratch.add_(mod.unsqueeze(0))
                e_bf = mod_scratch.chunk(6, dim=2)
                _restore_e = False
            else:
                # [Faster-Wan2.2 P34 — 2026-05-31] At extreme seq (p720_81f:
                # e is 4.4 GB), the out-of-place `(mod.unsqueeze(0) + e)` add
                # OOMs because it allocates a fresh 4.4 GB tensor alongside e.
                # In-place `e.add_(mod)` mutates e to e+mod, takes chunks (which
                # are VIEWS of e), then we sub_(mod) at end of block to restore.
                # Net: zero transient for the modulation tensor; trades graph-
                # capture friendliness (e is mutated mid-graph) for memory.
                # Caller is the model loop in WanModel.forward; e is built fresh
                # for that forward — safe to mutate-and-restore here.
                e.add_(mod.unsqueeze(0))
                e_bf = e.chunk(6, dim=2)
                _restore_e = True

            # [Faster-Wan2.2 FFFw REJECTED 2026-06-06] Tried fused Triton kernel
            # per-token-AdaLN modulation in tools/gguf_vendor/fused_norm_mod_triton.py.
            # Kernel correctness self-test passes within bf16 rounding tolerance,
            # but e2e p720_17f cat_boxing measures -0.0115 composite Q regression
            # (above 0.005 noise band). Root cause: Triton kernel does ALL ops in
            # fp32 internally with single cast at store, while eager path casts to
            # bf16 between LN and modulation (3 intermediate bf16 quantizations).
            # The diffusion model is sensitive to those specific bf16 rounding
            # artifacts. The Triton path is numerically MORE precise but produces
            # slightly different bf16 outputs → drifts noise predictions → ~1.4%
            # composite Q drop. Modest wall win (-3.9%) not worth quality cost.
            # See tools/gguf_vendor/fused_norm_mod_triton.py (kept for future
            # bf16-rounding-matched variant) + memory/audit_FFFw_*_2026-06-06.md.
            _n1 = self.norm1(x)
            _scale = (1 + e_bf[1].squeeze(2))
            _shift = e_bf[0].squeeze(2)
            _attn_in = _n1 * _scale + _shift
            if not hasattr(WanAttentionBlock, "_p34_attn_in_logged"):
                print(f"[P34-debug Block] norm1(x).dtype={_n1.dtype} scale.dtype={_scale.dtype} "
                      f"shift.dtype={_shift.dtype} attn_in.dtype={_attn_in.dtype} "
                      f"e_bf[0].dtype={e_bf[0].dtype} e_bf[1].dtype={e_bf[1].dtype}", flush=True)
                WanAttentionBlock._p34_attn_in_logged = True
            y = self.self_attn(_attn_in, seq_lens, grid_sizes, freqs)
            x = x + y * e_bf[2].squeeze(2)

            def cross_attn_ffn(x, context, context_lens, e_bf):
                x = x + self.cross_attn(self.norm3(x), context, context_lens)
                y = self.ffn(
                    self.norm2(x) * (1 + e_bf[4].squeeze(2)) + e_bf[3].squeeze(2))
                x = x + y * e_bf[5].squeeze(2)
                return x

            x = cross_attn_ffn(x, context, context_lens, e_bf)
            # [P34] Restore e (we did e.add_(mod) above; reverse it so the
            # NEXT block sees the original e_block in WanModel.forward kwargs)
            if _restore_e:
                e.sub_(mod.unsqueeze(0))
            return x

        # Legacy FP32 path
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2),
            seq_lens, grid_sizes, freqs)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        # [Faster-Wan2.2 P15] BF16-native head modulation (mirrors P9 in WanAttentionBlock).
        # Stock path forced FP32 autocast for modulation chunk + head linear, which round-
        # trips x: BF16 → FP32 → BF16 (head output cast back to x.dtype). Modulation values
        # are randn/sqrt(dim) ≈ 0.014 — safely BF16-representable. P9 validated this pattern.
        # Env BASIWAN_NO_BF16_HEAD=1 reverts to FP32 autocast.
        _bf16_head = (_os.environ.get("BASIWAN_NO_BF16_HEAD") != "1") and (
            x.dtype == torch.bfloat16 or e.dtype == torch.bfloat16
        )
        if _bf16_head:
            # [P34] At p720_81f we force e to bf16 in WanModel; downstream x
            # may drift to fp32. Cast x to bf16 so the modulation+norm chain
            # stays in bf16 (saves 1.5 GB at this shape).
            _target = torch.bfloat16
            if x.dtype != _target:
                x = x.to(_target)
            e_bf = (self.modulation.unsqueeze(0).to(_target) + e.to(_target).unsqueeze(2)).chunk(2, dim=2)
            x = self.head(self.norm(x) * (1 + e_bf[1].squeeze(2)) + e_bf[0].squeeze(2))
            return x
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class VaceWanAttentionBlock(WanAttentionBlock):
    """[#388] VACE control block (ali-vilab/VACE wan/modules/model.py). A
    standard WanAttentionBlock plus:
      * before_proj (block 0 ONLY): fuses the patch-embedded 96-ch control into
        the main latent stream — c = before_proj(c) + x — so the control "enters"
        at the head of the vace stack.
      * after_proj (EVERY block): the hint tap — c_skip = after_proj(c) — added
        back into the main block stream after the corresponding main layer.
    Both are zero-initialized in TRAINING (untrained graft = no-op); the trained
    QuantStack GGUF carries non-zero weights — never re-init them. The orchestration
    (stack of 8 → 8 hints) lives in WanModel.forward_vace to keep BASIWAN's
    explicit-loop style and reuse the parent block forward verbatim."""

    def __init__(self, dim, ffn_dim, num_heads, window_size=(-1, -1),
                 qk_norm=True, cross_attn_norm=False, eps=1e-6, block_id=0):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm,
                         cross_attn_norm, eps)
        self.block_id = block_id
        if block_id == 0:
            self.before_proj = nn.Linear(dim, dim)
        self.after_proj = nn.Linear(dim, dim)


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    # diffusers' compile_repeated_blocks() compiles one kernel cache shared
    # across all instances of the listed class name. Without this, the call
    # raises ValueError("_repeated_blocks attribute is empty"). All 30 of our
    # Wan transformer blocks are WanAttentionBlock instances with identical
    # graph shape — sharing the cache cuts compile latency 30× on first step.
    _repeated_blocks = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 vace_layers=None,
                 vace_in_dim=96):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # [#388] VACE control branch (ali-vilab/VACE). Constructed ONLY when
        # vace_layers is given (Wan2.2-Fun-VACE-A14B config: [0,5,10,15,20,25,
        # 30,35], one vace_block per layer). When None — every existing T2V/I2V
        # model — none of this is built, so those models stay byte-identical.
        # vace_patch_embedding: Conv3d(96->dim, patch_size) embeds the 96-ch
        # control latent; vace_blocks: VaceWanAttentionBlock x len(vace_layers).
        self.vace_layers = vace_layers
        self.vace_in_dim = vace_in_dim
        if vace_layers is not None:
            self.vace_layers_mapping = {
                layer: idx for idx, layer in enumerate(vace_layers)}
            self.vace_patch_embedding = nn.Conv3d(
                vace_in_dim, dim, kernel_size=patch_size, stride=patch_size)
            self.vace_blocks = nn.ModuleList([
                VaceWanAttentionBlock(dim, ffn_dim, num_heads, window_size,
                                      qk_norm, cross_attn_norm, eps, block_id=i)
                for i in range(len(vace_layers))
            ])

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # [Faster-Wan2.2 P8] TeaCache (arXiv 2411.19108) — temporal feature cache.
        # Caches the block-residual across denoising steps; reuses when modulation
        # embedding e0's L1-distance signal predicts low drift in output. CFG-paired.
        #
        # NOT CALIBRATED FOR Wan2.2. The coefficients below are the published
        # Wan2.1-T2V-14B fit (from LiewFeng/TeaCache4Wan2.1). Wan2.2 is MoE A14B
        # with separate high_noise / low_noise expert WanModel instances — each
        # expert is a different network and needs its own degree-4 polynomial fit.
        # No published Wan2.2 coefficients exist (verified 2026-05-27). Quality
        # under these borrowed coefficients is UNMEASURED on Wan2.2 output.
        #
        # To use: requires explicit ack via env BASIWAN_TEACACHE_UNCALIBRATED=1
        # (else gate refuses to fire even when enable_teacache=True). Refit
        # harness: tools/teacache_calibrate.py (task #106).
        self.enable_teacache = False
        self.teacache_thresh = 0.15
        self.num_steps = 0
        self.ret_steps = 5  # warm-up: always compute
        self.use_ref_steps = True
        self.teacache_calibrated_for = 'wan2.1-t2v-14b'  # NOT wan2.2
        self.teacache_coefficients = [-5784.54137548, 5449.50911966,
                                       -1811.16591783, 256.27178429, -13.02252404]
        self.cnt = 0
        self.previous_e0_even = None
        self.previous_e0_odd = None
        self.previous_residual_even = None
        self.previous_residual_odd = None
        self.accumulated_rel_l1_distance_even = 0.0
        self.accumulated_rel_l1_distance_odd = 0.0
        self._teacache_warned = False
        self.teacache_expert_name = None
        self._teacache_calibration_path = _os.environ.get(
            'BASIWAN_TEACACHE_CALIBRATION')
        self._teacache_calibration_blob = None
        if self._teacache_calibration_path:
            self._load_teacache_calibration_blob()

        # initialize weights
        self.init_weights()

    def _load_teacache_calibration_blob(self):
        try:
            with open(self._teacache_calibration_path,
                      'r',
                      encoding='utf-8') as handle:
                self._teacache_calibration_blob = json.load(handle)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TeaCache calibration JSON from "
                f"{self._teacache_calibration_path!r}: {exc}") from exc

    @staticmethod
    def _extract_teacache_branch_coefficients(expert_blob, branch_name):
        branch_blob = expert_blob.get(branch_name)
        coeffs = None
        if isinstance(branch_blob, dict):
            coeffs = (branch_blob.get('coefficients')
                      or branch_blob.get('teacache_coefficients'))
        elif branch_blob is not None:
            coeffs = branch_blob

        if coeffs is None and isinstance(expert_blob.get('teacache_coefficients'),
                                         dict):
            coeffs = expert_blob['teacache_coefficients'].get(branch_name)

        if coeffs is None and isinstance(expert_blob.get('coefficients'), list):
            coeffs = expert_blob['coefficients']

        if not isinstance(coeffs, list) or len(coeffs) != 5:
            raise ValueError(
                f"TeaCache calibration for branch {branch_name!r} must provide "
                f"five polynomial coefficients, got {coeffs!r}")
        return [float(value) for value in coeffs]

    def _apply_teacache_calibration(self):
        if (self._teacache_calibration_blob is None
                or self.teacache_expert_name is None):
            return

        experts = self._teacache_calibration_blob.get('experts')
        if not isinstance(experts, dict):
            raise KeyError(
                "TeaCache calibration JSON must contain an 'experts' mapping")

        expert_blob = experts.get(self.teacache_expert_name)
        if not isinstance(expert_blob, dict):
            raise KeyError(
                f"TeaCache calibration JSON does not contain expert "
                f"{self.teacache_expert_name!r}")

        self.teacache_coefficients = {
            'cond': self._extract_teacache_branch_coefficients(
                expert_blob, 'cond'),
            'uncond': self._extract_teacache_branch_coefficients(
                expert_blob, 'uncond'),
        }
        self.teacache_calibrated_for = self._teacache_calibration_blob.get(
            'calibrated_for', self.teacache_calibrated_for)

    def set_teacache_expert_name(self, expert_name):
        self.teacache_expert_name = expert_name
        self._apply_teacache_calibration()
        return self

    def _teacache_coefficients_for_branch(self, branch_name):
        coeffs = self.teacache_coefficients
        if isinstance(coeffs, dict):
            branch_coeffs = coeffs.get(branch_name)
            if branch_coeffs is None:
                branch_coeffs = coeffs.get('default')
            if branch_coeffs is None:
                raise KeyError(
                    f"Missing TeaCache coefficients for branch {branch_name!r}")
            return branch_coeffs
        return coeffs

    def forward_vace(self, x, vace_context, seq_len, kwargs):
        """[#388] Run the VACE control stack → return one hint per vace_block.
        Faithful to ali-vilab forward_vace (verified): patch-embed the 96-ch
        control, pad to seq_len, then stream `c` through the vace_blocks. block 0
        fuses the main latent (c = before_proj(c) + x); every block taps a hint
        via after_proj while c continues. The Python-list collection is exactly
        equivalent to the upstream stack/unbind trick. vace_blocks receive the
        SAME modulation e, rope freqs, text context, seq_lens as the main blocks
        (kwargs), minus the shared mod_scratch (they run before the main loop;
        use the safe per-call allocator path)."""
        # patch-embed control → tokens, pad to seq_len (mirror the main x path)
        c = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
        c = [u.flatten(2).transpose(1, 2) for u in c]
        c = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
            for u in c
        ])
        vkwargs = dict(kwargs)
        vkwargs['mod_scratch'] = None   # don't share the main loop's scratch
        hints = []
        for i, vblock in enumerate(self.vace_blocks):
            if i == 0:
                c = vblock.before_proj(c) + x
            c = WanAttentionBlock.forward(vblock, c, **vkwargs)
            hints.append(vblock.after_proj(c))
        return hints

    def forward(
        self,
        x,
        t,
        context,
        seq_len,
        y=None,
        vace_context=None,
        vace_context_scale=1.0,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        # [Faster-Wan2.2 P34 — 2026-05-31] At extreme seq (p720_81f), the fp32
        # e_block tensor is 8.85 GB (1×75600×6×5120 × 4 bytes) — alone this
        # exceeds half of 24 GB VRAM. Switch the time computation to bf16 to
        # halve it to 4.4 GB. Sinusoidal positional embedding values are in
        # [-1, 1] and the time/projection Linears are FP8 — bf16 precision is
        # plenty. Override via BASIWAN_TIME_FP32=1.
        # x can be a list (per-sample), unwrap to get dtype/shape
        _x_dtype = (x[0].dtype if isinstance(x, (list, tuple)) else x.dtype)
        import os as _os
        _time_fp32 = _os.environ.get("BASIWAN_TIME_FP32") == "1"
        _time_bf16 = (not _time_fp32) and (_x_dtype == torch.bfloat16) and (seq_len > 50000)
        if not hasattr(WanModel, "_p34_debug_logged"):
            print(f"[P34-debug WanModel] _x_dtype={_x_dtype} seq_len={seq_len} "
                  f"_time_bf16={_time_bf16} _time_fp32={_time_fp32}", flush=True)
            WanModel._p34_debug_logged = True
        if _time_bf16:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                bt = t.size(0)
                t = t.flatten()
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).to(torch.bfloat16))
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            # Force bf16 regardless of FP8 dispatch output_dtype quirks
            if e0.dtype != torch.bfloat16:
                e0 = e0.to(torch.bfloat16)
            if e.dtype != torch.bfloat16:
                e = e.to(torch.bfloat16)
            if not hasattr(WanModel, "_p34_e0_logged"):
                print(f"[P34-debug WanModel] e0.dtype={e0.dtype} e0.shape={tuple(e0.shape)}", flush=True)
                WanModel._p34_e0_logged = True
        else:
            with torch.amp.autocast('cuda', dtype=torch.float32):
                bt = t.size(0)
                t = t.flatten()
                if not hasattr(WanModel, '_te_debug'):
                    te0 = self.time_embedding[0]
                    print(f"[DEBUG WanModel] time_embedding[0] type={type(te0).__name__}", flush=True)
                    print(f"[DEBUG WanModel] time_embedding[0] weight: ", flush=True)
                    if hasattr(te0, 'weight'):
                        w = te0.weight
                        print(f"  shape={tuple(w.shape) if w is not None else None} dtype={w.dtype if w is not None else None}", flush=True)
                    if hasattr(te0, '_basiwan_packed'):
                        print(f"  _basiwan_packed={te0._basiwan_packed}", flush=True)
                    print(f"[DEBUG WanModel] t.shape={tuple(t.shape)} t.dtype={t.dtype}", flush=True)
                    print(f"[DEBUG WanModel] bt={bt} seq_len={seq_len} freq_dim={self.freq_dim} dim={self.dim}", flush=True)
                    WanModel._te_debug = True
                e = self.time_embedding(
                    sinusoidal_embedding_1d(self.freq_dim,
                                            t).unflatten(0, (bt, seq_len)).float())
                e0 = self.time_projection(e).unflatten(2, (6, self.dim))
                assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        # [Faster-Wan2.2 P18] Cache text_embedding output keyed on input context list[0]
        # identity. Context is constant per generation (and per CFG branch) — recomputing
        # the pad + text_embedding GEMM every step is wasted. Also: this fix makes P12's
        # WanCrossAttention K/V cache actually work — P12 keys on id(context_post_embed),
        # but stock recomputes that tensor every step, so the cache never hit before this.
        # The cache stores up to 2 entries (one for cond, one for uncond).
        context_lens = None
        _ctx_key = id(context[0]) if (context and len(context) > 0) else None
        if not hasattr(self, '_text_emb_cache'):
            self._text_emb_cache = {}
        if _ctx_key in self._text_emb_cache:
            context = self._text_emb_cache[_ctx_key]
        else:
            context = self.text_embedding(
                torch.stack([
                    torch.cat(
                        [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                    for u in context
                ]))
            # Cap at 4 entries (cond+uncond per expert is the max for Wan2.2 MoE);
            # evict oldest if full.
            if len(self._text_emb_cache) >= 4:
                self._text_emb_cache.pop(next(iter(self._text_emb_cache)))
            self._text_emb_cache[_ctx_key] = context

        # [Faster-Wan2.2 P30] Pre-convert e0 (FP32) to the runtime dtype ONCE.
        # Each WanAttentionBlock previously did `e.to(x.dtype)` per block, which
        # in eager just transiently allocates and frees ~960MB BF16, but in a
        # captured CUDA Graph the per-block alloc gets pinned in the private
        # pool — 40 blocks × ~960MB at preview shape = 38 GB pool, blowing
        # past the 24GB GPU and causing replay thrashing / fault. Pre-convert
        # so the block path sees a constant tensor: no per-block alloc inside
        # the graph. Eager paths get a marginal speedup too (1 conversion vs 40).
        # When BASIWAN_NO_BF16_MOD=1 (legacy FP32 ada-LN path requested),
        # WanAttentionBlock falls into the `assert e.dtype == torch.float32`
        # branch — keep e_block in FP32 in that case so the assert passes.
        _legacy_fp32_mod = _os.environ.get("BASIWAN_NO_BF16_MOD") == "1"
        _block_dtype = (torch.float32 if _legacy_fp32_mod
                        else (x.dtype if torch.is_tensor(x) else torch.bfloat16))
        e_block = e0.to(_block_dtype) if e0.dtype != _block_dtype else e0

        # [Faster-Wan2.2 P31] Shared scratch tensor for the per-block
        # `(self.modulation + e)` intermediate. Without this each block alloc'd
        # ~480 MB BF16 at preview shape (or 2 GB at Lightning); inside a
        # captured CUDA Graph all 40 of those allocs get pinned in the private
        # pool, exhausting 24 GB. One shared scratch → constant 480 MB regardless
        # of block count. Reuses the same address across blocks via in-place
        # copy_/add_ in WanAttentionBlock.forward.
        # [Faster-Wan2.2 P34 — 2026-05-31] Drop scratch at extreme seq. At
        # p720_81f e_block is (1, 75600, 6, 5120) bf16 = 4.4 GB — the persistent
        # scratch buffer can't fit alongside everything else, even with all
        # other P34 mitigations. Falling back to per-block out-of-place add
        # (legacy pre-P31 path) costs 4.4 GB transient per block but each is
        # GC'd between blocks. Trades graph-pinning friendliness for fit.
        # Auto-disable when e_block.numel() × bytes_per_elem > 3 GB. Override
        # via BASIWAN_NO_MOD_SCRATCH=1 / =0.
        _no_scratch_env = _os.environ.get("BASIWAN_NO_MOD_SCRATCH")
        if _no_scratch_env is None:
            _bytes = e_block.numel() * e_block.element_size()
            _disable_scratch = _bytes > 3 * (1024 ** 3)
        else:
            _disable_scratch = _no_scratch_env == "1"
        if _disable_scratch:
            self._mod_scratch = None
        elif (not hasattr(self, '_mod_scratch')
                or self._mod_scratch is None
                or self._mod_scratch.shape != e_block.shape
                or self._mod_scratch.dtype != e_block.dtype
                or self._mod_scratch.device != e_block.device):
            self._mod_scratch = torch.empty_like(e_block)

        kwargs = dict(
            e=e_block,
            mod_scratch=self._mod_scratch,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens)
        # [Faster-Wan2.2 2026-06-02] cache-dit integration: its
        # CachedBlocks_Pattern_Base.forward signature is
        # (hidden_states, encoder_hidden_states, **kwargs). Our native
        # block iterations pass `context` only. When cache-dit has
        # wrapped this model (sets `_is_cached=True`), expose `context`
        # ALSO under the `encoder_hidden_states` alias so cache-dit's
        # positional contract is satisfied. The native-block path (no
        # cache-dit) skips this branch — passing the alias kwarg would
        # break Wan's typed-arg block.forward(self, x, e, seq_lens,
        # grid_sizes, freqs, context, context_lens, mod_scratch=None).
        if getattr(self, "_is_cached", False):
            kwargs["encoder_hidden_states"] = context

        # [Faster-Wan2.2 P8] TeaCache gate: skip block stack if modulation hasn't drifted.
        # Refuses to fire on Wan2.2 unless caller explicitly acks uncalibrated coefficients.
        _teacache_armed = (getattr(self, 'enable_teacache', False) and self.num_steps > 0
                           and _os.environ.get('BASIWAN_TEACACHE_UNCALIBRATED') == '1')
        if getattr(self, 'enable_teacache', False) and not _teacache_armed and not self._teacache_warned:
            import warnings
            warnings.warn(
                "TeaCache requested but coefficients are fit on Wan2.1-T2V-14B, "
                "NOT Wan2.2 (MoE A14B). Set BASIWAN_TEACACHE_UNCALIBRATED=1 to "
                "force-enable with unmeasured quality, or refit via "
                "tools/teacache_calibrate.py before relying on this path.",
                RuntimeWarning, stacklevel=2)
            self._teacache_warned = True
        # [#388] VACE control hints — the vace stack runs ONCE per denoising
        # step on the constant control latent, producing one hint per
        # vace_layer; injected additively after the mapped main blocks below.
        # None when no vace_context → the block loops stay byte-identical to the
        # non-VACE path. VACE + TeaCache is an untested combo (teacache caches
        # the main-block residual, which would wrongly bake in the hint adds) —
        # force-disable teacache for VACE forwards.
        vace_hints = None
        if vace_context is not None and self.vace_layers is not None:
            vace_hints = self.forward_vace(x, vace_context, seq_len, kwargs)
            _teacache_armed = False
        if _teacache_armed:
            import numpy as np
            cutoff_steps = self.num_steps - self.ret_steps
            modulated_inp = e0 if self.use_ref_steps else e
            is_even = (self.cnt % 2 == 0)
            branch_name = 'cond' if is_even else 'uncond'
            prev_e0 = self.previous_e0_even if is_even else self.previous_e0_odd
            acc_attr = 'accumulated_rel_l1_distance_even' if is_even else 'accumulated_rel_l1_distance_odd'
            res_attr = 'previous_residual_even' if is_even else 'previous_residual_odd'
            if (self.cnt < self.ret_steps or self.cnt >= cutoff_steps or prev_e0 is None):
                should_calc = True
                setattr(self, acc_attr, 0.0)
            else:
                rel = ((modulated_inp - prev_e0).abs().mean()
                       / prev_e0.abs().mean()).cpu().item()
                new_acc = getattr(self, acc_attr) + float(
                    np.poly1d(
                        self._teacache_coefficients_for_branch(branch_name)
                    )(rel))
                if new_acc < self.teacache_thresh:
                    should_calc = False
                    setattr(self, acc_attr, new_acc)
                else:
                    should_calc = True
                    setattr(self, acc_attr, 0.0)
            if is_even:
                self.previous_e0_even = modulated_inp.clone()
            else:
                self.previous_e0_odd = modulated_inp.clone()
            if not should_calc and getattr(self, res_attr) is not None:
                x = x + getattr(self, res_attr)
            else:
                ori_x = x.clone()
                for block in self.blocks:
                    x = block(x, **kwargs)
                setattr(self, res_attr, x - ori_x)
            self.cnt = (self.cnt + 1) % max(self.num_steps, 1)
        else:
            # [P34] Force x to bf16 at extreme seq — guarantees the entire
            # block stack stays in bf16 regardless of upstream autocast quirks.
            if (seq_len > 50000 and x.dtype != torch.bfloat16
                    and _os.environ.get("BASIWAN_FORCE_BLOCK_BF16", "1") == "1"):
                x = x.to(torch.bfloat16)
                if not hasattr(WanModel, "_p34_block_cast_logged"):
                    print(f"[P34-debug WanModel] cast x to bf16 before block loop", flush=True)
                    WanModel._p34_block_cast_logged = True
            for i, block in enumerate(self.blocks):
                x = block(x, **kwargs)
                # [#388] inject the VACE hint AFTER the mapped main block (the
                # verified injection point: x = x + hints[block_id]*scale).
                if vace_hints is not None and i in self.vace_layers_mapping:
                    _h = vace_hints[self.vace_layers_mapping[i]]
                    x = x + _h.to(x.dtype) * vace_context_scale

        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

    @torch.no_grad()
    def apply_lora_safetensors(self, lora_path, strength=1.0):
        """[Faster-Wan2.2 P19] Apply a Wan2.2-Lightning LoRA in-place by merging
        lora_up @ lora_down (scaled by alpha/rank) into the base Linear weights.

        Format: lightx2v/Wan2.2-Lightning safetensors with keys like
            diffusion_model.blocks.0.cross_attn.k.lora_down.weight  (rank, in_dim)
            diffusion_model.blocks.0.cross_attn.k.lora_up.weight    (out_dim, rank)
            diffusion_model.blocks.0.cross_attn.k.alpha             scalar

        After merge, base weight is W_base + strength * (alpha/rank) * (up @ down).
        Resulting model runs at 4 steps + no-CFG (per Lightning training recipe).
        """
        from safetensors import safe_open
        merged = 0; skipped = 0
        # Group keys by module path
        modules_by_path = {}
        with safe_open(lora_path, framework='pt') as f:
            keys = list(f.keys())
            for k in keys:
                if not k.startswith('diffusion_model.'):
                    skipped += 1
                    continue
                rest = k[len('diffusion_model.'):]
                # Key formats:
                #   blocks.0.cross_attn.k.lora_down.weight  → module 'blocks.0.cross_attn.k', kind 'lora_down'
                #   blocks.0.cross_attn.k.lora_up.weight    → kind 'lora_up'
                #   blocks.0.cross_attn.k.alpha             → kind 'alpha' (scalar, no .weight suffix)
                if rest.endswith('.alpha'):
                    module_path = rest[:-len('.alpha')]
                    kind = 'alpha'
                elif rest.endswith('.lora_down.weight'):
                    module_path = rest[:-len('.lora_down.weight')]
                    kind = 'lora_down'
                elif rest.endswith('.lora_up.weight'):
                    module_path = rest[:-len('.lora_up.weight')]
                    kind = 'lora_up'
                else:
                    skipped += 1
                    continue
                modules_by_path.setdefault(module_path, {})[kind] = k

            for module_path, parts in modules_by_path.items():
                if 'lora_down' not in parts or 'lora_up' not in parts:
                    skipped += 1; continue
                # Resolve target nn.Linear in self
                target = self
                try:
                    for p in module_path.split('.'):
                        target = target[int(p)] if p.isdigit() else getattr(target, p)
                except (AttributeError, IndexError, ValueError):
                    skipped += 1; continue
                if not isinstance(target, nn.Linear):
                    skipped += 1; continue
                lora_down = f.get_tensor(parts['lora_down'])  # (rank, in_dim)
                lora_up = f.get_tensor(parts['lora_up'])      # (out_dim, rank)
                rank = lora_down.shape[0]
                if 'alpha' in parts:
                    alpha = f.get_tensor(parts['alpha']).item()
                else:
                    alpha = rank
                scale = strength * (alpha / rank)
                delta = (lora_up @ lora_down) * scale  # (out_dim, in_dim)
                target.weight.add_(delta.to(target.weight.dtype).to(target.weight.device))
                merged += 1
        return {'merged': merged, 'skipped': skipped}

        # init output layer
        nn.init.zeros_(self.head.head.weight)
