# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modified from Wan-Video/Wan2.2 (Apache-2.0) for BASI WAN KENOBI: GGUF
# quantized path, block-swap offload, persistent worker, I2V graft, tiled
# VAE, profiling. See THIRD_PARTY_LICENSES.md.
import logging
import os

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = [
    'Wan2_1_VAE',
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)

        return super().forward(x)


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        # Cast gamma/bias to x.dtype — see vae2_2.py for rationale.
        gamma = self.gamma.to(x.dtype) if isinstance(self.gamma, torch.Tensor) else self.gamma
        bias = self.bias.to(x.dtype) if isinstance(self.bias, torch.Tensor) else self.bias
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * gamma + bias


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        PyTorch >=2.1 supports BF16 native nearest/nearest-exact;
        drop the FP32 round-trip (saves ~470 MB allocator traffic per call at the
        highest-resolution stage). Fallback for old PyTorch.
        """
        try:
            return super().forward(x)
        except RuntimeError:
            return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x):
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
        )
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3, 3, padding=1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self._decode_out_channels = 3
        self._decoder_temporal_scale = 2**sum(self.temperal_upsample)
        self._decoder_spatial_scale = 2**max(0, len(dim_mult) - 1)
        self._decode_tile_area_threshold = 2048
        # Encode-side tiling backstop in PIXEL area. Encode tiling
        # is primarily env-gated (BASIWAN_VAE_TILING=1) so it never surprises the
        # i2v/vace encode paths; this auto-threshold only kicks in above ~720p.
        self._encode_tile_area_threshold = 1280 * 720

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def _encode_full(self, x):
        """Untiled encoder pass -> pre-conv1 output. Byte-for-byte the original
        encode() inner loop (temporal causal chunks 1,4,4,...)."""
        t = x.shape[2]
        iter_ = 1 + (t - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        return out

    def _encode_tiled(self, x, tile_h=256, tile_w=256, overlap=32):
        """Spatial tiled ENCODE — mirror of _decode_tiled, in pixel->latent.

        Caps the encode peak (the 14.75GB hog at 480x832: 73 motion frames + ref
        encoded full-frame) by running the encoder on spatial PIXEL tiles and
        blending the pre-conv1 LATENT outputs with the same 2D Hann window used by
        the decode path. conv1 is 1x1 (pointwise) so it commutes with the blend —
        applied once by encode() on the assembled output. Causal rule (per the
        diffusers/kijai reference): each spatial tile gets a FRESH _enc_feat_map;
        the cache is threaded only across that tile's temporal chunks, NEVER across
        spatial tiles (sharing would feed one region's features into another's
        causal conv). Wan2.1 VAE has patch_size=None so no patchify is needed."""
        scale = self._decoder_spatial_scale  # pixel<->latent factor (8)
        b, _, t, H, W = x.shape
        latent_h, latent_w = H // scale, W // scale
        tile_h_latent, tile_w_latent, overlap_latent = self._tile_params_to_latent(
            tile_h, tile_w, overlap)
        h_starts = self._tile_starts(latent_h, tile_h_latent, overlap_latent)
        w_starts = self._tile_starts(latent_w, tile_w_latent, overlap_latent)
        iter_ = 1 + (t - 1) // 4
        enc_ch = self.z_dim * 2
        acc_dtype = torch.float32 if x.dtype in (torch.float16,
                                                 torch.bfloat16) else x.dtype
        out_accum = torch.zeros((b, enc_ch, iter_, latent_h, latent_w),
                                device=x.device, dtype=acc_dtype)
        out_weight = torch.zeros((1, 1, 1, latent_h, latent_w),
                                 device=x.device, dtype=acc_dtype)
        for h0 in h_starts:
            for w0 in w_starts:
                h1 = min(h0 + tile_h_latent, latent_h)
                w1 = min(w0 + tile_w_latent, latent_w)
                ph0, ph1 = h0 * scale, h1 * scale
                pw0, pw1 = w0 * scale, w1 * scale
                # Fresh causal cache for THIS spatial tile.
                self._enc_feat_map = [None] * self._enc_conv_num
                tile_out = None
                for i in range(iter_):
                    self._enc_conv_idx = [0]
                    if i == 0:
                        chunk = x[:, :, :1, ph0:ph1, pw0:pw1]
                    else:
                        chunk = x[:, :, 1 + 4 * (i - 1):1 + 4 * i, ph0:ph1, pw0:pw1]
                    enc_i = self.encoder(chunk, feat_cache=self._enc_feat_map,
                                         feat_idx=self._enc_conv_idx)
                    tile_out = enc_i if tile_out is None else torch.cat(
                        [tile_out, enc_i], 2)
                lh, lw = tile_out.shape[-2], tile_out.shape[-1]
                window = self._tile_window(
                    lh, lw, min(overlap_latent, lh), min(overlap_latent, lw),
                    has_top=h0 > 0, has_bottom=h1 < latent_h,
                    has_left=w0 > 0, has_right=w1 < latent_w,
                    device=x.device, dtype=acc_dtype)
                out_accum[:, :, :, h0:h0 + lh, w0:w0 + lw] += tile_out.to(acc_dtype) * window
                out_weight[:, :, :, h0:h0 + lh, w0:w0 + lw] += window
        eps = torch.finfo(out_weight.dtype).eps
        return (out_accum / out_weight.clamp_min(eps)).to(x.dtype)

    def encode(self, x, scale):
        self.clear_cache()
        if os.environ.get("BASIWAN_FORCE_FULL_VAE") == "1":
            tiled = False
        else:
            tiled = (os.environ.get("BASIWAN_VAE_TILING") == "1"
                     or x.shape[-2] * x.shape[-1] > self._encode_tile_area_threshold)
        out = self._encode_tiled(x) if tiled else self._encode_full(x)
        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]
        self.clear_cache()
        return mu

    def _decode_full(self, z):
        iter_ = z.shape[2]
        x = self.conv2(z)
        # Keep decode accumulation linear-time.
        outs = []
        for i in range(iter_):
            self._conv_idx = [0]
            outs.append(self.decoder(
                x[:, :, i:i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx))
        return torch.cat(outs, dim=2) if iter_ > 1 else outs[0]

    def _tile_starts(self, size, tile_size, overlap):
        if tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got {tile_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= tile_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than tile_size ({tile_size})")
        if tile_size >= size:
            return [0]
        step = tile_size - overlap
        starts = list(range(0, size - tile_size + 1, step))
        if starts[-1] != size - tile_size:
            starts.append(size - tile_size)
        return starts

    def _tile_params_to_latent(self, tile_h, tile_w, overlap):
        scale = self._decoder_spatial_scale

        def ceil_div(value):
            return max(1, (value + scale - 1) // scale)

        tile_h_latent = ceil_div(tile_h)
        tile_w_latent = ceil_div(tile_w)
        overlap_latent = 0 if overlap <= 0 else ceil_div(overlap)
        overlap_latent = min(
            overlap_latent,
            max(0, min(tile_h_latent, tile_w_latent) - 1),
        )
        return tile_h_latent, tile_w_latent, overlap_latent

    def _hann_ramp(self, length, device, dtype, descending=False):
        if length <= 0:
            return None
        if length == 1:
            ramp = torch.ones(1, device=device, dtype=dtype)
        else:
            ramp = torch.hann_window(length * 2,
                                     periodic=False,
                                     device=device,
                                     dtype=dtype)[:length]
            ramp_max = ramp.max()
            if ramp_max > 0:
                ramp = ramp / ramp_max
        if descending:
            ramp = torch.flip(ramp, dims=(0, ))
        return ramp

    def _tile_window(self,
                     height,
                     width,
                     overlap_h,
                     overlap_w,
                     has_top,
                     has_bottom,
                     has_left,
                     has_right,
                     device,
                     dtype):
        window_h = torch.ones(height, device=device, dtype=dtype)
        window_w = torch.ones(width, device=device, dtype=dtype)
        if has_top and overlap_h > 0:
            window_h[:overlap_h] *= self._hann_ramp(overlap_h, device, dtype)
        if has_bottom and overlap_h > 0:
            window_h[-overlap_h:] *= self._hann_ramp(
                overlap_h, device, dtype, descending=True)
        if has_left and overlap_w > 0:
            window_w[:overlap_w] *= self._hann_ramp(overlap_w, device, dtype)
        if has_right and overlap_w > 0:
            window_w[-overlap_w:] *= self._hann_ramp(
                overlap_w, device, dtype, descending=True)
        return (window_h.view(1, 1, 1, height, 1) *
                window_w.view(1, 1, 1, 1, width))

    def _clone_feat_cache(self, feat_map):
        cloned = []
        for feat in feat_map:
            if isinstance(feat, torch.Tensor):
                cloned.append(feat.clone())
            else:
                cloned.append(feat)
        return cloned

    def _decode_output_frames(self, latent_frames):
        if latent_frames <= 0:
            return 0
        # Causal temporal upsample warms up on the first latent chunk, then each
        # later chunk expands by the full temporal scale.
        return 1 + (latent_frames - 1) * self._decoder_temporal_scale

    def _decode_tiled(self, z, tile_h=256, tile_w=256, overlap=32):
        """Spatial tiled decode with 2D Hann blending in output-pixel space."""
        x = self.conv2(z)
        b, _, iter_, latent_h, latent_w = x.shape
        tile_h_latent, tile_w_latent, overlap_latent = self._tile_params_to_latent(
            tile_h, tile_w, overlap)
        h_starts = self._tile_starts(latent_h, tile_h_latent, overlap_latent)
        w_starts = self._tile_starts(latent_w, tile_w_latent, overlap_latent)
        out_h = latent_h * self._decoder_spatial_scale
        out_w = latent_w * self._decoder_spatial_scale
        overlap_out = overlap_latent * self._decoder_spatial_scale
        acc_dtype = torch.float32 if x.dtype in (torch.float16,
                                                 torch.bfloat16) else x.dtype

        out_accum = torch.zeros(
            (b, self._decode_out_channels, self._decode_output_frames(iter_),
             out_h, out_w),
            device=x.device,
            dtype=acc_dtype,
        )
        out_weight = torch.zeros(
            (1, 1, 1, out_h, out_w),
            device=x.device,
            dtype=acc_dtype,
        )
        # Cache pre-warm. Without this, every spatial
        # tile processes frame 0 with feat_cache=None then frames 1+ with
        # accumulated cache history. CausalConv3d behaves differently between
        # "first call (cache=None, no temporal pad)" and "later calls (cache
        # contains 2 prior frames, gets concatenated to input)". The frame-0→1
        # boundary in every tile produces visible temporal flicker.
        # Pre-warm cache by running one global frame-0 decode at quarter
        # resolution (bounds peak VRAM), then use the warm state as initial
        # cache for every spatial tile.
        # Default kept at "0" — Fix A pre-warm CRASHES at the
        # tile loop (NOT inside the try block). The try/except wraps the
        # PRE-WARM decoder call which completes successfully and populates
        # self._feat_map with H=45 entries (quarter-res). Then the tile loop
        # (lines 754-768, NO try/except) runs decoder on H=32 tiles using the
        # H=45 cache → RuntimeError at frame 2 of frame loop.
        # The "load-bearing prewarm" hypothesis (my earlier guess) was wrong.
        # The I-1.3 ship at composite Q 0.842 must have had this default at
        # "0" too. The today-vs-yesterday composite delta (0.7808 vs ~0.842)
        # has a different root cause — TBD. For now, prewarm OFF is what works.
        if iter_ > 1 and os.environ.get("BASIWAN_VAE_PREWARM", "0") == "1":
            try:
                # Quarter-resolution frame-0 sample for cache warm-up.
                # Slice latent_h and latent_w by stride 2 to keep tensor sizes small.
                sample_h_stride = 2 if latent_h >= 4 else 1
                sample_w_stride = 2 if latent_w >= 4 else 1
                sample_x = x[:, :, 0:1, ::sample_h_stride, ::sample_w_stride]
                prewarm_idx = [0]
                with torch.no_grad():
                    _ = self.decoder(
                        sample_x,
                        feat_cache=self._feat_map,
                        feat_idx=prewarm_idx,
                    )
                # Cache is now populated. Use as initial state for every tile.
                initial_feat_map = self._clone_feat_cache(self._feat_map)
                # Reset the persistent cache so subsequent decode() calls start fresh.
                self._feat_map = [None] * self._conv_num
            except Exception as _e:
                # Pre-warm failed (OOM, dtype mismatch, etc.): fall back to None.
                initial_feat_map = self._clone_feat_cache(self._feat_map)
        else:
            initial_feat_map = self._clone_feat_cache(self._feat_map)
        window_cache = {}

        for h0 in h_starts:
            for w0 in w_starts:
                h1 = min(h0 + tile_h_latent, latent_h)
                w1 = min(w0 + tile_w_latent, latent_w)
                has_top = h0 > 0
                has_bottom = h1 < latent_h
                has_left = w0 > 0
                has_right = w1 < latent_w
                # Wan2.1 decoder caches causal Conv3d state across frames. Each
                # spatial tile needs an independent cache history.
                tile_feat_map = self._clone_feat_cache(initial_feat_map)
                window = None
                out_h0 = h0 * self._decoder_spatial_scale
                out_w0 = w0 * self._decoder_spatial_scale
                out_t0 = 0

                for i in range(iter_):
                    conv_idx = [0]
                    decoded = self.decoder(
                        x[:, :, i:i + 1, h0:h1, w0:w1],
                        feat_cache=tile_feat_map,
                        feat_idx=conv_idx,
                    )
                    out_t1 = out_t0 + decoded.shape[2]
                    out_h1 = out_h0 + decoded.shape[-2]
                    out_w1 = out_w0 + decoded.shape[-1]
                    if window is None:
                        window_key = (
                            decoded.shape[-2],
                            decoded.shape[-1],
                            has_top,
                            has_bottom,
                            has_left,
                            has_right,
                        )
                        if window_key not in window_cache:
                            window_cache[window_key] = self._tile_window(
                                decoded.shape[-2],
                                decoded.shape[-1],
                                min(overlap_out, decoded.shape[-2]),
                                min(overlap_out, decoded.shape[-1]),
                                has_top,
                                has_bottom,
                                has_left,
                                has_right,
                                device=x.device,
                                dtype=acc_dtype,
                            )
                        window = window_cache[window_key]
                        out_weight[:, :, :, out_h0:out_h1, out_w0:out_w1] += window
                    elif decoded.shape[-2:] != window.shape[-2:]:
                        # spatial output dim must not change between chunks; window normalization assumes it.
                        raise RuntimeError(
                            f"tiled VAE: chunk {i} decoded HxW {tuple(decoded.shape[-2:])} "
                            f"differs from chunk 0's {tuple(window.shape[-2:])}; window cache invalid"
                        )

                    decoded_acc = decoded.to(acc_dtype) * window
                    out_accum[:, :, out_t0:out_t1, out_h0:out_h1, out_w0:out_w1] += decoded_acc
                    out_t0 = out_t1

        eps = torch.finfo(out_weight.dtype).eps
        return (out_accum / out_weight.clamp_min(eps)).to(x.dtype)

    def decode(self, z, scale):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        # BASIWAN_FORCE_FULL_VAE=1 overrides the
        # tile threshold and forces a single full-resolution decode. Diagnostic
        # for the today-vs-ship composite Q drift: both Marlin and F.linear
        # paths lose ~0.10 on imaging_quality, suggesting tiled-VAE is the
        # regression source. WARNING: needs ~16+ GB VRAM at p720_17f.
        if os.environ.get("BASIWAN_FORCE_FULL_VAE") == "1":
            tiled = False
        else:
            tiled = (os.environ.get("BASIWAN_VAE_TILING") == "1"
                     or z.shape[-2] * z.shape[-1] >
                     self._decode_tile_area_threshold)
        out = self._decode_tiled(z) if tiled else self._decode_full(z)
        self.clear_cache()
        return out

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device), assign=True)

    return model


class Wan2_1_VAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 device="cuda"):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            z_dim=z_dim,
        ).eval().requires_grad_(False).to(device)

    def encode(self, videos):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.encode(u.unsqueeze(0), self.scale).float().squeeze(0)
                for u in videos
            ]

    def decode(self, zs):
        with amp.autocast(dtype=self.dtype):
            return [
                self.model.decode(u.unsqueeze(0),
                                  self.scale).float().clamp_(-1, 1).squeeze(0)
                for u in zs
            ]
