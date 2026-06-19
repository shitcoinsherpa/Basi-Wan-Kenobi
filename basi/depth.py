"""#388 Depth-control extraction for VACE restyle. Runs Depth-Anything-V2-Small
(Apache-2.0; the Large variant is CC-BY-NC — ship Small only) per frame and
formats the result as the EXACT control-video tensor the VACE VAE expects.

Verified format (ali-vilab vace/annotators/depth.py + VaceVideoProcessor):
  single-channel depth [0,1] per frame -> *255 -> uint8 -> replicate to 3ch
  -> normalize to [-1,1] via pixel/127.5 - 1.0.
The uint8 intermediary is load-bearing: it quantizes the depth distribution the
way the model was trained/used; skipping it shifts the distribution.

The format assembly (assemble_depth_control) is pure tensor logic and unit-
tested on CPU; the model load/inference (load_depth_model, extract_depth) needs
the GPU + transformers but is a thin wrapper over AutoModelForDepthEstimation.
"""
from __future__ import annotations

_DEPTH_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"  # Apache-2.0


def load_depth_model(cache_dir=None, device="cuda"):
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    proc = AutoImageProcessor.from_pretrained(_DEPTH_MODEL, cache_dir=cache_dir)
    model = AutoModelForDepthEstimation.from_pretrained(
        _DEPTH_MODEL, cache_dir=cache_dir).eval()
    if device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    return model, proc


def _depth_frame_01(depth_raw):
    """Per-frame min-max normalize a raw depth map to [0,1] (ali-vilab does
    per-image normalization). depth_raw: (H,W) float tensor."""
    import torch
    lo = depth_raw.min()
    hi = depth_raw.max()
    return (depth_raw - lo) / (hi - lo + 1e-6)


def assemble_depth_control(depth01_frames):
    """Format a list/stack of per-frame [0,1] single-channel depth maps into the
    VACE control video tensor (3, F, H, W) in [-1,1], following the verified
    *255 -> uint8 -> replicate-3ch -> /127.5-1 path. Pure tensor logic.

    depth01_frames: (F, H, W) float tensor in [0,1].
    returns: (3, F, H, W) float tensor in [-1,1].
    """
    import torch
    d = depth01_frames.clamp(0, 1)
    u8 = (d * 255.0).round().clamp(0, 255).to(torch.uint8)     # quantize like the annotator
    rgb = u8.unsqueeze(0).repeat(3, 1, 1, 1).float()            # (3,F,H,W) replicate
    return rgb / 127.5 - 1.0                                    # -> [-1,1]


def extract_depth(frames_rgb, model, proc, batch=8):
    """frames_rgb: (3, F, H, W) in [-1,1] (the source video). Returns the depth
    control video (3, F, H, W) in [-1,1] at the SAME H,W. GPU path."""
    import torch
    from PIL import Image
    _, Fn, H, W = frames_rgb.shape
    imgs = (((frames_rgb.clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)
            .to(torch.uint8).permute(1, 2, 3, 0).cpu().numpy())   # (F,H,W,3)
    dev = next(model.parameters()).device
    out = []
    for i in range(0, Fn, batch):
        pil = [Image.fromarray(imgs[j]) for j in range(i, min(i + batch, Fn))]
        inp = proc(images=pil, return_tensors="pt").to(dev)
        with torch.no_grad():
            pred = model(**inp).predicted_depth          # (b, h', w')
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1).float(), size=(H, W),
            mode="bicubic", align_corners=False).squeeze(1)   # (b,H,W)
        for k in range(pred.shape[0]):
            out.append(_depth_frame_01(pred[k]).cpu())
    depth01 = torch.stack(out, dim=0)                    # (F,H,W) in [0,1]
    return assemble_depth_control(depth01)
