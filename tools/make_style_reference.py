"""Style-reference maker for MOVA T2AV (the all-users generalization).

MOVA is I2V: its first-frame reference sets the SCENE + STYLE. To let a user type ANY prompt and
get that scene in their show's style, we MANUFACTURE a per-prompt reference here: SDXL + IP-Adapter
(InstantStyle style-only) takes a STYLE-reference frame from the show's training clips + the user's
prompt -> a styled, scene-matched still. That PNG is then fed to MOVA as --ref.

VALIDATED (env_mova, Windows shipping env) on a novel prompt: a Moral Orel clip frame as
the style ref + "a boy and a floating red basilisk dragon in front of a church" -> a clay boy +
clay red floating dragon + clay church (real claymation, brand-new scene, NO per-show training).
Training-free; generalizes to any show. Runs in env_mova (diffusers + the same Windows torch MOVA
uses), so the app spawns it with the env_mova python exactly like tools/mova_sample.py.

Recipe: InstantStyle style-only scale 0.6 (red/floating won;
claymation held; 1.0 was style-dominant and muted the prompt), guidance 7.0, 30 steps. Lower scale
-> prompt wins more; raise toward 1.0 if a show's style is being lost.

  make_style_reference.py --sdxl <base.safetensors> --style-ref <frame.png> --prompt "<text>"
                          --out <out.png> [--scale 0.6] [--steps 30] [--guidance 7.0]
                          [--width 1024] [--height 1024] [--seed 42]
ASCII. WINDOWS NOTE: pass a native Windows path (drive-letter form) for --sdxl -- diffusers
from_single_file rejects WSL /mnt paths as 'invalid URL'.
"""
import os, sys, argparse
os.environ.setdefault("PYTHONUTF8", "1"); os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# IP-Adapter repo + weights (h94/IP-Adapter): the SDXL vit-h adapter + its CLIP image encoder.
# Permissive (Apache-2.0); SDXL base sd_xl_base_1.0 is openrail++ (commercial-OK). Both fine for a
# free app. diffusers downloads/caches these on first run.
IP_REPO = "h94/IP-Adapter"
IP_SUBFOLDER = "sdxl_models"
IP_WEIGHT = "ip-adapter_sdxl_vit-h.safetensors"
IP_IMAGE_ENCODER = "models/image_encoder"
DEFAULT_NEG = "photorealistic, photo, real person, live action, 3d render, cgi, blurry, lowres, text, watermark, signature"


def log(m): print(f"[style-ref] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdxl", required=True, help="SDXL base .safetensors (Windows path on Windows)")
    ap.add_argument("--style-ref", required=True, help="in-style frame from the show's training clips")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--scale", type=float, default=0.6, help="InstantStyle style strength (tuned 0.6)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.0)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    from transformers import CLIPVisionModelWithProjection

    log(f"sdxl={a.sdxl}")
    log(f"style_ref={a.style_ref} prompt={a.prompt!r} scale={a.scale} steps={a.steps} guidance={a.guidance}")
    enc = CLIPVisionModelWithProjection.from_pretrained(
        IP_REPO, subfolder=IP_IMAGE_ENCODER, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_single_file(a.sdxl, image_encoder=enc, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(IP_REPO, subfolder=IP_SUBFOLDER, weight_name=IP_WEIGHT)
    # InstantStyle: inject the reference's STYLE only (up.block_0), not its content/layout, so the
    # PROMPT drives the scene and the reference drives the style.
    pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, float(a.scale), 0.0]}})
    pipe.enable_model_cpu_offload()

    style_img = Image.open(a.style_ref).convert("RGB")
    g = torch.Generator(device="cuda").manual_seed(int(a.seed))
    img = pipe(prompt=a.prompt, negative_prompt=a.negative, ip_adapter_image=style_img,
               num_inference_steps=int(a.steps), guidance_scale=float(a.guidance),
               height=int(a.height), width=int(a.width), generator=g).images[0]
    img.save(a.out)
    log(f"saved {a.out}")
    # Machine-readable: the app parses this to locate the produced reference (abs path).
    from pathlib import Path
    print(f"STYLE_REF_OUT|{Path(a.out).resolve()}", flush=True)


if __name__ == "__main__":
    main()
