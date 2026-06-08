# BASI WAN K3N0B1

Flux-Gym-style UI for **Wan2.2 video LoRA training** on consumer GPUs.

Drop in some video clips → get back a `.safetensors` LoRA compatible with the
[Faster-Wan2.2](../transformers/work/Faster-Wan2.2) inference fork, ComfyUI, kohya, peft.

## Status

**v0.1** — full Flux-Gym-style stack: 3-column UI, auto-caption button, generated
cache+train scripts, in-app spawn, ComfyUI symlink, preview render. End-to-end
smoke test (musubi-tuner subprocess against a real dataset) pending.

**Pinokio integration:** `pinokio.js` + `install.js` + `start.js` + `update.js`
ready in repo root. One-click install picks PyTorch index by detected GPU
(cu121 / cu128 nightly / rocm6.1).

| Component | Status |
|---|---|
| Spec + architecture decision | ✓ done |
| musubi-tuner backend integration | ✓ installed, smoke-test pending |
| Dataset validator (4n+1 frame snap, resolution bucketing) | ✓ done |
| 24G VRAM preset (FP8 base, rank 32, blocks_to_swap 10) | ✓ done |
| Gradio UI scaffold (Flux-Gym pattern) | ✓ done |
| Qwen2.5-VL auto-captioning (tier-aware 3B/7B-AWQ/7B; Qwen3-VL opt-in needs transformers>=4.57) | ✓ done |
| Live preview (Faster-Wan2.2 + Lightning, trainer-paused) | ✓ scaffold |
| Output format converters (comfyui/sd-scripts/peft) | pending |
| VSA training compat (verdict: not feasible without 1100-LOC musubi fork) | ✓ closed |
| TAEHV tiny-VAE for preview decode (Faster-Wan2.2 P26) | ✓ done |
| CUDA Graphs at Lightning shape (Faster-Wan2.2 P25) | ✓ shipped, bench pending |

## Stack

- **Backend**: [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) — actively maintained, native Wan2.2-A14B support, `--blocks_to_swap` for 24GB VRAM, sd-scripts safetensors output format
- **Frontend**: Gradio (3-column Flux-Gym layout); plain `gr.Textbox(autoscroll=True)` for live logs (last 200 lines), Stop button signals SIGTERM/KILL the script's process group
- **Inference / live preview**: [Faster-Wan2.2](../transformers/work/Faster-Wan2.2) Lightning+FP8+TAEHV path (~3-4 s at 832×480×17 steady-state)
- **Auto-caption**: Qwen2.5-VL (7B@24G / 7B-AWQ@16G / 3B@12G, tier-aware) — bilingual matches Wan's training distribution; Qwen3-VL-8B opt-in requires transformers≥4.57 (musubi pin keeps us at 4.56.1 by default)

## About Wan2.2 MoE — read this before training

Wan2.2-T2V-A14B is a **two-expert Mixture of Experts** along the diffusion timestep axis:
- **High-noise expert** — handles early denoising (timestep ≥ 0.875), responsible for coarse motion + composition.
- **Low-noise expert** — handles late denoising (timestep < 0.875), responsible for fine detail + texture.

The expert dropdown in the trainer maps to musubi's `--dit` and `--dit_high_noise` flags:

| Choice | What gets trained | Use when |
|---|---|---|
| **low** | Late-step expert only (single `--dit` = low-noise weights) | Texture / appearance / surface LoRAs — most character / style LoRAs |
| **high** | Early-step expert only (single `--dit` = high-noise weights) | Motion / pose / composition LoRAs |
| **both** | Dual-expert (`--dit` + `--dit_high_noise` + `--timestep_boundary 0.875`) | Max-quality LoRAs affecting both motion AND appearance; needs 40GB+ |

A "low"-only LoRA will not affect early-step (high-noise) behavior. If a character LoRA trained only on "low" doesn't transfer motion / pose correctly, retrain on "both" or "high".

## LoRA stacking warning (preview)

The live preview combines **two** LoRAs on top of the base weights:
1. Wan2.2-Lightning 4-step distillation LoRA (auto-applied at strength 1.0)
2. Your trained user LoRA (latest `.safetensors` in workspace)

With both at strength 1.0, output can blow out (over-saturated colors, artifacts). If your preview looks washed-out, lower the user-LoRA strength to ~0.7 via the `BASIWAN_USER_LORA_STRENGTH` env var.

## Lightning LoRA override

Default Lightning checkpoint: `Wan2.2-T2V-A14B-4steps-lora-250928`. Use a different distill:
```bash
export BASIWAN_LIGHTNING_LORA_DIR=/path/to/your/lightning-lora
```
…before launching the gym.

## VSA training/inference mismatch

If you train **without** VSA (the default), your trained LoRA will misbehave in any inference path with VSA enabled (artifacts, attention drift). The gym does not currently train with VSA (research closed: not feasible without a 1100-LOC musubi fork). Don't set `BASIWAN_NABLA=1` or other sparse-attention env vars when previewing.

## VRAM presets

| Preset | Card | rank | blocks_to_swap | max safe frames | FP8 base | optimizer |
|---|---|---|---|---|---|---|
| 8G | 4060 8GB | 8 | 40 | 17 | yes | adafactor |
| 12G | 3060 12GB / 4070 | 16 | 30 | 17 | yes | adafactor |
| 16G | 4060Ti 16GB / 4080 | 32 | 20 | 33 | yes | adamw8bit |
| **24G** | **3090 / 4090** | **32** | **10** | **49** | yes | adamw8bit |
| 40G+ | A100 40/80 / H100 | 64 | 0 | 81 | yes | adamw8bit + dual-expert |

Higher frame counts may still fit if your card has slack; the UI shows a may-OOM warning above the safe cap but doesn't lock the slider.

## Quickstart

### Via Pinokio (recommended)
1. Drop this repo into Pinokio's "Add Custom App" dialog (or publish to the index).
2. Click **Install** — Pinokio creates a venv, installs PyTorch matched to your GPU
   (cu121 / cu128 nightly / rocm6.1 / mps / cpu), then deps + musubi-tuner.
3. Click **Start** — Gradio comes up on `http://localhost:7860`.

### Manual (Linux/WSL)
```bash
python3.10 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install bitsandbytes
git clone https://github.com/kohya-ss/musubi-tuner.git ext/musubi-tuner
pip install -e ext/musubi-tuner

# verify install (imports, CUDA, ckpts, ffprobe, TAEHV auto-download)
python scripts/smoke_test.py

# (optional) verify the BASI pipeline end-to-end against the bundled sample clip
python scripts/smoke_test_pipeline.py

# launch
python app.py     # → http://localhost:7860
```

### Required checkpoints (auto-checked by smoke_test.py)
- `Wan2.2-T2V-A14B` base — set `BASI_CKPT_DIR` or place under
  `/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B/`
- `Wan2.2-Lightning` 4-step LoRA — `BASI_LIGHTNING_LORA_DIR` or
  `/mnt/d/Ai/checkpoints/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928/`
- `taew2_1.pth` (TAEHV) — auto-downloaded by smoke_test.py on first run.

## Layout

```
basi-wan-k3n0b1/
├── README.md
├── app.py                 # Gradio entry (3-col layout, auto-caption, sample gallery, preview)
├── pinokio.js             # Pinokio menu surface (Install / Start / Open / Outputs / Update)
├── install.js             # Pinokio install — venv, GPU-aware torch, musubi-tuner
├── start.js               # Pinokio start — launches Gradio on :7860
├── update.js              # Pinokio update — git pull + pip refresh
├── requirements.txt       # Python deps (gradio, qwen-vl-utils, transformers==4.56.1, etc.)
├── basi/
│   ├── presets.py         # 12G/16G/24G/40G+ configs + auto_select(detect_vram_gb)
│   ├── dataset.py         # ffprobe validator + 4n+1 + bucketing
│   ├── train.py           # musubi subprocess wrapper + cache.sh + train.sh generation
│   ├── caption.py         # Qwen2.5-VL auto-caption (tier-aware)
│   ├── preview.py         # preview render via Faster-Wan2.2 Lightning+FP8+TAEHV (P19/P24/P25/P26)
│   └── export.py          # [TODO] format converters
├── ext/
│   └── musubi-tuner/      # git clone of kohya-ss/musubi-tuner
└── outputs/               # per-LoRA artifacts (dataset/, sample/, preview/, *.safetensors)
```

## Wan2.2 model weights expected

The UI defaults to the standard Wan-AI checkpoint paths:
- `/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B/{high,low}_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors` (musubi auto-detects sharded weights)
- `/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth`
- `/mnt/d/Ai/checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth`

## Training output format

musubi-tuner writes `.safetensors` with `diffusion_model.{module}.lora_{up,down}.weight` + `.alpha` keys — drop-in for Faster-Wan2.2's `WanModel.apply_lora_safetensors()` loader.

## License

MIT. Inherits component licenses (musubi-tuner: Apache 2.0, Wan2.2: Apache 2.0, Gradio: Apache 2.0).
