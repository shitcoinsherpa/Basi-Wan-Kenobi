# BASI WAN K3N0B1

A Pinokio app for **Wan2.2 video** on consumer GPUs — two tabs:

- **Studio** — generate + edit video on the bundled Faster-Wan2.2 engine (`wan/`),
  optimized for 24GB and down to ~12GB via GGUF Q4 + block-swap.
- **Gym** — Flux-Gym-style **LoRA training**: drop in clips → get a `.safetensors`
  LoRA compatible with the Studio engine, ComfyUI, kohya, peft.

### Studio features

| Feature | What it does |
|---|---|
| **Text-to-video** | Wan2.2-A14B MoE (high/low expert), 4-step Lightning, ~3-4 s/clip steady-state at 832×480×17 |
| **Generate with your LoRA** | apply a Gym-trained LoRA at a chosen strength (hot-swappable, no restart) |
| **Continue** | extend the last clip from one of its final frames (I2V), with a Qwen3-VL "suggest next prompt" |
| **Restyle (V2V)** | re-style any uploaded video — fast SDEdit (8-step) or any-length sliding-window |
| **Depth-lock (VACE)** | restyle while locking geometry to the source depth (corr 0.973), stack a style LoRA on top |
| **Keyframe edit (VACE)** | upload edited frames + positions → generate a clip anchored to them (anchors preserved) |

## Status

**v0.1** — full Flux-Gym-style stack: 3-column UI, auto-caption button, generated
cache+train scripts, in-app spawn, ComfyUI symlink. End-to-end smoke test
(musubi-tuner subprocess against a real dataset) scripted in
`scripts/smoke_test_pipeline.py`; runs against locally-placed models.

**Pinokio integration:** `pinokio.js` + `install.js` + `start.js` + `update.js`
ready in repo root. One-click install picks PyTorch index by detected GPU
(cu121 / cu128 nightly / rocm6.1).

| Component | Status |
|---|---|
| Spec + architecture decision | ✓ done |
| musubi-tuner backend integration | ✓ installed + E2E smoke scripted (`scripts/smoke_test_pipeline.py`) |
| Dataset validator (4n+1 frame snap, resolution bucketing) | ✓ done |
| 24G VRAM preset (FP8 base, rank 32, blocks_to_swap 10) | ✓ done |
| Gradio UI scaffold (Flux-Gym pattern) | ✓ done |
| Qwen2.5-VL auto-captioning (tier-aware 3B/7B-AWQ/7B; Qwen3-VL opt-in needs transformers>=4.57) | ✓ done |
| Live preview (Faster-Wan2.2 + Lightning, trainer-paused) | ⚠ backend reroute pending |
| Output format converters (comfyui/sd-scripts/peft) | ✓ done (`basi/export.py`) |
| VSA training compat (verdict: not feasible without 1100-LOC musubi fork) | ✓ closed |
| TAEHV tiny-VAE for preview decode (Faster-Wan2.2 P26) | ✓ done |
| CUDA Graphs at Lightning shape (Faster-Wan2.2 P25) | ✓ shipped, bench pending |

## Stack

- **Backend**: [musubi-tuner](https://github.com/kohya-ss/musubi-tuner) — actively maintained, native Wan2.2-A14B support, `--blocks_to_swap` for 24GB VRAM, sd-scripts safetensors output format
- **Frontend**: Gradio (3-column Flux-Gym layout); plain `gr.Textbox(autoscroll=True)` for live logs (last 200 lines), Stop button signals SIGTERM/KILL the script's process group
- **Inference / live preview**: bundled Faster-Wan2.2 engine (`wan/`) — Lightning+FP8+TAEHV path (~3-4 s at 832×480×17 steady-state)
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

### Build prerequisites (NVIDIA inference)

Studio inference runs on a from-source CUDA kernel (the Q4_K/Q6_K fast path).
It compiles automatically — at install (`tools/build_kernel.py`) or on first
generate — but the **build toolchain must be present**:

- **ninja** — installed automatically (in `requirements.txt`).
- **MSVC C++** (Windows) — "Visual Studio Build Tools" + the *Desktop development
  with C++* workload. Launch from an *x64 Native Tools Command Prompt*, or ensure
  `cl.exe` is on `PATH`.
- **CUDA Toolkit** (`nvcc`) — matching your PyTorch CUDA version; its `bin/` on `PATH`.

`tools/build_kernel.py` reports exactly which tools are missing (it never fails the
install). There is currently **no pure-PyTorch fallback** — the kernel is required
for inference, so install the toolchain before first generate. (Linux: install
`ninja`, `gcc/g++`, and the CUDA Toolkit.)

### Required checkpoints (auto-checked by smoke_test.py)
- `Wan2.2-T2V-A14B` base — set `BASI_CKPT_DIR` or place under
  `./checkpoints/Wan2.2-T2V-A14B/`
- `Wan2.2-Lightning` 4-step LoRA — `BASI_LIGHTNING_LORA_DIR` or
  `./checkpoints/Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928/`
- `taew2_1.pth` (TAEHV) — auto-downloaded by smoke_test.py on first run.

All checkpoints resolve under **`BASIWAN_CKPT_DIR`** (default `./checkpoints/`,
next to the app). Set that one env var to point everything at an existing model
drive; no per-file paths needed.

### Model download

Place models under `./checkpoints/` (or your `BASIWAN_CKPT_DIR`). Sizes are
approximate. Fetch the base + Lightning LoRA before first run; the GGUF/VACE/depth
sets are only needed for the features that use them.

| Component | Hugging Face repo | Needed for | Fetch |
|---|---|---|---|
| Wan2.2-T2V-A14B (base, VAE, T5) | `Wan-AI/Wan2.2-T2V-A14B` (~50 GB) | training + all inference | manual |
| Wan2.2-I2V-A14B (base) | `Wan-AI/Wan2.2-I2V-A14B` (~50 GB) | Continue / image-to-video | manual (optional) |
| Lightning 4-step LoRA | `lightx2v/Wan2.2-Lightning` | fast 4-step inference | manual |
| T2V GGUF Q4 pair | `QuantStack/Wan2.2-T2V-A14B-GGUF` | low-VRAM T2V (Studio) | manual (optional) |
| I2V GGUF Q4 pair | `QuantStack/Wan2.2-I2V-A14B-GGUF` | low-VRAM Continue | manual (optional) |
| VACE-Fun GGUF Q4 pair | `QuantStack/Wan2.2-VACE-Fun-A14B-GGUF` | depth-lock restyle + keyframe edit | `python tools/_fetch_vace_gguf.py` |
| Depth-Anything-V2-Small | `depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0) | depth-lock restyle | `python tools/_fetch_vace_gguf.py` |
| TAEHV `taew2_1.pth` | (madebyollin/taehv) | fast VAE decode | auto (smoke_test) |
| Captioner / "Suggest prompt" VLMs | Qwen3-VL-4B + tier captioner | gym captioning, prompt suggest | auto (`tools/prefetch_vlms.py` at install) |

> **Licensing**: ship only `Depth-Anything-V2-Small` (Apache-2.0) — the Large
> variant is CC-BY-NC. Of the Qwen2.5-VL captioners only the 7B is Apache; 3B/72B
> are not. See `THIRD_PARTY_LICENSES.md`. `tools/_fetch_vace_gguf.py` and the dev
> bench/probe tools honor `BASIWAN_CKPT_DIR`.

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
│   └── export.py          # LoRA format converters (comfyui/sd-scripts/peft)
├── ext/
│   └── musubi-tuner/      # git clone of kohya-ss/musubi-tuner
└── outputs/               # per-LoRA artifacts (dataset/, sample/, preview/, *.safetensors)
```

## Wan2.2 model weights expected

The UI defaults to `./checkpoints/` (override with `BASIWAN_CKPT_DIR`):
- `./checkpoints/Wan2.2-T2V-A14B/{high,low}_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors` (musubi auto-detects sharded weights)
- `./checkpoints/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth`
- `./checkpoints/Wan2.2-T2V-A14B/Wan2.1_VAE.pth`

## Training output format

musubi-tuner writes `.safetensors` with `diffusion_model.{module}.lora_{up,down}.weight` + `.alpha` keys — drop-in for Faster-Wan2.2's `WanModel.apply_lora_safetensors()` loader.

## License & Credits

Original code: MIT (see [LICENSE](LICENSE)). Portions of this repository
are derived from Apache-2.0 projects (Wan 2.2, city96/ComfyUI-GGUF,
diffusers/transformers) and remain under their licenses -- see
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

This app stands on the work of the Wan team, kohya-ss, city96, QuantStack,
LightX2V, the Qwen team, IST-DASLab/vLLM (Marlin), madebyollin, kijai,
deepbeepmeep, and cocktailpeanut -- full acknowledgments and the citations
their licenses or model cards request are in [CREDITS.md](CREDITS.md).

Generated content is yours; Wan claims no rights over it. Their stated
acceptable-use policy (no illegal content, harassment, disinformation, or
harm) applies to what you make with this tool.
