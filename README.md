# BASI WAN KENOBI

*Created by [llmsherpa](https://x.com/LLMSherpa) ([@shitcoinsherpa](https://github.com/shitcoinsherpa)) of [BT6](https://bt6.gg).*

A Pinokio app for **Wan2.2 video** on consumer NVIDIA GPUs (from ~12GB VRAM). Two tabs:

- **Studio** — generate and edit video on the bundled Wan2.2 engine (`wan/`), GGUF Q4 + block-swap for low VRAM.
- **Gym** — FluxGym-style **LoRA training**: drop in clips → get a `.safetensors` LoRA (works in Studio, ComfyUI, kohya, peft).

Optionally adds **MOVA** joint **audio+video** training and generation.

## Studio

| Mode | What it does |
|---|---|
| **Text-to-video** | Wan2.2-A14B MoE (high/low expert), 4-step Lightning |
| **Generate with your LoRA** | apply a Gym-trained LoRA at a chosen strength (hot-swap, no restart) |
| **Continue** | extend the last clip from a final frame (I2V), with a Qwen3-VL "suggest next prompt" |
| **Restyle (V2V)** | re-style an uploaded video — fast SDEdit (8-step) or any-length sliding window |
| **Depth-lock (VACE)** | restyle while locking geometry to source depth; stack a style LoRA on top |
| **Keyframe edit (VACE)** | upload edited frames + positions → generate a clip anchored to them |
| **Talking character (S2V)** | drive a reference character with an audio track (lip-sync), chunked for long clips |
| **MOVA (audio+video)** | text → styled scene with synchronized audio; per-prompt style reference (SDXL + IP-Adapter) so the scene follows the prompt in a trained LoRA's style |

Modes and resolutions are gated by detected VRAM — small cards don't see options they can't run.

## Gym

LoRA training, two kinds:

- **Wan LoRA** (video) — musubi-tuner backend; native Wan2.2-A14B, block-swap for 24GB, sd-scripts safetensors output.
- **MOVA A/V LoRA** (joint audio+video) — trains the video expert + audio DiT + dual-tower bridge; NF4-resident base, 240p.

Both train on **Windows + Linux**.

Workflow: drop clips → auto-caption (Qwen3-VL) → train → sample per epoch → export.

## Install (Pinokio, recommended)

1. Add this repo in Pinokio ("Add Custom App" or the index).
2. **Install** — provisions the venv, GPU-matched PyTorch, deps, musubi-tuner, and fetches the inference weights (~40GB) and MOVA base (~78GB) via `tools/ensure_weights.py` (resumable — re-run to finish a partial download).
3. **Start** — Gradio opens at `http://localhost:7860`.

The Wan training base (~50GB) downloads on the first Gym run. MOVA installs into its own `env_mova` (conda, py3.12); both MOVA A/V training and generation run on Windows + Linux.

## Requirements

- **NVIDIA GPU**, ~12GB VRAM minimum. MOVA generation needs ~40GB+ system RAM (50-80GB recommended).
- **CUDA Toolkit** (`nvcc`) on `PATH` matching your PyTorch CUDA version.
- **MSVC C++ Build Tools** (Windows) for the from-source Q4_K/Q6_K inference kernel. Pinokio's bundled toolchain usually provides this; the install screen lists it as a prerequisite if not. The kernel compiles at install (`tools/build_kernel.py`) or on first generate; a missing toolchain is reported, never silently fatal. (Linux: `gcc/g++` + CUDA Toolkit; `ninja` ships in `requirements.txt`.)

## VRAM presets (Wan LoRA training)

| Preset | Card | rank | blocks_to_swap | max safe frames | optimizer |
|---|---|---|---|---|---|
| 8G | 4060 8GB | 8 | 40 | 17 | adafactor |
| 12G | 3060 12GB / 4070 | 16 | 30 | 17 | adafactor |
| 16G | 4060Ti 16GB / 4080 | 32 | 20 | 33 | adamw8bit |
| **24G** | **3090 / 4090** | **32** | **10** | **49** | adamw8bit |
| 40G+ | A100 / H100 | 64 | 0 | 81 | adamw8bit + dual-expert |

Higher frame counts may still fit on a card with slack; the UI warns above the safe cap but doesn't lock the slider.

## Wan2.2 MoE — before training

Wan2.2-T2V-A14B is a two-expert MoE along the diffusion timestep axis:

| Train | What | Use for |
|---|---|---|
| **low** | late-step expert (`--dit`) | texture / appearance / style — most character LoRAs |
| **high** | early-step expert (`--dit_high_noise`) | motion / pose / composition |
| **both** | dual-expert (`--timestep_boundary 0.875`) | max quality affecting motion AND appearance; needs 40GB+ |

A "low"-only LoRA won't change early-step behavior. If pose/motion doesn't transfer, retrain on "both" or "high".

## LoRA stacking (preview)

Live preview stacks the Lightning 4-step distill LoRA (strength 1.0) + your trained LoRA. Both at 1.0 can blow out colors; lower the user LoRA with `BASIWAN_USER_LORA_STRENGTH` (e.g. 0.7).

## Models

Weights auto-download via `tools/ensure_weights.py` to `./checkpoints/` (override with `BASIWAN_CKPT_DIR`). To reuse models from another Pinokio app (e.g. an existing SDXL from ComfyUI/Forge) or a shared drive, set `BASIWAN_SHARED_DIR` — the fetcher hardlinks an existing copy instead of re-downloading.

| Component | Repo | For |
|---|---|---|
| Wan2.2-T2V-A14B (base, VAE, T5) | `Wan-AI/Wan2.2-T2V-A14B` | training + inference |
| Wan2.2 GGUF Q4 pairs (T2V/I2V/VACE/S2V) | `QuantStack/Wan2.2-*-GGUF` | low-VRAM Studio |
| Lightning 4-step LoRAs | `lightx2v/Wan2.2-Lightning` | fast 4-step inference |
| wav2vec2-large-xlsr-53-english | `jonatasgrosman/...` | S2V audio encoder |
| Depth-Anything-V2-Small | `depth-anything/Depth-Anything-V2-Small-hf` | depth-lock restyle |
| MOVA-360p | `OpenMOSS-Team/MOVA-360p` | MOVA A/V (~78GB) |
| SDXL base + IP-Adapter | `stabilityai/...` + `h94/IP-Adapter` | MOVA T2AV style reference |
| Captioner VLMs | Qwen3-VL-8B/4B + Qwen2.5-VL-7B-AWQ (tier-aware) | Gym captioning, prompt suggest |

## Manual install (Linux/WSL)

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt bitsandbytes
git clone https://github.com/kohya-ss/musubi-tuner.git ext/musubi-tuner && pip install -e ext/musubi-tuner
python tools/ensure_weights.py --groups inference
python app.py     # → http://localhost:7860
```

## License & credits

Original code: **AGPL-3.0** (see [LICENSE](LICENSE)). Bundled and fetched third-party components keep their own (permissive) licenses — see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Built on the work of the Wan team, kohya-ss, OpenMOSS, city96, QuantStack, LightX2V, the Qwen team, Stability AI, h94, jonatasgrosman, Depth-Anything, IST-DASLab/vLLM (Marlin), madebyollin, kijai, deepbeepmeep, and cocktailpeanut — full acknowledgments in [CREDITS.md](CREDITS.md).

Generated content is yours; Wan claims no rights over it. Their acceptable-use policy (no illegal content, harassment, disinformation, or harm) applies to what you make.
