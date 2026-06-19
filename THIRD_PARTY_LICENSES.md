# Third-Party Licenses

BASI WAN K3N0B1's original code is licensed under **AGPL-3.0** (see LICENSE /
[LICENSES/AGPL-3.0.txt](LICENSES/AGPL-3.0.txt)). This file documents the
third-party components in this repository and those fetched at install time,
and satisfies the notice requirements of their licenses. The full Apache-2.0
text is at [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

All third-party components below are permissive (Apache-2.0 / MIT / BSD / ISC)
and are one-way compatible into an AGPL-3.0 combined work (the FSF lists each as
GPLv3/AGPLv3-compatible). Each component retains its own license and notices;
the combined distribution is governed by AGPL-3.0.

## Code vendored or derived in this repository

| Component | Origin | License | Where |
|---|---|---|---|
| Wan 2.2 reference inference code | [Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) © Alibaba Wan Team | Apache-2.0 | `wan/` — heavily modified derivative. Original copyright headers retained; per-file modification notices added (Apache §4(b)). |
| GGUF loader / dequant / ops | [city96/ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) © City96 | Apache-2.0 | `tools/gguf_vendor/{loader,dequant,ops}.py` — headers retained, modifications noted. |
| Marlin permutation tables + LOP3 dequant idiom | [IST-DASLab/marlin](https://github.com/IST-DASLab/marlin) / [vLLM](https://github.com/vllm-project/vllm) | Apache-2.0 | `tools/gguf_vendor/basiwan_q4_kernel/`, `tools/basiwan_v2_kernel/` — independent kernel reimplementation informed by the Marlin design; tables reproduced from upstream are credited in-file. |
| qwen-vl-utils helper | [kq-chen/qwen-vl-utils](https://github.com/kq-chen/qwen-vl-utils) | Apache-2.0 | `wan/utils/qwen_vl_utils.py` (header retained) |
| Flow-match solvers | [huggingface/diffusers](https://github.com/huggingface/diffusers) | Apache-2.0 | `wan/utils/fm_solvers*.py` (headers retained) |
| T5 module | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache-2.0 | `wan/modules/t5.py` (header retained) |

## Components fetched at install time (not redistributed by this repo)

| Component | License | Notes |
|---|---|---|
| [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner) (git clone → `ext/`) | Apache-2.0 (README-declared; upstream ships no LICENSE file). Its `hunyuan_model*` subdirectories follow Tencent HunyuanVideo licenses — BASIWAN does not use them. | Wan LoRA training backend |
| [OpenMOSS/MOVA](https://github.com/OpenMOSS/MOVA) (git clone → `ext/mova`, pinned @ `0fde19d`) | Apache-2.0 | MOVA joint A/V model code. **Modified at install** by `scripts/patch_mova_pipeline.py` (try/except guards around 3 Linux-only `yunchang` imports so the package imports on Windows/macOS) — each changed file carries an in-file "[basiwan]" notice per Apache-2.0 §4(b). Its LICENSE travels with the clone. |
| [OpenMOSS-Team/MOVA-360p](https://huggingface.co/OpenMOSS-Team/MOVA-360p) (HF download, ~77.7GB) | Apache-2.0 (model-card `license: apache-2.0`; no non-commercial/RAIL clause) | MOVA joint A/V weights. Downloaded to the user's machine; **not redistributed/vendored** by this repo. |
| [feifeibear/long-context-attention](https://github.com/feifeibear/long-context-attention) (`yunchang`, pip into `env_mova`) | Apache-2.0 | Multi-GPU context-parallel attention; pure-python, unused on single-GPU inference (imports guarded). |
| [descriptinc/audiotools](https://github.com/descriptinc/audiotools) + [descript-audio-codec](https://github.com/descriptinc/descript-audio-codec) (`descript-audiotools`, pip) | MIT — © Descript | MOVA's DAC audio codec |
| [open-mmlab/mmengine](https://github.com/open-mmlab/mmengine) (pip) | Apache-2.0 | MOVA config/registry runtime |
| [librosa](https://github.com/librosa/librosa) / [soundfile](https://github.com/bastibe/python-soundfile) / [numba](https://github.com/numba/numba) (pip, MOVA audio deps) | ISC / BSD-3-Clause / BSD-2-Clause | transitive audio stack |
| Wan-AI / QuantStack / lightx2v / Qwen / google model weights | Apache-2.0 (each; see CREDITS.md) | Downloaded from Hugging Face by the user's machine |
| [madebyollin/taehv](https://github.com/madebyollin/taehv) (pip from git) | MIT — Copyright (c) Ollin Boer Bohan | LICENSE travels with the package |
| gradio, huggingface_hub, transformers, diffusers, accelerate, peft, safetensors | Apache-2.0 | pip |
| PyTorch | BSD-style | pip/installer |
| bitsandbytes | MIT | pip |
| imageio-ffmpeg | BSD-2-Clause (wrapper) | **See FFmpeg note below** |
| av (PyAV) | BSD-3-Clause (bundles FFmpeg libraries, LGPL) | pip (musubi dependency) |

## FFmpeg note — LOAD-BEARING, read before changing packaging

`imageio-ffmpeg` downloads a **GPLv3-licensed ffmpeg binary** (gyan.dev /
johnvansickle builds with x264/x265) onto the user's machine at pip-install
time. BASIWAN invokes it strictly as a separate executable via subprocess —
the app is not a derivative work, and because the binary is fetched by pip
on the user's machine, this project does not distribute it.

**Do not ever bundle the imageio-ffmpeg wheel, the ffmpeg binary, or a
populated `env/` into a release artifact** (zip, installer, docker image).
Doing so would make this project a GPLv3 distributor, with source-offer
obligations. If offline packaging is ever needed, use an LGPL-only ffmpeg
build instead.

## Future components (noted for when they land)

- Wan2.2-VACE-Fun / Fun-InP (alibaba-pai): Apache-2.0.
- Depth-Anything-V2: code Apache-2.0; **Small model Apache-2.0, but
  Base/Large/Giant models are CC-BY-NC-4.0 (non-commercial)** — ship Small.
- Qwen2.5-VL family is license-heterogeneous (7B = Apache-2.0; 3B and 72B
  are NOT) — re-check before changing captioner tiers.
