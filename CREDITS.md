# Credits & Acknowledgments

BASI WAN K3N0B1 is built on the work of many people. Progress is putting
blocks upon foundations laid by others — these are the signatures on the
cornerstones. Thank you, all of you.

## Models

- **Wan 2.2** — the Wan Team at Alibaba ([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2), Apache-2.0).
  The T2V-A14B and I2V-A14B models, the Wan 2.1 VAE, and the umT5-XXL text
  encoder packaging are the heart of this app. Wan explicitly claims no
  rights over content you generate. Citation requested by the authors:

  ```bibtex
  @article{wan2025,
        title={Wan: Open and Advanced Large-Scale Video Generative Models},
        author={Team Wan and Ang Wang and Baole Ai and Bin Wen and Chaojie Mao
                and Chen-Wei Xie and Di Chen and Feiwu Yu and Haiming Zhao and
                Jianxiao Yang and Jianyuan Zeng and Jiayu Wang and Jingfeng
                Zhang and Jingren Zhou and Jinkai Wang and Jixuan Chen and
                Kai Zhu and Kang Zhao and Keyu Yan and Lianghua Huang and
                Mengyang Feng and Ningyi Zhang and Pandeng Li and Pingyu Wu and
                Ruihang Chu and Ruili Feng and Shiwei Zhang and Siyang Sun and
                Tao Fang and Tianxing Wang and Tianyi Gui and Tingyu Weng and
                Tong Shen and Wei Lin and Wei Wang and Wei Wang and Wenmeng
                Zhou and Wente Wang and Wenting Shen and Wenyuan Yu and
                Xianzhong Shi and Xiaoming Huang and Xin Xu and Yan Kou and
                Yangyu Lv and Yifei Li and Yijing Liu and Yiming Wang and
                Yingya Zhang and Yitong Huang and Yong Li and You Wu and
                Yu Liu and Yulin Pan and Yun Zheng and Yuntao Hong and
                Yupeng Shi and Yutong Wang and Yuxuan Yan and Zhaowei Bin and
                Zhen Han and Zhi-Fan Wu and Ziyu Liu},
        journal = {arXiv preprint arXiv:2503.20314},
        year={2025}
  }
  ```

- **umT5-XXL** — Google ([google/umt5-xxl](https://huggingface.co/google/umt5-xxl), Apache-2.0);
  UniMax, Chung et al., ICLR 2023.
- **GGUF quantizations** — [QuantStack](https://huggingface.co/QuantStack)
  (Wan2.2-T2V-A14B-GGUF and Wan2.2-I2V-A14B-GGUF, Apache-2.0, inheriting
  Wan's terms), building on conversion tooling by
  [city96](https://github.com/city96/ComfyUI-GGUF).
- **Wan2.2-Lightning 4-step distill LoRAs** (T2V 250928 + I2V Seko-V1) —
  the LightX2V team at ModelTC
  ([lightx2v/Wan2.2-Lightning](https://huggingface.co/lightx2v/Wan2.2-Lightning),
  Apache-2.0). These make 4-step generation possible; descended from the
  Self-Forcing line of research.
- **Qwen3-VL-8B / Qwen3-VL-4B / Qwen2.5-VL-7B-AWQ** — the Qwen team at
  Alibaba (Apache-2.0). Power auto-captioning and prompt suggestions.
  Citations requested: Qwen3 Technical Report (arXiv:2505.09388),
  Qwen2.5-VL (arXiv:2502.13923), Qwen2-VL (arXiv:2409.12191).
- **TAEHV** (tiny video VAE, taew2_1) — Ollin Boer Bohan
  ([madebyollin/taehv](https://github.com/madebyollin/taehv), MIT).

## Code

- **musubi-tuner** — kohya-ss
  ([kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner),
  Apache-2.0). The entire LoRA training backend of the Gym.
- **Wan 2.2 reference implementation** — the Wan Team (Apache-2.0). Our
  `wan/` inference tree is a heavily modified derivative; original
  copyright headers are retained and modifications are noted per file.
- **ComfyUI-GGUF** — city96 (Apache-2.0). Our GGUF loading/dequantization
  path in `tools/gguf_vendor/` derives from it; headers retained.
- **Marlin** — IST-DASLab ([IST-DASLab/marlin](https://github.com/IST-DASLab/marlin),
  Apache-2.0) and the [vLLM](https://github.com/vllm-project/vllm) port
  (Apache-2.0). Our CUDA kernels are an independent reimplementation
  informed by the Marlin design; the quantization permutation tables and
  the LOP3 fp16 dequant idiom follow Marlin. Citation requested by the
  authors: Frantar et al., *MARLIN: Mixed-Precision Auto-Regressive
  Parallel Inference on Large Language Models*, arXiv:2408.11743.
- **qwen-vl-utils** — kq-chen (vendored helper, header retained).
- **diffusers / transformers** — Hugging Face (Apache-2.0); our flow-match
  solvers and T5 module are modified copies, headers retained.
- **ComfyUI-WanVideoWrapper** — kijai (Apache-2.0). No code copied, but
  many techniques in the consumer-GPU Wan ecosystem were proven there
  first; this app stands on that prior art.
- **Wan2GP** — deepbeepmeep (prior art for accessible Wan video on small
  GPUs, distributed on Pinokio).
- **FluxGym** — cocktailpeanut. The Gym's UX is modeled on it, and the
  Pinokio scripts follow his conventions. Pinokio itself —
  [pinokiocomputer/pinokio](https://github.com/pinokiocomputer/pinokio) (MIT).
- **FFmpeg** — the FFmpeg team. Video encoding/decoding throughout, used
  as an external tool via imageio-ffmpeg / PyAV (see THIRD_PARTY_LICENSES.md
  for the GPL notes on the bundled binaries).
- **gradio** (Apache-2.0), **PyTorch** (BSD-3), **bitsandbytes** (MIT),
  **huggingface_hub / safetensors / accelerate / peft** (Apache-2.0).

## A note on generated content

Wan 2.2 and Wan2.2-Lightning are Apache-2.0 with a stated acceptable-use
policy: no illegal content, harassment, disinformation, or harm to others.
You own what you make; you're also responsible for it.
