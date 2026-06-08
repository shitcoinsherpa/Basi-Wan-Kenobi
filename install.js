// BASIWAN — Pinokio install script. Canonical cocktailpeanut shape:
// requires.bundle "ai" pulls Pinokio's bundled conda; uv pip throughout;
// venv "env" (canonical path); torch install delegated to torch.js.
module.exports = {
  requires: { bundle: "ai" },
  run: [
    // ── PyTorch via separated script (handles all platforms + GPU branches)
    {
      method: "script.start",
      params: { uri: "torch.js", params: { venv: "env" } },
    },

    // ── Common Python deps
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: ["uv pip install -r requirements.txt"],
      },
    },

    // ── TAEHV explicit install (git URL; some corporate networks block git+ in pip -r)
    // The requirements.txt line covers most installs, but `uv pip install -r`
    // swallows the failure silently when git URLs are blocked. This step fails
    // loudly with a clear error.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install git+https://github.com/madebyollin/taehv.git || echo '[basiwan] taehv install failed — your network may block git+ URLs. Preview will not work until taehv is installed manually.'",
        ],
      },
    },

    // ── bitsandbytes: skip on Windows native (no official wheel)
    {
      when: "{{platform !== 'win32'}}",
      method: "shell.run",
      params: {
        venv: "env",
        message: ["uv pip install bitsandbytes"],
      },
    },
    {
      when: "{{platform === 'win32'}}",
      method: "log",
      params: {
        text: "Skipping bitsandbytes (no Windows-native wheel). adamw8bit will be unavailable; use WSL2 if you need 8-bit optimizers.",
      },
    },

    // ── musubi-tuner clone + editable install
    // Pin musubi-tuner to v0.3.0. Upstream HEAD is actively moving; a future
    // rename of --blocks_to_swap or --task choices would silently break the
    // gym. Bump deliberately via update.js when ready.
    //
    // Upstream pins transformers==4.56.1; we loosen to >=4.57,<5 so the
    // Qwen3-VL captioner works on the same env. --no-deps prevents pip from
    // resolving the editable install back into the hard pin and downgrading.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "git clone https://github.com/kohya-ss/musubi-tuner.git ext/musubi-tuner || true",
          "(cd ext/musubi-tuner && git fetch --tags && git checkout v0.3.0)",
          "(cd ext/musubi-tuner && git rev-parse HEAD > ../musubi-tuner.sha)",
          "python -c \"from pathlib import Path; p=Path('ext/musubi-tuner/pyproject.toml'); t=p.read_text(); t=t.replace('\\\"transformers==4.56.1\\\"', '\\\"transformers>=4.57.0,<5\\\"'); p.write_text(t)\"",
          "uv pip install -e ext/musubi-tuner --no-deps",
          "uv pip install accelerate diffusers safetensors einops opencv-python av sentencepiece protobuf prompt_toolkit voluptuous toml ftfy easydict",
        ],
      },
    },

    // ── Platform-specific final notes
    {
      when: "{{platform === 'darwin'}}",
      method: "log",
      params: {
        text: "macOS detected. Wan2.2-A14B requires ~50GB; usable only on M-series with >=32GB unified memory, and even then expect very slow inference (MPS path is not optimized for video diffusion). Linux+NVIDIA is the supported target.",
      },
    },
    {
      method: "log",
      params: {
        text: "Install done. Wan2.2 base weights (~50GB) are NOT auto-downloaded. Place them under ./checkpoints/Wan2.2-T2V-A14B/ OR set BASIWAN_CKPT_DIR before Start. First training run will load from disk; expect ~3 min on first cold start.",
      },
    },
  ],
};
