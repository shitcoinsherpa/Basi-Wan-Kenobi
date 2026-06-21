// BASIWAN — Pinokio install script. Canonical cocktailpeanut shape:
// requires.bundle "ai" pulls Pinokio's bundled conda; uv pip throughout;
// venv "env" (canonical path); torch install delegated to torch.js.
//
// musubi-tuner setup is split into atomic single-command shell.run blocks
// for Windows PTY compatibility: a multi-command array hangs Pinokio's
// "done" detector (git's detached-HEAD advice + subshell exit codes confuse
// the PTY watcher). Each step is idempotent and re-runnable.
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

    // ── TAEHV explicit install (git URL; some networks block git+ in pip -r)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "uv pip install git+https://github.com/madebyollin/taehv.git || echo [basiwan] taehv install failed - preview disabled until installed manually",
        ],
      },
    },

    // ── bitsandbytes: official Windows wheels exist since 0.43 (CUDA-
    // functional on Windows native). The 24g preset's adamw8bit optimizer
    // needs it.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: ["uv pip install bitsandbytes || echo [basiwan] bitsandbytes install failed - adamw8bit optimizer unavailable, presets fall back to adafactor"],
      },
    },

    // ── musubi-tuner — atomic steps (each idempotent / re-runnable)
    //
    // Pin to v0.3.0. Upstream HEAD is actively moving; a future rename of
    // --blocks_to_swap or --task choices would silently break the gym.
    // Bump deliberately via update.js when ready.

    // 1) Clone (idempotent: echoes a note on re-run instead of failing)
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/kohya-ss/musubi-tuner.git ext/musubi-tuner || echo [basiwan] musubi-tuner already cloned",
      },
    },

    // 2) Suppress git's detached-HEAD advice block — its long output was
    //    confusing Pinokio's PTY "done" detector on Windows.
    {
      method: "shell.run",
      params: {
        path: "ext/musubi-tuner",
        message: "git config advice.detachedHead false",
      },
    },

    // 3) Fetch tags quietly
    {
      method: "shell.run",
      params: {
        path: "ext/musubi-tuner",
        message: "git fetch --tags --quiet",
      },
    },

    // 4) Checkout the pin
    {
      method: "shell.run",
      params: {
        path: "ext/musubi-tuner",
        message: "git checkout --quiet v0.3.0",
      },
    },

    // 5) Record SHA for traceability
    {
      method: "shell.run",
      params: {
        path: "ext/musubi-tuner",
        message: "git rev-parse HEAD > ../musubi-tuner.sha",
      },
    },

    // 6) Patch pyproject.toml: loosen transformers pin for Qwen3-VL.
    //    Uses a dedicated script (scripts/patch_musubi_pyproject.py) for
    //    Windows cmd.exe shell-escape compatibility — the inline
    //    `python -c` form mangles quoting on cmd.exe.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "python scripts/patch_musubi_pyproject.py",
      },
    },

    // 7) Editable install of musubi-tuner without resolving deps (the
    //    pyproject pin would otherwise downgrade transformers)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "uv pip install -e ext/musubi-tuner --no-deps",
      },
    },

    // 8) Musubi's runtime deps (transformers handled in requirements.txt)
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: "uv pip install accelerate diffusers safetensors einops opencv-python av sentencepiece protobuf prompt_toolkit voluptuous toml ftfy easydict",
      },
    },

    // ── VLM models at INSTALL time, with visible progress. Prefetching the
    // Suggest-prompt model (Qwen3-VL-4B) and the VRAM-tier captioner (8B at
    // 24G) here avoids a silent 8-17 GB download on first button click (which
    // looks like a dead app behind a frozen progress bar). huggingface_hub
    // prints live byte progress in the install terminal. Failure is non-fatal:
    // the app's launch-time background prefetch is the safety net.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python tools/prefetch_vlms.py || echo [basiwan] VLM prefetch failed - models will download in background on first launch",
        ],
      },
    },

    // ── Model weights: ensure the INFERENCE set is present so Studio works
    // out of the box — GGUF Q4 DiT pairs (T2V/I2V/VACE/S2V) + the base
    // T5/VAE/tokenizer subset (NOT the ~28GB bf16 DiT shards; GGUF replaces
    // them) + Lightning LoRAs + S2V audio encoder + depth model. One
    // consolidated, idempotent, resumable acquirer (tools/ensure_weights.py)
    // with live byte progress. Non-fatal: partials resume on re-run / first
    // use. The ~50GB Wan TRAINING base is fetched on the first Gym run.
    {
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python tools/ensure_weights.py --groups inference || echo [basiwan] weight download incomplete - resumes on re-run/first use",
        ],
      },
    },

    // ── Pre-build the BASIWAN CUDA kernels (NVIDIA only) so the first
    // generate doesn't stall on a silent ~50s nvcc compile, and a missing
    // toolchain surfaces HERE not mid-generate. Tolerant: build_kernel.py always
    // exits 0 — a skipped pre-build falls back to the runtime lazy build.
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env",
        message: [
          "python tools/build_kernel.py",
        ],
      },
    },

    // ── MOVA (joint audio+video) — full setup as part of the basic install: a dedicated
    // env_mova venv (isolated so MOVA's deps can't perturb the Wan/musubi env), the patched
    // OpenMOSS clone, and the ~77.7GB MOVA-360p weights. Self-contained in install_mova.js,
    // invoked here exactly like torch.js so MOVA A/V is ready out of the box. Every step inside
    // is idempotent/resumable, so re-running Install finishes any partial MOVA download.
    {
      method: "script.start",
      params: { uri: "install_mova.js" },
    },

    // ── Platform-specific final notes
    {
      when: "{{platform === 'darwin'}}",
      method: "log",
      params: {
        text: "macOS is not supported/tested: this app needs an NVIDIA GPU + a from-source CUDA kernel. Apple MPS isn't optimized for video diffusion (expect very slow or non-working inference even on M-series with >=32GB). Use Windows or Linux with an NVIDIA GPU.",
      },
    },
    {
      method: "log",
      params: {
        text: "Install done. Inference weights (GGUF DiT pairs + VAE/T5/tokenizer + Lightning + S2V + depth) plus the MOVA-360p joint A/V base (~77.7GB) were fetched to ./checkpoints (resumable — just re-run Install to finish any partial download). MOVA runs in its own env_mova venv; A/V generation needs ~50-80GB system RAM. The ~50GB Wan training base downloads automatically on your first Wan Gym run. Set BASIWAN_CKPT_DIR before Start to use a shared model drive.",
      },
    },
  ],
};
