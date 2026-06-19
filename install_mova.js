// BASIWAN — OPTIONAL MOVA (joint audio+video) install. Separate from install.js because
// MOVA is a heavy opt-in: its own venv + ~77.7GB weights. Kept in its OWN venv ("env_mova")
// so MOVA's deps (descript-audiotools pins protobuf 3.19; its own diffusers/transformers
// versions) can NEVER perturb the carefully-tuned Wan/musubi inference+training env.
//
// Inference (text->A/V) runs cross-platform on plain torch + SDPA (no flash-attn, no
// torchcodec, no yunchang) — scripts/patch_mova_pipeline.py guards the 3 Linux-only yunchang
// imports so the package imports on Windows/macOS. TRAINING additionally needs torchcodec
// (Linux-only), installed via the [train] extra on Linux only.
//
// Pinned to OpenMOSS/MOVA @ 0fde19d (2026-05-07) — the SHA our LoRA format + sampler +
// yunchang patch target. Bump deliberately (re-verify the patch needles) when ready.
module.exports = {
  run: [
    // ── Create env_mova as a Python 3.12 conda env FIRST. The main BASIWAN env is 3.10
    // (Pinokio's bundled conda), but MOVA requires-python >=3.12 — pip would refuse to
    // install it into a 3.10 env. Idempotent (echo on re-run). Subsequent venv:"env_mova"
    // steps activate this existing 3.12 env.
    {
      method: "shell.run",
      params: {
        message: "conda create -y -p ./env_mova python=3.12 || echo [basiwan] env_mova already exists",
      },
    },

    // ── PyTorch into the dedicated env_mova venv (same branched logic as the main install)
    {
      method: "script.start",
      params: { uri: "torch.js", params: { venv: "env_mova" } },
    },

    // ── torch.js leaves torchaudio/torchvision UNPINNED, and pip grabs the newest torchaudio
    // (2.11.x) whose ABI mismatches torch 2.7.0 -> "WinError 127" on import. Pin torchaudio to
    // the torch-matched 2.7.0. (NVIDIA cu128; MOVA is NVIDIA-only like the rest of BASIWAN.)
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "env_mova",
        message: "uv pip install torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
      },
    },

    // ── bitsandbytes: required at IMPORT time (mova.engine.optimizers imports it) AND for our
    // NF4-base emulation at inference (the measured style-transfer win). It's in MOVA's [train]
    // extra only, so install it on ALL platforms here (official Windows wheels since 0.43).
    {
      method: "shell.run",
      params: {
        venv: "env_mova",
        message: "uv pip install bitsandbytes || echo [basiwan] bitsandbytes install failed - MOVA unavailable (needed for import + NF4)",
      },
    },

    // ── Clone OpenMOSS/MOVA (idempotent: echo on re-run instead of failing)
    {
      method: "shell.run",
      params: {
        message: "git clone https://github.com/OpenMOSS/MOVA.git ext/mova || echo [basiwan] MOVA already cloned",
      },
    },
    { method: "shell.run", params: { path: "ext/mova", message: "git config advice.detachedHead false" } },
    { method: "shell.run", params: { path: "ext/mova", message: "git fetch --quiet origin 0fde19d46d95301a26656b6ddac2387419bcb271 || git fetch --quiet" } },
    { method: "shell.run", params: { path: "ext/mova", message: "git checkout --quiet 0fde19d46d95301a26656b6ddac2387419bcb271" } },

    // ── Patch the 3 unconditional yunchang imports -> guarded (Windows/macOS importable).
    {
      method: "shell.run",
      params: { venv: "env_mova", message: "python scripts/patch_mova_pipeline.py" },
    },

    // ── MOVA INFERENCE deps (core, cross-platform: no torchcodec/yunchang/flash). Installed
    //    WITHOUT MOVA's bundled torch pin so the env_mova torch above (cu128) is preserved.
    {
      method: "shell.run",
      params: {
        venv: "env_mova",
        message: "uv pip install ./ext/mova",
      },
    },
    // huggingface_hub for the weight fetch below (descript-audiotools/diffusers pull most deps).
    {
      method: "shell.run",
      params: { venv: "env_mova", message: "uv pip install huggingface_hub" },
    },

    // ── TRAINING extra (torchcodec + bitsandbytes) — Linux only. torchcodec has no reliable
    //    Windows wheels; MOVA training is gated to Linux. Inference is unaffected on all OSes.
    {
      when: "{{platform === 'linux'}}",
      method: "shell.run",
      params: {
        venv: "env_mova",
        message: "uv pip install ./ext/mova[train] || echo [basiwan] MOVA [train] extra failed (torchcodec) - MOVA training unavailable; inference still works",
      },
    },

    // ── MOVA-360p weights (~77.7GB) into ./checkpoints (resumable; one consolidated acquirer).
    {
      method: "shell.run",
      params: {
        venv: "env_mova",
        message: "python tools/ensure_weights.py --groups mova || echo [basiwan] MOVA weights incomplete - resumes on re-run/first use",
      },
    },

    {
      method: "log",
      params: {
        text: "MOVA (audio+video) install done. ~77.7GB weights fetched to ./checkpoints/MOVA-360p (resumable — re-run to finish any partial). Inference (text->A/V) runs on all OSes; training is Linux-only (torchcodec). Needs ~50-80GB system RAM for generation.",
      },
    },
  ],
};
