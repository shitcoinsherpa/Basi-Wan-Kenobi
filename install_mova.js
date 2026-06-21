// BASIWAN — OPTIONAL MOVA (joint audio+video) install. Separate from install.js because
// MOVA is a heavy opt-in: its own env ("env_mova") + ~77.7GB weights. Kept in its OWN env so
// MOVA's deps (descript-audiotools pins protobuf 3.19; its own diffusers/transformers versions)
// can NEVER perturb the carefully-tuned Wan/musubi inference+training env.
//
// env_mova is a CONDA env (created with `conda create -p ./env_mova python=3.12`, because MOVA
// requires-python >=3.12 while the main BASIWAN env is Python 3.10). A conda env has NO
// Scripts\activate, so Pinokio's `venv:` activation (which runs <env>\Scripts\activate) does NOT
// work for it. Instead, drive env_mova's own python directly via its full path
// (env_mova\python.exe on Windows, env_mova/bin/python on Linux) with `-m pip` — no activation,
// no uv/PATH dependency. MOVA_PY below is resolved on the user's host at require() time.
//
// Inference (A/V) runs on NVIDIA torch 2.7/cu128 + SDPA (no flash-attn, no torchcodec, no
// yunchang) — scripts/patch_mova_pipeline.py guards the 3 Linux-only yunchang imports so the
// package imports on Windows. MOVA is I2V: the STYLE comes from the reference image, not the
// torch version or compile path, so env_mova matches the main app's torch (no higher driver bar);
// it runs eager when Triton is absent. A/V TRAINING (the MOVA Gym) and generation both run on
// Windows + Linux on this env (torch 2.7/cu128); the [train] extra (torchcodec + bitsandbytes) is
// installed on all platforms.
//
// Pinned to OpenMOSS/MOVA @ 0fde19d — the SHA our LoRA format + sampler + yunchang patch
// target. Bump deliberately (re-verify the patch needles) when ready.
const _WIN = process.platform === "win32";
const MOVA_PY = _WIN ? "env_mova\\python.exe" : "env_mova/bin/python";
// Pinokio's base env sets pip's `require-virtualenv`, and pip does NOT count a CONDA env as a
// virtualenv (conda doesn't set sys.base_prefix like a venv) -> `env_mova\python.exe -m pip install`
// dies with "Could not find an activated virtualenv (required)". Driving python directly does not
// satisfy the check, so disable it inline for pip steps (cross-platform). MOVA_PIP = that env var
// + the env_mova python's pip.
const MOVA_PIP = (_WIN ? 'set "PIP_REQUIRE_VIRTUALENV=0" && ' : "PIP_REQUIRE_VIRTUALENV=0 ") + MOVA_PY + " -m pip";
module.exports = {
  run: [
    // ── Create env_mova as a Python 3.12 conda env FIRST (idempotent; echo on re-run). Every
    // later step calls MOVA_PY directly instead of activating it (conda envs have no Scripts\activate).
    {
      method: "shell.run",
      params: {
        message: "conda create -y -p ./env_mova python=3.12 || echo [basiwan] env_mova already exists",
      },
    },

    // ── PyTorch (+vision+audio) into env_mova, matched cu128 set, via env_mova's own pip. Matches
    // the MAIN app's torch 2.7/cu128 so MOVA never raises the driver bar (CUDA 13/cu130 needs much
    // newer drivers). MOVA's STYLE is the I2V reference image, not torch/compile, so no newer torch
    // is needed. NVIDIA path; CPU fallback keeps the env importable.
    {
      when: "{{gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        message: MOVA_PIP + " install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128",
      },
    },
    {
      when: "{{gpu !== 'nvidia'}}",
      method: "shell.run",
      params: {
        message: MOVA_PIP + " install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0",
      },
    },

    // ── Triton for the torch.compile FAST PATH. Compile is a SPEED + prompt-adherence win, NOT a
    // style lever (~33% faster per gen, 742s vs 1107s eager). On Linux the
    // torch 2.7 wheel already pulls triton 3.3.x (free). On Windows there's no PyPI triton, so
    // install the torch-2.7-matched triton-windows 3.3.x wheel. If it fails, mova_sample.py
    // auto-falls back to EAGER (same style, slower) -- optimization, never a hard requirement.
    {
      when: "{{platform === 'win32' && gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        message: MOVA_PIP + " install \"triton-windows==3.3.*\" || echo [basiwan] triton-windows install failed - MOVA will run eager (slower, same quality)",
      },
    },

    // ── bitsandbytes: required at IMPORT time (mova.engine.optimizers imports it) AND for our
    // NF4-base emulation at inference (the measured style-transfer win). Official Windows wheels
    // since 0.43.
    {
      method: "shell.run",
      params: {
        message: MOVA_PIP + " install -c tools/mova_torch_constraints.txt bitsandbytes || echo [basiwan] bitsandbytes install failed - MOVA unavailable (needed for import + NF4)",
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
      params: { message: MOVA_PY + " scripts/patch_mova_pipeline.py" },
    },

    // ── MOVA INFERENCE deps (core, cross-platform: no torchcodec/yunchang/flash). The
    //    -c constraints pins torch==2.7.0 so MOVA's deps (descript-audiotools, diffusers,
    //    mmengine, ...) CANNOT move it off the cu128 build installed above — without this they
    //    swap in a PyPI/CPU torch, and the next Update's torch step re-downloads the 3.3GB wheel.
    {
      method: "shell.run",
      params: { message: MOVA_PIP + " install -c tools/mova_torch_constraints.txt ./ext/mova" },
    },
    // huggingface_hub for the weight fetch below (descript-audiotools/diffusers pull most deps).
    {
      method: "shell.run",
      params: { message: MOVA_PIP + " install huggingface_hub" },
    },

    // ── MOVA TRAINING (Gym) decode dep: PyAV. MOVA's training dataset decodes video+audio; upstream
    //    uses torchcodec (cu13x wheels + FFmpeg sys libs — absent on this torch 2.7/cu128 env), so
    //    patch_mova_pipeline.py routes decode through PyAV (self-contained FFmpeg, all platforms) via
    //    a guarded fallback. That makes the Gym train MOVA A/V LoRAs on Windows + Linux on this env.
    //    bitsandbytes (NF4 trainer) was installed above. torchcodec is left out: it can't load here
    //    and the PyAV fallback supersedes it (a cu13x env that has it still gets used automatically).
    {
      method: "shell.run",
      params: {
        message: MOVA_PIP + " install av pyloudnorm || echo [basiwan] PyAV/pyloudnorm install failed - MOVA training decode / continuation audio-stitch unavailable (single-clip generation still works)",
      },
    },

    // ── MOVA-360p weights (~77.7GB) + SDXL base (T2AV style-reference) into ./checkpoints
    //    (resumable; one consolidated acquirer; the 'mova' group includes both).
    {
      method: "shell.run",
      params: {
        message: MOVA_PY + " tools/ensure_weights.py --groups mova || echo [basiwan] MOVA weights incomplete - resumes on re-run/first use",
      },
    },

    // ── IP-Adapter weights for the T2AV style-reference maker (SDXL + IP-Adapter InstantStyle).
    //    diffusers' load_ip_adapter reads the default HF cache, so prefetch the SDXL vit-h adapter
    //    + its CLIP image encoder there now -> the first MOVA generation is fully offline. Best-
    //    effort (diffusers will fetch on first use if this is skipped). h94/IP-Adapter = Apache-2.0.
    {
      method: "shell.run",
      params: {
        message: MOVA_PY + " -c \"from huggingface_hub import hf_hub_download as d; d('h94/IP-Adapter','sdxl_models/ip-adapter_sdxl_vit-h.safetensors'); [d('h94/IP-Adapter', f) for f in ['models/image_encoder/config.json','models/image_encoder/model.safetensors']]; print('[basiwan] IP-Adapter prefetched')\" || echo [basiwan] IP-Adapter prefetch skipped - fetched on first MOVA generation",
      },
    },

    {
      method: "log",
      params: {
        text: "MOVA (audio+video) install done. ~77.7GB weights fetched to ./checkpoints/MOVA-360p (resumable — re-run to finish any partial). MOVA A/V training (Gym) and generation (Studio) both run on Windows + Linux; generation needs ~40GB+ system RAM (50-80GB recommended).",
      },
    },
  ],
};
