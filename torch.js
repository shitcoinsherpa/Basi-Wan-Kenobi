// BASIWAN — Torch install (extracted per Pinokio canonical pattern).
// Called by install.js as `script.start uri: "torch.js"`.
// Branches on {{platform, gpu, arch}}; uses uv pip per house style.
module.exports = {
  run: [
    // Windows + NVIDIA
    {
      when: "{{platform === 'win32' && gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : 'env'}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
        ],
      },
    },
    // Linux + NVIDIA
    {
      when: "{{platform === 'linux' && gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : 'env'}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128",
        ],
      },
    },
    // Linux + AMD ROCm
    {
      when: "{{platform === 'linux' && gpu === 'amd'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : 'env'}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.3",
        ],
      },
    },
    // macOS Apple Silicon (MPS via PyTorch device abstraction)
    {
      when: "{{platform === 'darwin' && arch === 'arm64'}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : 'env'}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision torchaudio",
        ],
      },
    },
    // CPU fallback (and any other path)
    {
      when: "{{gpu === 'none' || gpu === undefined || (platform === 'win32' && gpu === 'amd')}}",
      method: "shell.run",
      params: {
        venv: "{{args && args.venv ? args.venv : 'env'}}",
        path: "{{args && args.path ? args.path : '.'}}",
        message: [
          "uv pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu",
        ],
      },
    },
  ],
};
