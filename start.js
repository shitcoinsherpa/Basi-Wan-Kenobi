// BASIWAN — Pinokio start script. Daemon mode; Gradio URL captured via the
// canonical "/http:\\/\\/\\S+/" pattern then explicitly stored via local.set
// so pinokio.js menu can pick it up reliably (no reliance on Pinokio's
// runtime auto-capture, which is fragile).
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // BASI's preview path defaults to /home/ryot/venvs/wan22/bin/python
          // (Faster-Wan2.2). Pinokio install layout puts everything under one
          // venv ("env/") — override per-platform: Linux/macOS use env/bin/python,
          // Windows uses env\Scripts\python.exe.
          BASIWAN_VENV_PYTHON: "{{platform === 'win32' ? cwd + '\\\\env\\\\Scripts\\\\python.exe' : cwd + '/env/bin/python'}}",
          // Inference runner perf env. Async block-swap is opt-in; the runner
          // auto-detects platform and falls back to sync on WSL2 dxgkrnl.
          BASIWAN_BLOCK_SWAP_ASYNC: "1",
          // PyTorch caching allocator: expandable_segments helps with
          // fragmentation at p720+; pinned_use_cuda_host_register routes
          // pinned allocations via RssAnon (not tmpfs) on Linux for the async
          // ring-buffer path.
          PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True,pinned_use_cuda_host_register:True",
          // BASIWAN pack cache lives on ext4 (not /mnt/d 9p) to avoid mid-process
          // corruption. ~/.cache/marlin_packs is the canonical location.
          BASIWAN_PACK_CACHE_DIR: "~/.cache/marlin_packs",
          BASIWAN_USE_PACK_CACHE: "1",
          BASIWAN_NO_POOL: "1",
        },
        message: "python app.py",
        on: [{ event: "/http:\\/\\/\\S+/", done: true }],
      },
    },
    { method: "local.set", params: { url: "{{input.event[0]}}" } },
  ],
};
