// BASIWAN — Pinokio start script. Daemon mode; Gradio URL captured via a
// "127.0.0.1"-anchored pattern then explicitly stored via local.set so
// pinokio.js menu can pick it up reliably.
//
// 2026-06-09: bind Gradio to 127.0.0.1 explicitly (was app.py default
// 0.0.0.0). Pinokio's popout webview routes the captured URL literally —
// "http://0.0.0.0:7860" 404s in the embedded Electron browser on Windows.
// Cocktailpeanut canonical apps (fluxgym/cogstudio) all bind 127.0.0.1.
// The capture regex is also tightened to only accept 127.0.0.1 / localhost
// so a stray "http://huggingface.co/..." log line can't be misclaimed.
module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          // Force Gradio to bind+print "http://127.0.0.1:PORT" so the
          // popout-rendered link is a routable loopback URL on every
          // platform. 0.0.0.0 is INADDR_ANY at the listener but a literal
          // host string at the browser — the embedded webview won't route it.
          GRADIO_SERVER_NAME: "127.0.0.1",
          // The preview path's built-in default targets a standalone dev venv.
          // Pinokio install layout puts everything under one venv ("env/") —
          // override per-platform: Linux/macOS use env/bin/python, Windows uses
          // env\Scripts\python.exe.
          BASIWAN_VENV_PYTHON: "{{platform === 'win32' ? cwd + '\\\\env\\\\Scripts\\\\python.exe' : cwd + '/env/bin/python'}}",
          // 2026-06-08 measured (memory/canonical_ship_recipe_2026-06-08.md):
          // Async ring buffer on Windows native OOMs at step 0 (K=4 ring +
          // resident blocks + activations exceed the caching allocator's
          // contiguous headroom). Sync swap fits and ships sub-Linux wall
          // (61.6s vs Linux 76s on the same p720_17f recipe).
          BASIWAN_BLOCK_SWAP_ASYNC: "{{platform === 'win32' ? '0' : '1'}}",
          // 2026-06-08 measured: expandable_segments is silently non-functional
          // on Windows pip wheels (PYTORCH_C10_DRIVER_API_SUPPORTED absent).
          // pinned_use_cuda_host_register is a Linux-only fix for the WSL2
          // tmpfs detour. On Windows we use garbage_collection_threshold:0.8
          // to reclaim cached blocks before the next allocation pressure
          // peak. DO NOT add max_split_size_mb — measured 7× regression at
          // p720_33f (memory/windows_full_vae_investigation_closed_2026-06-08.md).
          PYTORCH_CUDA_ALLOC_CONF: "{{platform === 'win32' ? 'garbage_collection_threshold:0.8' : 'expandable_segments:True,pinned_use_cuda_host_register:True'}}",
          // BASIWAN_VAE_BF16=1: bf16 VAE autocast, ship default (memory/audit_vae_bf16_ships_opt_in_2026-06-05.md).
          BASIWAN_VAE_BF16: "1",
          // BASIWAN v2 kernel (Q4_K+Q6_K rewrite). Ship default 2026-06-06.
          BASIWAN_V2: "1",
          // [2026-06-10] BASIWAN_RMS_BF16 / BASIWAN_LN_BF16 are deliberately
          // UNSET. model.py auto-switches norms to bf16 only at seq>50000
          // (p720_81f) where fp32 norms cost ~3 GB of transients per block
          // call — pinning "0" here overrode that gate and pushed 81f past
          // the 24 GB VRAM wall into allocator thrash (the 2026-06-10
          // never-finishing generation). At seq≤50000 (17f/33f/49f) the
          // auto path IS fp32 — ship recipe unchanged. bf16-norm quality is
          // bit-identical (FFFn5 calibration, audit_FFFl/FFFn5 memos).
          // BASIWAN pack cache. Expanded via Path.expanduser() in
          // basiwan_pack_cache.py::_cache_dir() (pathlib does NOT expand
          // "~" by itself — that bug cost every Windows cold start a full
          // ~8 min re-pack until 2026-06-09). Resolves to
          // %USERPROFILE%\.cache\marlin_packs on Windows native,
          // ~/.cache/marlin_packs (ext4, not /mnt 9p) on Linux.
          BASIWAN_PACK_CACHE_DIR: "~/.cache/marlin_packs",
          BASIWAN_USE_PACK_CACHE: "1",
          BASIWAN_NO_POOL: "1",
          // 2026-06-09 persistent-worker default on. Spawns the BASIWAN
          // runner ONCE per Pinokio Start (cold ~6 min on first Studio
          // Generate click) and reuses it for every subsequent click —
          // ~60s steady-state vs ~500s legacy subprocess-per-click. Set
          // to "0" to fall back to the legacy path (debug only). See
          // memory/persistent_worker_ship_2026-06-09.md.
          BASIWAN_PERSISTENT_WORKER: "1",
          // mmap pre-warm of the 9.4 GB pack files; eliminates the 100-160s
          // cold OS page-fault penalty during the first inference. Always on.
          BASIWAN_MMAP_PREWARM: "1",
        },
        message: "python app.py",
        // Only match loopback URLs. "http:" anchored to 127.0.0.1 or
        // localhost so an upstream library logging a hub/cdn URL can't
        // be misclaimed as the Gradio bind address.
        on: [{ event: "/http:\\/\\/(?:127\\.0\\.0\\.1|localhost):\\d+/", done: true }],
      },
    },
    { method: "local.set", params: { url: "{{input.event[0]}}" } },
  ],
};
