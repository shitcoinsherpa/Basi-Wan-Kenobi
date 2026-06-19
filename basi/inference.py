"""In-process Wan2.2 inference wrapper for the BASIWAN Studio tab.

Loads the BASIWAN pipe ONCE at app startup (in a background thread so
the UI is immediately usable). Subsequent generate() calls hit the
already-loaded pipe — wall time reflects ONLY inference, not weight load.

This wraps `tools/run_one_video_gguf.py` by re-using its helpers
(`_build_pipe_gguf`, `_install_chunked_ffn_gguf`, `_install_block_swap_gguf`,
etc.) without firing its CLI. We fake the argparse namespace via sys.argv
injection BEFORE importing the runner module — the runner's `args =
parse_args()` then resolves against our injected args at import time.
"""
from __future__ import annotations

import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Optional


# Module-level state shared with app.py
class _State:
    pipe: Any = None
    size_configs: Any = None
    status: str = "uninitialized"  # "uninitialized" | "loading" | "ready" | "error"
    error: Optional[str] = None
    load_time_s: float = 0.0
    lock: threading.Lock = threading.Lock()


STATE = _State()


def status_text() -> str:
    """Human-readable status for the Studio tab indicator."""
    with STATE.lock:
        if STATE.status == "uninitialized":
            return "**Studio**: inference pipe not yet started."
        if STATE.status == "loading":
            return ("**Studio**: loading Wan2.2 weights (one-time, "
                    "~30-60 sec on warm SSD)...")
        if STATE.status == "ready":
            return (f"**Studio**: ready. Pipe loaded in {STATE.load_time_s:.1f}s.")
        return f"**Studio error**: {STATE.error}"


def warm_up_pipe_async(
    gguf_high: str,
    gguf_low: str,
    base_dir: str,
    lora_dir: Optional[str] = None,
    lora_strength: float = 1.0,
    lora_mode: str = "forward",
    ffn_chunk_size: int = 4096,
    block_swap_n: int = 2,
):
    """Kick off the warmup in a background thread and return immediately."""
    with STATE.lock:
        if STATE.status in ("loading", "ready"):
            return  # already kicked off
        STATE.status = "loading"
        STATE.error = None

    def _do_warmup():
        try:
            t0 = time.time()
            _warm_up_pipe_sync(
                gguf_high=gguf_high,
                gguf_low=gguf_low,
                base_dir=base_dir,
                lora_dir=lora_dir,
                lora_strength=lora_strength,
                lora_mode=lora_mode,
                ffn_chunk_size=ffn_chunk_size,
                block_swap_n=block_swap_n,
            )
            elapsed = time.time() - t0
            with STATE.lock:
                STATE.status = "ready"
                STATE.load_time_s = elapsed
            print(f"[basiwan-inference] pipe ready in {elapsed:.1f}s", flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            with STATE.lock:
                STATE.status = "error"
                STATE.error = f"{type(exc).__name__}: {exc}"

    threading.Thread(target=_do_warmup, daemon=True, name="basiwan-warmup").start()


def _warm_up_pipe_sync(*, gguf_high, gguf_low, base_dir, lora_dir,
                        lora_strength, lora_mode, ffn_chunk_size, block_swap_n):
    """Synchronous pipe build. Runs inside the warmup thread."""
    # Inject sys.argv so the runner's argparse resolves against our config.
    app_root = Path(__file__).resolve().parent.parent
    tmp_dir = app_root / "outputs" / "_studio"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fake_mp4 = tmp_dir / ".warmup.mp4"
    fake_meta = tmp_dir / ".warmup.json"
    vae_path = str(Path(base_dir) / "Wan2.1_VAE.pth")

    argv = [
        "run_one_video_gguf.py",
        "--gguf-high", str(gguf_high),
        "--gguf-low", str(gguf_low),
        "--vae", vae_path,
        "--base-dir", str(base_dir),
        "--lora-strength", str(lora_strength),
        "--lora-mode", lora_mode,
        "--ffn-chunk-size", str(int(ffn_chunk_size)),
        "--block-swap-n", str(int(block_swap_n)),
        "--prompt", "warmup",
        "--width", "1280",
        "--height", "720",
        "--frames", "17",
        "--out", str(fake_mp4),
        "--meta", str(fake_meta),
    ]
    if lora_dir:
        argv += ["--lora-dir", str(lora_dir)]

    # Skip the kernel-dependent prepack on Windows (basiwan_q4_kernel needs
    # nvcc + MSVC). The bare_gguf BareGGMLLinear.forward runtime-dequants
    # on each call instead. Slower per step but no compile setup needed.
    os.environ.setdefault("BASIWAN_V2", "0")
    os.environ.setdefault("BASIWAN_SKIP_PREPACK", "1")
    os.environ.setdefault("BASIWAN_USE_PACK_CACHE", "0")
    os.environ.setdefault("BASIWAN_NO_POOL", "1")

    # The runner expects to be run from the app root (uses relative imports
    # like `from gguf_vendor.bare_gguf import ...`). Make sure the import
    # search path is correct.
    sys.path.insert(0, str(app_root))
    sys.path.insert(0, str(app_root / "tools"))
    sys.path.insert(0, str(app_root / "tools" / "gguf_vendor"))

    saved_argv = sys.argv
    saved_cwd = os.getcwd()
    sys.argv = argv
    os.chdir(str(app_root))
    try:
        # First import — argparse runs here against our injected argv.
        import importlib
        if "run_one_video_gguf" in sys.modules:
            runner = importlib.reload(sys.modules["run_one_video_gguf"])
        else:
            import run_one_video_gguf as runner  # type: ignore

        # Build the pipe + install all the gates.
        pipe, size_configs = runner._build_pipe_gguf(  # type: ignore
            runner.args.gguf_high,
            runner.args.gguf_low,
            runner.args.base_dir,
        )
        if ffn_chunk_size > 0:
            runner._install_chunked_ffn_gguf(pipe, ffn_chunk_size)
        if block_swap_n >= 0:
            runner._install_block_swap_gguf(pipe, block_swap_n)
        with STATE.lock:
            STATE.pipe = pipe
            STATE.size_configs = size_configs
        # Keep runner module available so subsequent generate() calls
        # can use its helpers without re-importing.
        STATE.runner = runner  # type: ignore[attr-defined]
    finally:
        sys.argv = saved_argv
        os.chdir(saved_cwd)


def generate(
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    guide: float,
    seed: int,
    out_mp4: Path,
) -> dict:
    """Run inference on the warm pipe. Returns metadata dict.
    Caller must verify STATE.status == 'ready' before calling."""
    with STATE.lock:
        pipe = STATE.pipe
        size_configs = STATE.size_configs
        runner = getattr(STATE, "runner", None)
    if pipe is None or runner is None:
        raise RuntimeError("Pipe not ready. Wait for STATE.status == 'ready'.")

    t0 = time.time()
    size = size_configs.get(f"{width}*{height}", (width, height))
    if seed is None or seed < 0:
        import random
        seed = random.randint(0, 2**31 - 1)
    video = pipe.generate(
        input_prompt=prompt,
        size=size,
        frame_num=int(frames),
        sampling_steps=int(steps),
        seed=int(seed),
        guide_scale=(float(guide), float(guide)),
        offload_model=True,
    )
    wall = time.time() - t0

    # Write the mp4. Video comes back as (C, T, H, W) in [-1, 1] tensor.
    # Normalize → uint8 → permute → imageio.mimwrite.
    import torch
    import numpy as np
    import imageio
    video_cpu = video.clamp(-1, 1).float().cpu()
    if video_cpu.dim() == 4:
        # (C, T, H, W) → (T, H, W, C)
        video_uint = ((video_cpu + 1.0) * 127.5).clamp(0, 255).byte()
        video_uint = video_uint.permute(1, 2, 3, 0).numpy()
    elif video_cpu.dim() == 5:
        # (B, C, T, H, W) — take first batch
        video_uint = ((video_cpu[0] + 1.0) * 127.5).clamp(0, 255).byte()
        video_uint = video_uint.permute(1, 2, 3, 0).numpy()
    else:
        raise RuntimeError(f"unexpected video tensor shape: {tuple(video.shape)}")
    fps = 16  # Wan2.2 default
    imageio.mimwrite(
        str(out_mp4), video_uint, fps=fps,
        codec="libx264", quality=8, pixelformat="yuv420p",
    )
    return {
        "prompt": prompt,
        "size": (width, height),
        "frames": int(frames),
        "steps": int(steps),
        "guide": float(guide),
        "seed": int(seed),
        "wall_s": round(wall, 2),
        "out": str(out_mp4),
    }
