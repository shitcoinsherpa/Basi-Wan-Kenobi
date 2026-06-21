"""ETA — pre-flight wall estimate + an EMA wall-time cache that learns from
every completed run. Pure functions (no torch, no gradio) so the formula + cache
are unit-testable in tools/_smoke_eta.py.

Design:
  wall ~= ANCHOR * (steps/4) * cfg * (frames/17)^1.5 * (pixels / (1280*720))
  ANCHOR = 64 s, the MEASURED champion p720/17f/4-step/guide=1 wall.
  cfg = 2.0 when guide != 1.0 (CFG runs cond+uncond every step), else 1.0.
  frames^1.5 = attention is superlinear in sequence length; pixels ~ linear.
This is ROUGH (order-of-magnitude honesty, never a promise). A MEASURED EMA for the
exact (gpu, shape, steps, cfg) key always beats the formula — so the estimate
sharpens as the user runs. We label the source ('measured'/'formula') and NEVER
present a formula guess as a measured number.

Cache: JSON at BASIWAN_WALLTIMES (default <repo>/cache/wall_times.json), keyed
"<gpu>|<W>x<H>x<F>|s<steps>|<cfg|nocfg>" -> {"ema": seconds, "n": samples}.
EMA alpha=0.5 (recent runs dominate; one anomaly can't poison the estimate).
"""
from __future__ import annotations
import json
from pathlib import Path

ANCHOR_S = 64.0          # measured champion: p720(1280x720)/17f/4-step/guide=1
_ANCHOR_PX = 1280 * 720
_ANCHOR_F = 17
_ANCHOR_STEPS = 4
EMA_ALPHA = 0.5


def _cfg_tag(guide) -> str:
    try:
        return "nocfg" if abs(float(guide) - 1.0) < 1e-6 else "cfg"
    except (TypeError, ValueError):
        return "cfg"


def formula_wall(width: int, height: int, frames: int, steps: int, guide) -> float:
    """Pure scaling estimate from the measured anchor. Seconds. Never < 1.0."""
    px = (int(width) * int(height)) / _ANCHOR_PX
    frame_factor = (max(int(frames), 1) / _ANCHOR_F) ** 1.5
    step_factor = max(int(steps), 1) / _ANCHOR_STEPS
    cfg_factor = 2.0 if _cfg_tag(guide) == "cfg" else 1.0
    return max(ANCHOR_S * step_factor * cfg_factor * frame_factor * px, 1.0)


def cache_key(gpu: str, width: int, height: int, frames: int, steps: int, guide) -> str:
    g = (gpu or "unknown").replace("|", "_")
    return f"{g}|{int(width)}x{int(height)}x{int(frames)}|s{int(steps)}|{_cfg_tag(guide)}"


def _cache_path() -> Path:
    import os
    env = os.environ.get("BASIWAN_WALLTIMES")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "cache" / "wall_times.json"


def load_cache(path: Path | None = None) -> dict:
    p = path or _cache_path()
    try:
        return json.loads(p.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def save_cache(cache: dict, path: Path | None = None) -> None:
    p = path or _cache_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, indent=2, sort_keys=True))
    tmp.replace(p)          # atomic


def estimate(gpu: str, width: int, height: int, frames: int, steps: int, guide,
             cache: dict | None = None) -> tuple[float, str]:
    """Best wall estimate (seconds, source). source='measured' if a learned EMA
    exists for this exact key, else 'formula'. The caller must label it so a
    formula guess is never shown as a measured fact."""
    c = cache if cache is not None else load_cache()
    key = cache_key(gpu, width, height, frames, steps, guide)
    hit = c.get(key)
    if hit and hit.get("ema", 0) > 0:
        return float(hit["ema"]), "measured"
    return formula_wall(width, height, frames, steps, guide), "formula"


def record(gpu: str, width: int, height: int, frames: int, steps: int, guide,
           wall_s: float, path: Path | None = None) -> dict:
    """Fold a completed run's MEASURED wall into the EMA cache. Returns the cache."""
    if not (wall_s and wall_s > 0):
        return load_cache(path)
    p = path or _cache_path()
    c = load_cache(p)
    key = cache_key(gpu, width, height, frames, steps, guide)
    prev = c.get(key, {})
    old = float(prev.get("ema", 0) or 0)
    n = int(prev.get("n", 0))
    ema = float(wall_s) if n == 0 else (EMA_ALPHA * float(wall_s) + (1 - EMA_ALPHA) * old)
    c[key] = {"ema": round(ema, 2), "n": n + 1}
    save_cache(c, p)
    return c


def remaining_live(step_i: int, n_steps: int, elapsed_s: float) -> float:
    """Live in-run ETA for the diffusion loop: project remaining from the steps
    completed so far. step_i is 0-based, just-completed step index. Seconds."""
    done = max(step_i + 1, 1)
    n = max(int(n_steps), 1)
    if done >= n:
        return 0.0
    per_step = elapsed_s / done
    return max((n - done) * per_step, 0.0)


def human(seconds: float) -> str:
    s = max(int(round(seconds)), 0)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"
