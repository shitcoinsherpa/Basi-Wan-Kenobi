"""MOVA A/V INFERENCE plumbing (text -> audio+video), parallel to basi/mova_train.py.

MOVA inference is the user-facing counterpart of the Gym's MOVA training: it runs a trained
MOVA LoRA (or the base) to generate a joint audio+video clip from a TEXT PROMPT (T2AV — the
shipped workflow; the reference frame is a neutral gray frame synthesized by tools/mova_sample.py,
so no image upload is required).

Unlike Wan Studio (the persistent GGUF worker), MOVA runs in its OWN venv `env_mova` (installed by
install_mova.js: plain torch + SDPA, no flash/torchcodec/yunchang — cross-platform) and is spawned
as a one-off subprocess, like the trainer. This module is pure/testable: path resolution, LoRA
discovery, the command builder, and capability gating. app.py imports these + streams the spawn.

ASCII.
"""
import os
from pathlib import Path

BASI_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = BASI_ROOT / "outputs"
MOVA_SAMPLE = BASI_ROOT / "tools" / "mova_sample.py"


def resolve_mova_python() -> str:
    """The env_mova interpreter. Override with BASIWAN_MOVA_PYTHON. Default to the Pinokio
    env_mova venv (Windows: env_mova\\Scripts\\python.exe, Unix: env_mova/bin/python). Dev
    fallback: the WSL conda 'mova' env. Returns the best-guess path even if absent — callers
    gate on mova_installed()."""
    env = os.environ.get("BASIWAN_MOVA_PYTHON")
    if env:
        return env
    # env_mova is a CONDA env (python 3.12; the main env is 3.10 and MOVA needs >=3.12).
    # Conda puts python.exe at the env ROOT on Windows (NOT Scripts/) and bin/python on Linux.
    # Check both the conda layout and the venv layout (Scripts/) for robustness.
    em = BASI_ROOT / "env_mova"
    cands = [em / "python.exe", em / "Scripts" / "python.exe"] if os.name == "nt" \
        else [em / "bin" / "python"]
    for c in cands:
        if c.exists():
            return str(c)
    dev = Path.home() / "miniforge3" / "envs" / "mova" / "bin" / "python"
    if dev.exists():
        return str(dev)
    return str(cands[0])


def mova_spawn_env() -> dict:
    """Environment for spawning the env_mova subprocess from the app.

    The app runs INSIDE Pinokio's activated MAIN env (its conda activation exports CONDA_PREFIX
    and prepends the main env's DLL dirs — `env/`, `env\\Library\\bin` — to PATH). Spawning
    env_mova's python WITHOUT re-activating leaves those main-env DLL dirs first on PATH, so
    env_mova loads the WRONG CUDA/MKL DLLs -> access violation (0xC0000005) during model load
    (the MOVA-generation crash). The Wan worker doesn't hit this because it IS the main env.
    Here we re-activate env_mova exactly as Pinokio's `venv: env_mova` would: put env_mova's
    conda DLL dirs first on PATH, drop the main env's conda dirs, set CONDA_PREFIX, and clear
    PYTHONPATH/PYTHONHOME so the child can't pull main-env site-packages."""
    env = os.environ.copy()
    em = BASI_ROOT / "env_mova"
    if os.name == "nt":
        conda_dirs = [em, em / "Library" / "mingw-w64" / "bin", em / "Library" / "usr" / "bin",
                      em / "Library" / "bin", em / "Scripts"]
    else:
        conda_dirs = [em / "bin"]
    main_env = str(BASI_ROOT / "env").lower()
    kept = [p for p in env.get("PATH", "").split(os.pathsep)
            if p and main_env not in p.replace("/", os.sep).lower()]
    env["PATH"] = os.pathsep.join([str(d) for d in conda_dirs if d.exists()] + kept)
    env["CONDA_PREFIX"] = str(em)
    for k in ("CONDA_DEFAULT_ENV", "CONDA_PROMPT_MODIFIER", "PYTHONPATH", "PYTHONHOME"):
        env.pop(k, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def mova_base_dir() -> Path:
    """The MOVA-360p weights dir. Honors BASIWAN_CKPT_DIR (shared model drive), else
    <repo>/checkpoints/MOVA-360p — the same resolution the rest of the app uses."""
    base = Path(os.environ.get("BASIWAN_CKPT_DIR", str(BASI_ROOT / "checkpoints")))
    return base / "MOVA-360p"


def mova_installed() -> bool:
    """True iff MOVA can run: the env_mova interpreter exists AND the base weights are present."""
    return Path(resolve_mova_python()).exists() and (mova_base_dir() / "model_index.json").exists()


def list_mova_loras() -> list[tuple[str, str]]:
    """Discover trained MOVA LoRAs for the Studio dropdown — ONE clean entry per workspace
    (its final checkpoint), labeled by the workspace name. Per-step checkpoints are training
    history, not generation picks, so they're not listed; if a workspace has no
    checkpoint_final (e.g. a converge-stopped run), its newest checkpoint_stepN is used.
    Workspaces whose name starts with '_' are skipped (archive/scratch convention, e.g.
    outputs/_mova_archive, _studio, _test). Returns [(label, checkpoint_dir)], base first."""
    out: list[tuple[str, str]] = [("(none — MOVA base, no LoRA)", "")]
    if not OUTPUTS_ROOT.exists():
        return out
    def _label(ws):
        # Optional friendly display name: a `.basi_label` file in the workspace overrides the
        # dir name (so a LoRA can read e.g. "Moral Orel" without renaming the folder). Falls
        # back to the directory name.
        f = ws / ".basi_label"
        try:
            if f.exists():
                t = f.read_text(encoding="utf-8").strip()
                if t:
                    return t
        except Exception:
            pass
        return ws.name
    rows: list[tuple[str, str]] = []  # (label, dir)
    for ws in sorted(OUTPUTS_ROOT.iterdir()):
        if not ws.is_dir() or ws.name.startswith("_"):
            continue
        final = ws / "checkpoint_final"
        if (final / "lora_weights.pt").exists():
            rows.append((_label(ws), str(final)))
            continue
        steps = [(int(c.name[len("checkpoint_step"):]), c) for c in ws.iterdir()
                 if c.is_dir() and c.name.startswith("checkpoint_step")
                 and c.name[len("checkpoint_step"):].isdigit()
                 and (c / "lora_weights.pt").exists()]
        if steps:
            rows.append((_label(ws), str(max(steps, key=lambda s: s[0])[1])))
    rows.sort(key=lambda r: r[0].lower())
    out.extend(rows)
    return out


def gen_mova_sample_command(*, python: str, base: str, lora_dir: str | None, prompt: str,
                            out_dir: str, tag: str, height: int, width: int, frames: int,
                            steps: int, cfg: float = 5.0) -> list[str]:
    """Build the cross-platform mova_sample.py invocation (run directly with the env_mova python —
    no bash/WSL). No --ref -> the sampler synthesizes a neutral gray frame (T2AV). NF4-base
    emulation auto-enables from the LoRA's lora_config (base=nf4), matching training precision."""
    cmd = [str(python), str(MOVA_SAMPLE),
           "--base", str(base), "--prompt", prompt,
           "--out", str(out_dir), "--tag", tag,
           "--height", str(height), "--width", str(width),
           "--frames", str(frames), "--steps", str(steps),
           "--cfg", str(cfg), "--offload", "group"]
    if lora_dir:
        cmd += ["--lora", str(lora_dir)]
    return cmd


def mova_caps(vram_gb: int, ram_gb: int) -> dict:
    """MOVA A/V generation capability for this machine. Pure: (vram_gb, ram_gb) ->
    {available, note}. MOVA-360p runs via group offload (~12GB VRAM) but pages the model
    through HOST RAM — the real gate. Measured/vendor: ~50-80GB host RAM for 360p group offload.
    Gate: VRAM >= 11GB AND RAM >= 40GB; recommend 64GB+. The UI disables the mode + explains
    when a machine can't run it (don't brick small boxes)."""
    if vram_gb and vram_gb < 11:
        return {"available": False,
                "note": f"needs ~12GB VRAM (group offload); this GPU has ~{vram_gb}GB."}
    if ram_gb and ram_gb < 40:
        return {"available": False,
                "note": (f"needs ~50-80GB system RAM (MOVA pages the model through host RAM); "
                         f"this machine has ~{ram_gb}GB. 64GB+ recommended.")}
    head = "ready" if ram_gb >= 60 else "tight on RAM (64GB+ recommended)"
    return {"available": True,
            "note": (f"{head} — ~12GB VRAM via group offload, ~{ram_gb or '?'}GB system RAM. "
                     "240p joint A/V, ~minutes/clip.")}


def detect_ram_gb() -> int:
    """Total system RAM in GB (psutil), 0 if unavailable."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024**3))
    except Exception:
        return 0
