"""MOVA A/V INFERENCE plumbing (text -> audio+video), parallel to basi/mova_train.py.

MOVA inference is the user-facing counterpart of the Gym's MOVA training: it runs a trained
MOVA LoRA (or the base) to generate a joint audio+video clip from a text prompt PLUS a REFERENCE
IMAGE. MOVA is I2V — the reference frame is the visual anchor that carries style/material/palette
(the prompt drives content). A LoRA trained on a given style MUST be sampled against an in-style
reference frame or the style is lost and the base model's photoreal default shows through. Each
trained LoRA ships a representative `ref.png` in its workspace (the Gym picks an
in-style frame from the training clips); the user can also upload their own reference. Only as a
last resort, with no ref at all, does tools/mova_sample.py synthesize a neutral gray frame.

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
    """The env_mova interpreter. Override with BASIWAN_MOVA_PYTHON (e.g. a dev box pointing at a
    conda env). Default to the Pinokio env_mova venv that install_mova.js creates (Windows: the
    conda env ROOT env_mova\\python.exe, also Scripts\\python.exe; Unix: env_mova/bin/python).
    Returns the best-guess path even if absent — callers gate on mova_installed()."""
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


def mova_lora_ref(lora_dir: str | None) -> str | None:
    """Resolve the I2V reference frame for a MOVA LoRA checkpoint dir. By convention the
    representative in-style frame lives at <workspace>/ref.png (the workspace is the checkpoint
    dir's parent; the Gym writes it from the training clips). Returns the path if present, else
    None (caller may fall back to a user upload or, last resort, a synthesized gray frame)."""
    if not lora_dir:
        return None
    ref = Path(lora_dir).parent / "ref.png"
    return str(ref) if ref.exists() else None


STYLE_REF_MAKER = BASI_ROOT / "tools" / "make_style_reference.py"


def resolve_sdxl_base() -> str | None:
    """Resolve the SDXL base checkpoint for the T2AV style-reference maker. install_mova.js fetches
    sd_xl_base_1.0.safetensors into checkpoints (BASIWAN_CKPT_DIR or <repo>/checkpoints) via the
    'mova' ensure_weights group. Override with BASIWAN_SDXL_BASE. Returns a path string (Windows-
    native on Windows so diffusers from_single_file accepts it) or None if not present."""
    env = os.environ.get("BASIWAN_SDXL_BASE")
    if env and Path(env).exists():
        return env
    ckpt = Path(os.environ.get("BASIWAN_CKPT_DIR", str(BASI_ROOT / "checkpoints")))
    p = ckpt / "sd_xl_base_1.0.safetensors"
    return str(p) if p.exists() else None


def mova_lora_style_ref(lora_dir: str | None) -> str | None:
    """Resolve the IP-Adapter STYLE source for a MOVA LoRA. Prefers a dedicated <workspace>/style.png
    (a high-texture CLOSE-UP frame the Gym picks specifically for style transfer — close character
    shots transfer a niche style far better than the wide I2V ref.png). Falls back to ref.png
    (the I2V anchor frame) if no style.png. Returns a path or None."""
    if not lora_dir:
        return None
    ws = Path(lora_dir).parent
    for name in ("style.png", "ref.png"):
        p = ws / name
        if p.exists():
            return str(p)
    return None


def visual_only_prompt(prompt: str) -> str:
    """Strip the spoken-words clause from a MOVA prompt for the SDXL style reference. The reference
    is an IMAGE — only the visual scene matters; the 'The boy says, in English, "..."' clause is
    speech and would bleed text/noise into the still. Cut at the first speech marker; keep the
    leading visual sentence(s). Robust to the magic-rewrite's phrasings (He/She/They/<name> says,
    'says, in English', 'asks', etc.)."""
    import re
    p = (prompt or "").strip()
    # cut at the first ", in English," speech lead-in or a '<subject> says/asks/answers,' marker.
    m = re.search(r"\b(?:says|asks|answers|replies|responds)\b\s*,?\s*(?:in\s+English\s*,?\s*)?[\"“]", p, re.I)
    if not m:
        m = re.search(r",\s*in\s+English\s*,", p, re.I)
    if m:
        p = p[:m.start()].rstrip().rstrip(",.")
        # Cutting at the speech verb can leave a dangling verbless lead-in (e.g. "...the dragon. He"
        # / "...table. The boy"). Drop a short trailing fragment after the last sentence boundary if
        # it has no verb (just a subject) so the SDXL prompt ends on a clean visual sentence.
        head, sep, tail = p.rpartition(".")
        if sep and head.strip():
            words = tail.split()
            has_verb = re.search(r"\b(?:is|are|stand|stands|sit|sits|walk|walks|talk|talks|turn|turns|"
                                 r"hold|holds|look|looks|speak|speaks|gestures?|faces?)\b", tail, re.I)
            if len(words) <= 5 and not has_verb:
                p = head.strip()
    if p and not p.endswith((".", "!", "?")):
        p += "."
    return p


# Verbs that close a leading subject noun-phrase ("a clay boy in a sweater SITS on his bed ...").
_LEAD_VERB = (r"\b(?:sits?|stands?|kneels?|lies|lays?|leans?|reclines?|perches?|rests?|walks?|runs?|"
              r"holds?|looks?|faces?|turns?|talks?|speaks?|gestures?|hugs?|waves?|smiles?|is|are|was|"
              r"were)\b")
_SPEAK_VERB = r"(?:says|asks|answers|replies|responds|whispers|shouts|exclaims|adds|continues)"


def _speaker_token(prompt: str) -> str | None:
    """The subject named as speaking: a pronoun, a Name, or 'the <noun>' right before says/asks/etc."""
    import re
    m = re.search(r"(?:^|[.,;:]\s*)((?:[Tt]he\s+[\w-]+)|[A-Z][\w'’-]+|[Hh]e|[Ss]he|[Tt]hey|[Ii]t)\s+"
                  + _SPEAK_VERB + r"\b", prompt or "")
    return m.group(1).strip() if m else None


def _leading_subject(visual: str):
    """Split a visual sentence into (leading-subject NP, remainder) at its first verb. Returns
    (None, visual) when no clean short subject is found."""
    import re
    m = re.search(_LEAD_VERB, visual, re.I)
    if not m:
        return None, visual
    subj = visual[:m.start()].strip().rstrip(",").strip()
    rest = visual[m.start():].strip()
    if not subj or len(subj.split()) > 12:
        return None, visual
    return subj, rest


def foreground_speaker(visual: str, prompt: str) -> str:
    """Rewrite a visual scene so the SPEAKER is the foreground subject. MOVA assigns the spoken line
    to whichever subject the I2V reference foregrounds, so a two-subject scene ('a boy beside a
    bright red dragon') otherwise lets the more eye-catching subject capture the voice. Identify the
    speaker from the 'X says ...' clause: a pronoun or a name matching the leading subject -> splice
    a close-up lead in front of the action; a NAMED non-leading speaker -> prepend a close-up of
    them. Shallow depth of field pushes the other subjects back. No speech clause -> foreground the
    leading subject (the main subject is the natural focus)."""
    import re
    spk = _speaker_token(prompt)
    subj, rest = _leading_subject(visual)
    DOF = ", shallow depth of field, blurred background"

    def _strip_framing(s):
        s = re.sub(r",?\s*\b(?:medium|wide|long|full|close[- ]?up|establishing|two)\s*[- ]?shot\b",
                   "", s, flags=re.I)
        s = re.sub(r"\s*,\s*,", ", ", s)
        return s.strip().rstrip(". ,").strip()

    is_pronoun = bool(spk) and spk.lower() in ("he", "she", "they", "it")
    matches_lead = bool(spk) and subj is not None and re.search(
        r"\b" + re.escape(re.sub(r"^[Tt]he\s+", "", spk)) + r"\b", subj, re.I) is not None

    if subj is not None and (spk is None or is_pronoun or matches_lead):
        action = _strip_framing(rest)
        lead = f"close-up of {subj} in the foreground, sharp focus, facing the camera"
        # If the action introduces a companion (beside/next to X), relocate them to an explicit
        # background clause: SDXL then binds the companion's attributes (e.g. a dragon's wings) to
        # the companion, not the foreground speaker, and the foreground/background split tells MOVA
        # who is talking. Restrict to clear companion prepositions ('with' is usually an attribute).
        cm = re.search(r"\b(?:beside|next to|alongside)\b\s+(.+)$", action, re.I)
        if cm:
            companion = cm.group(1).strip().rstrip(". ,").strip()
            action_main = action[:cm.start()].strip().rstrip(", ").strip()
            body = f"{lead}, {action_main}" if action_main else lead
            return f"{body}; in the soft-focus background, {companion}, shallow depth of field"
        return f"{lead}, {action}{DOF}"
    if spk and not is_pronoun:
        return (f"close-up of {spk} in the foreground, sharp focus, facing the camera; "
                + _strip_framing(visual) + DOF)
    return _strip_framing(visual) + DOF


def sdxl_reference_prompt(prompt: str, trigger: str = "", style_tag: str | None = None) -> str:
    """Build the SDXL prompt for the style reference from a MOVA prompt. FOUR transforms:
    (1) visual_only_prompt — drop the spoken clause (image needs the scene);
    (2) strip the MOVA/Wan TRIGGER token (e.g. 'moralorel,') — it's a LoRA trigger, meaningless
    noise to SDXL; (3) foreground_speaker — recompose so the SPEAKER is the foreground subject,
    since MOVA binds the voice to whoever the reference foregrounds; (4) APPEND a style descriptor
    (e.g. 'claymation, stop-motion clay figures') — the IP-Adapter image ref ALONE at scene-flexible
    scale 0.6 under-transfers a niche style; with explicit style words the output is strongly
    in-style (same image+scale, cartoon -> clay)."""
    import re
    p = visual_only_prompt(prompt)
    t = (trigger or "").strip().strip(",").strip()
    if t:
        # remove a leading "trigger," / "trigger " and any standalone occurrences
        p = re.sub(r"^\s*" + re.escape(t) + r"\s*,?\s*", "", p, flags=re.I)
        p = re.sub(r"\b" + re.escape(t) + r"\b", "", p, flags=re.I).strip().lstrip(",").strip()
    p = foreground_speaker(p, prompt)
    if style_tag and style_tag.strip():
        p = p.rstrip(". ") + ", " + style_tag.strip()
    return p.strip()


def make_style_reference(*, prompt: str, style_ref: str, out_dir: str, python: str,
                         env: dict | None = None, sdxl_base: str | None = None, trigger: str = "",
                         style_tag: str | None = None,
                         scale: float = 0.6, steps: int = 30, guidance: float = 7.0,
                         seed: int = 42, width: int = 1024, height: int = 1024) -> str | None:
    """Manufacture a per-prompt, style+scene-matched still for the MOVA I2V reference. Spawns
    tools/make_style_reference.py (SDXL + IP-Adapter InstantStyle) via the env_mova `python` --
    style from `style_ref` (an in-style frame; the LoRA workspace's ref.png serves this), scene from
    the prompt. This is the all-users T2AV generalization (scale 0.6 best). Returns the styled
    PNG path, or None if SDXL is
    not provisioned / the run fails (caller falls back to the bundled ref WITH a warning -- never
    silently scene-locks). Runs in env_mova (same env as MOVA), so the caller passes resolve_mova_python().

    WINDOWS: sdxl_base must be a native drive-letter path (diffusers from_single_file rejects /mnt).
    On Windows we hand off the path as-is; the app resolves a Windows path via BASIWAN_CKPT_DIR."""
    import subprocess
    base = sdxl_base or resolve_sdxl_base()
    if not base:
        return None
    od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)
    out_png = od / "styleref.png"
    # A per-LoRA style descriptor (e.g. "claymation, stop-motion clay figures") lives at
    # <workspace>/style_tag.txt (the Gym derives it at train time); it's REQUIRED for niche styles
    # to hold at scale 0.6. Read it if not passed.
    if style_tag is None:
        st = Path(style_ref).parent / "style_tag.txt"
        try:
            style_tag = st.read_text(encoding="utf-8").strip() if st.exists() else None
        except Exception:
            style_tag = None
    vprompt = sdxl_reference_prompt(prompt, trigger=trigger, style_tag=style_tag)
    cmd = [str(python), str(STYLE_REF_MAKER),
           "--sdxl", str(base), "--style-ref", str(style_ref), "--prompt", vprompt,
           "--out", str(out_png), "--scale", str(scale), "--steps", str(steps),
           "--guidance", str(guidance), "--seed", str(seed),
           "--width", str(width), "--height", str(height)]
    proc = subprocess.run(cmd, env=(env or None), capture_output=True, text=True)
    # Parse the machine-readable STYLE_REF_OUT| line; fall back to the known out path if present.
    for line in (proc.stdout or "").splitlines():
        if line.startswith("STYLE_REF_OUT|"):
            p = line.split("|", 1)[1].strip()
            if Path(p).exists():
                return p
    return str(out_png) if out_png.exists() else None


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
