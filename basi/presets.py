"""VRAM-tiered training presets for musubi-tuner Wan2.2 LoRA."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Preset:
    name: str
    vram_gb: int
    blocks_to_swap: int
    network_dim: int
    network_alpha: int
    learning_rate: float
    # Conservative frame cap per VRAM tier. Training adds gradients +
    # optimizer states ≈ 2-3× the inference activation footprint, so the
    # training frontier is well below the inference frontier. These are
    # best-guess starting points; user can override via the slider, but
    # the UI surfaces a may-OOM warning when above the cap.
    max_target_frames: int = 17
    fp8_base: bool = True
    optimizer: str = "adamw8bit"
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    gradient_checkpointing_cpu_offload: bool = False
    attention: str = "sdpa"  # sdpa | flash_attn | xformers | sage_attn
    batch_size: int = 1
    timestep_sampling: str = "shift"
    discrete_flow_shift: float = 3.0
    extra_args: list[str] = field(default_factory=list)


PRESETS: dict[str, Preset] = {
    "8g": Preset(
        name="8G (RTX 4060 / 4060Ti 8GB / 3060 8GB) — experimental",
        vram_gb=8, blocks_to_swap=40, network_dim=8, network_alpha=8,
        learning_rate=2e-4,
        max_target_frames=17,  # 8GB: only 17f safe (no slack)
        gradient_checkpointing_cpu_offload=True,
        # Smallest viable: all 40 blocks swapped, rank-8 LoRA, adafactor.
        # Untested against full training run — exposed for the user to try.
        optimizer="adafactor",
        # Wan2.2 has 40 transformer blocks; swap them all to keep only the
        # active block resident. With rank-8 LoRA we stay under ~7GB peak
        # at training shape per musubi reports — fits 8GB with headroom.
    ),
    "12g": Preset(
        name="12G (RTX 3060 12GB / 4070 / 4070Ti)",
        vram_gb=12, blocks_to_swap=30, network_dim=16, network_alpha=16,
        learning_rate=2e-4,
        max_target_frames=17,  # 12GB: 17f safe; 33f may OOM during VAE cache step
        gradient_checkpointing_cpu_offload=True,
        # AdamW-split for 12G per Flux-Gym precedent; musubi calls it adafactor
        optimizer="adafactor",
    ),
    "16g": Preset(
        name="16G (RTX 4060Ti 16GB / 4080)",
        vram_gb=16, blocks_to_swap=20, network_dim=32, network_alpha=16,
        # alpha=dim/2 [2026-06-09]: gives effective LoRA scale 0.5, so
        # inference lora_strength ≈ 1.0 lands at the trained behavior.
        learning_rate=2e-4,  # kohya canonical for Wan; was 1e-4 (too conservative)
        max_target_frames=33,  # 16GB: 33f stable
    ),
    "24g": Preset(
        name="24G (RTX 3090 / 4090) — recommended default",
        # [2026-06-10] blocks_to_swap 30 — MEASURED ladder on 4090 at
        # 832x480x33f training: bs=10 >79 s/step, bs=20 188 s/step with
        # 2.67 GB spilled into WDDM shared memory (Windows silently
        # overflows VRAM to PCIe-speed system RAM instead of OOMing),
        # bs=30 = 18.35 s/step with spill off the hot path. The lesson:
        # on Windows, leave 3-4 GB dedicated-VRAM headroom for the
        # desktop or the driver spills and steps run at bus speed.
        vram_gb=24, blocks_to_swap=30, network_dim=32, network_alpha=16,
        # [2026-06-09] alpha=dim/2 (was 32) lets inference strength ≈ 1.0
        # land at trained behavior without doubling the merge weight.
        # lr=2e-4 (was 1e-4): kohya canonical for Wan2.2 + adamw8bit;
        # community report #565 confirmed 1e-4 underlearned character ID.
        learning_rate=2e-4,
        # 24GB inference frontier at p720_81f only with bf16-norm + auto-swap
        # stack (CURRENT_BEST_2026-06-06). Training adds grad+optimizer state;
        # 49f is the safe default. 81f is opt-in with may-OOM warning.
        max_target_frames=49,
        # --force_v2_1_time_embedding shaves training VRAM. MUST match at
        # inference — Wan2.2-Lightning 250928 LoRA was NOT trained with
        # this flag, so user LoRAs that use it can't stack with Lightning
        # cleanly. Omitting by default; expose as advanced opt-in if needed.
    ),
    "40g_plus": Preset(
        name="40G+ (A100 40GB / 80GB / H100)",
        vram_gb=40, blocks_to_swap=0, network_dim=64, network_alpha=32,
        # alpha=dim/2 (was 64); see 24g comment
        learning_rate=2e-4,  # was 1e-4
        batch_size=2,
        max_target_frames=81,  # 40GB+: full 81f comfortable
        # [2026-06-10] --offload_inactive_dit REMOVED: musubi issue #595 +
        # docs/wan.md — needs ~42GB shared VRAM / ~96GB RAM on Windows and
        # is "almost always slower than running two separate trainings".
        # Dual-expert = two sequential runs with timestep windowing.
    ),
}


# ── Canonical VRAM tier ladder ──────────────────────────────────────────────
# ALL four tier selectors (training, MOVA, inference, S2V) pick from ONE shared
# breakpoint ladder: 40 / 22 / 14 / 10 GiB (>=22 leaves system headroom). They
# differ ONLY in the per-mode payload and the low-end FLOOR, for measured reasons:
# S2V is ~2x heavier (12GB floor); MOVA floors at its mova_12g preset; training/
# inference run down to 8-10GB. Each selector is a (min_vram, payload) tuple
# iterated high->low, so the ladder is visible data and the per-mode caps explicit
# (one consistent idiom, not four hand-rolled if-chains). Behavior is golden-tested
# in tools/_smoke_tiers.py against the pre-refactor outputs.
_TRAIN_TIERS = (
    (40, "40g_plus"), (22, "24g"), (14, "16g"), (10, "12g"), (0, "8g"),
)


def auto_select(vram_gb: int) -> Preset:
    """Auto-pick a training preset by VRAM (canonical ladder; see _TRAIN_TIERS)."""
    for min_v, key in _TRAIN_TIERS:
        if vram_gb >= min_v:
            return PRESETS[key]
    return PRESETS["8g"]


# ============================================================================
# MOVA (joint audio-video) LoRA training presets — the A/V side of the gym.
# Parallel to the Wan Preset above but for MOVA's OWN trainer (not musubi): the
# video tower is kept GPU-RESIDENT as 4-bit NF4 (bitsandbytes) instead of musubi
# block-swap, so there's no blocks_to_swap knob. Measured on a 4090 (memory
# mova-levers-synthesis-2026-06-15): 240p/81f NF4 = ~17.7s/step, ~17GB VRAM,
# ~6GB host RAM, trains joint A/V (M6 PASS). base="nf4" is the consumer default
# (low host RAM); base="fp8" falls back to MOVA's CPU-offload (needs ~46GB RAM).
# ============================================================================
@dataclass
class MovaPreset:
    name: str
    vram_gb: int
    base: str = "nf4"             # "nf4" resident (low RAM) | "fp8" offload (big RAM)
    rank: int = 32                # LoRA rank. v3 (2026-06-19): rank 16 underfit STYLE; 32 + FFN closed it.
    alpha: int = 16               # LoRA alpha (0 -> == rank). Style convention alpha=rank/2 (measured v3).
    learning_rate: float = 3e-4   # measured stable w/ grad-clip 1.0 (M6)
    max_target_frames: int = 81   # 81f@24fps = 3.375s; the comfortable 240p train len
    optimizer: str = "adamw8bit"
    max_train_epochs: int = 16
    save_every_n_epochs: int = 2
    grad_clip: float = 1.0        # REQUIRED — lr diverges without it (M6)
    lora_ffn: bool = True         # v3: LoRA the video DiT FFN linears too (attention-only underfits style).
    # v3 measured (2026-06-19): FREEZE the audio tower once audio is solved (it overtrains while video
    # still learns; AV-sync lives in the trainable cross-attn bridge, so audio is preserved AND the
    # rank/FFN capacity is redirected to video style). Whisper CER held 0.056/0.10 vs unfrozen.
    train_audio_tower: bool = False


MOVA_PRESETS: dict[str, MovaPreset] = {
    "mova_12g": MovaPreset(
        name="MOVA 12G — minimal (240p, short clips, NF4)",
        # smallest tier: keep rank low + FFN off to fit 12GB (extrapolated, not measured).
        vram_gb=12, base="nf4", rank=16, alpha=8, lora_ffn=False, max_target_frames=33),
    "mova_16g": MovaPreset(
        name="MOVA 16G (240p, NF4)",
        vram_gb=16, base="nf4", rank=16, alpha=8, max_target_frames=49),
    "mova_24g": MovaPreset(
        name="MOVA 24G (RTX 3090/4090) — recommended (240p, NF4-resident)",
        # MEASURED v3 recipe (2026-06-19): 240p/81f NF4, rank32/alpha16, FFN LoRA, audio frozen ->
        # peak 16.6GB, coherent A/V (Whisper CER 0.056/0.10), claymation style transfers. Fits a
        # normal 24GB-GPU + 32GB-RAM box (where fp8-offload's 46GB-RAM fails).
        vram_gb=24, base="nf4", rank=32, alpha=16, max_target_frames=81),
    "mova_40g_plus": MovaPreset(
        name="MOVA 40G+ (A100/H100) — bigger rank, more frames",
        vram_gb=40, base="nf4", rank=64, alpha=32, max_target_frames=129),
}


# MOVA floors at mova_12g (A/V is heavy; no 8g/10g tier). Same ladder, mode floor.
_MOVA_TIERS = (
    (40, "mova_40g_plus"), (22, "mova_24g"), (14, "mova_16g"), (0, "mova_12g"),
)


def auto_select_mova(vram_gb: int) -> MovaPreset:
    """Auto-pick a MOVA A/V preset by VRAM (canonical ladder; see _MOVA_TIERS)."""
    for min_v, key in _MOVA_TIERS:
        if vram_gb >= min_v:
            return MOVA_PRESETS[key]
    return MOVA_PRESETS["mova_12g"]


def detect_vram_gb() -> int:
    """Detect primary GPU VRAM via torch.cuda.mem_get_info."""
    import torch
    if not torch.cuda.is_available():
        return 0
    _, total = torch.cuda.mem_get_info(0)
    return int(total / (1024**3))


@dataclass
class GpuCapability:
    """Detected GPU compute properties used by inference-scheme recommender.

    `name`           : torch.cuda.get_device_name()
    `vram_gb`        : total VRAM in GiB
    `compute_major`  : sm_major (e.g. 8 for Ampere, 8 for Ada under sm_89)
    `compute_minor`  : sm_minor (e.g. 9 for Ada sm_89, 6 for Ampere sm_86)
    `has_fp8_cores`  : True if Ada sm_89+ or Hopper sm_90+ (hardware FP8 tensor cores)
    `is_ada`         : True for sm_89 (matters for our P34 stack's FP8 fast path)
    `is_amd_rocm`    : True if torch.version.hip — needs the bf16 fallback path
    `is_apple_mps`   : True if MPS device — needs the GGUF Q4 community route
    """
    name: str
    vram_gb: int
    compute_major: int
    compute_minor: int
    has_fp8_cores: bool
    is_ada: bool
    is_amd_rocm: bool
    is_apple_mps: bool


def detect_capability() -> GpuCapability:
    """Probe the active CUDA / ROCm / MPS device."""
    import torch
    if hasattr(torch, "version") and getattr(torch.version, "hip", None):
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "AMD GPU (ROCm)"
        _, total = (torch.cuda.mem_get_info(0)
                    if torch.cuda.is_available() else (0, 0))
        return GpuCapability(
            name=name, vram_gb=int(total / (1024**3)),
            compute_major=0, compute_minor=0,
            has_fp8_cores=False, is_ada=False,
            is_amd_rocm=True, is_apple_mps=False)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS does not report VRAM via the standard API. Assume Apple Silicon
        # unified memory, conservatively treat as 16GB-tier (M1 Pro baseline).
        return GpuCapability(
            name="Apple Silicon (MPS)", vram_gb=16,
            compute_major=0, compute_minor=0,
            has_fp8_cores=False, is_ada=False,
            is_amd_rocm=False, is_apple_mps=True)
    if not torch.cuda.is_available():
        return GpuCapability(
            name="CPU only", vram_gb=0,
            compute_major=0, compute_minor=0,
            has_fp8_cores=False, is_ada=False,
            is_amd_rocm=False, is_apple_mps=False)
    cap = torch.cuda.get_device_capability(0)
    _, total = torch.cuda.mem_get_info(0)
    is_ada = cap == (8, 9)
    has_fp8 = cap >= (8, 9)  # Ada sm_89 and Hopper sm_90+ have FP8 tensor cores
    return GpuCapability(
        name=torch.cuda.get_device_name(0),
        vram_gb=int(total / (1024**3)),
        compute_major=cap[0], compute_minor=cap[1],
        has_fp8_cores=has_fp8, is_ada=is_ada,
        is_amd_rocm=False, is_apple_mps=False)


# ----------------------------------------------------------------------------
# Inference-scheme recommender. Mirrors the matrix in
# `Faster-Wan2.2/docs/lower_vram_support.md` so the preview path uses the
# same scheme/auto-tune the inference fork ships:
#
#   FP8 (torchao)     — Ada sm_89+ only; fastest on RTX 4090/4080/4070-Ti
#                       (and 5090 once validated)
#   Int8 (torchao)    — Ampere sm_86 (3090/3080) — FP8 falls back to slow
#                       software path on these cards; Int8 weight-only is
#                       hardware-supported via INT8 tensor cores
#   GGUF Q4_K_M       — Ampere sm_75/Turing/AMD ROCm/Apple MPS — lowest
#                       VRAM tier; preview falls back here when no FP8/INT8
#                       hardware path applies
#   BF16              — used as default by the inference fork when the
#                       prequant scheme can't be determined
# ----------------------------------------------------------------------------
INFERENCE_SCHEMES = ("fp8", "int8", "gguf_q4", "bf16")


def recommend_inference_scheme(cap: GpuCapability) -> str:
    """Pick a scheme for the preview-inference path based on the GPU.

    Always returns one of `INFERENCE_SCHEMES`. The decision is a function
    of compute capability AND VRAM headroom — an Ampere 24GB still wants
    Int8 (FP8 has no hardware path), but an Ada 8GB also wants Int8 or
    GGUF (FP8 fits the weights but transients OOM)."""
    if cap.is_amd_rocm or cap.is_apple_mps:
        return "gguf_q4"
    if cap.has_fp8_cores:
        # Ada/Hopper with FP8 hardware. At 8GB we ship FP8+N=0 (per the
        # measured 6.55GB floor at p480_17f); under that we'd need Int8.
        if cap.vram_gb >= 8:
            return "fp8"
        return "int8"
    # Ampere / Turing / older Pascal. No FP8 tensor cores.
    if cap.compute_major == 8 and cap.compute_minor >= 0:
        # Ampere 3090/3080 etc — Int8 has full tensor-core support
        return "int8"
    # Turing sm_75, older — GGUF Q4 is the safest fallback
    return "gguf_q4"


def auto_block_swap_for_preview(cap: GpuCapability) -> str:
    """Value to pass as BASIWAN_BLOCK_SWAP_N. We default to 'auto' so
    the inference fork's run_one_video.py picks N at pipe-build time using
    measured peak_alloc — that's the shape-aware tuning shipped in
    docs/lower_vram_support.md. Set this explicitly here so the preview
    path opts in. (The runner respects an explicit int override too.)"""
    return "auto"


def detect_host_ram_gb() -> int:
    """Total host RAM in GiB. psutil if available, else /proc/meminfo, else 0.
    0 means 'unknown' — callers must NOT gate on RAM when it's 0 (don't punish a
    probe failure)."""
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024 ** 3))
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(int(line.split()[1]) / (1024 ** 2))  # kB -> GiB
    except Exception:
        pass
    return 0


# [#392] Inference VRAM tiers. The 24GB row is MEASURED (champion p720_17f at
# block-swap N=2 + the 363s p720_81f run); the lower rows are CONSERVATIVE
# extrapolations meant to be tightened by #394 (simulated-VRAM OOM walls) — they
# gate FIT, not speed. block_swap_n is for the ~40-block/expert Wan2.2-A14B
# (higher N = more CPU offload = fits a smaller card, slower). vae='full' needs
# ~22GB headroom (16.3GB measured peak at p720_17f); below that, TAEHV tiny-VAE.
# Thresholds are GiB-FLOORS, not marketing GB: a 24GB card reports total ~=
# 23.99 GiB -> int()=23, a 16GB card ~=15, a 12GB card ~=11 (detect_capability
# does int(total/1024^3)). So 24GB->22, 16GB->14, 12GB->10 — otherwise every
# real 24GB 4090 would mis-tier DOWN (caught by _smoke_tier). 22 also matches the
# runner's existing >=22 full-VAE gate.
#
# block_swap_n = blocks kept GPU-RESIDENT (run_one_video_gguf.py:998: i<n -> GPU,
# i>=n -> CPU). HIGHER N = MORE VRAM, not less. So smaller cards need LOW N (more
# swap). N=2 is the measured champion (heavy swap; fits the huge MoE experts even
# on 24GB) and is the safe universal floor — bigger cards COULD raise it for speed,
# but N=2 is the most-likely-to-fit. max_px caps RESOLUTION per tier (the real
# limiter on small cards): #394 measured 12GB OOMs at p720_17f, so 12/8GB cap to
# p480. Measured cells (sim-VRAM FIT gate, _tier_validate): 24/N2/p720 FIT,
# 16/N2/p720 FIT (direct re-measure 2026-06-13; also fit at N=10). 12/N2/p720 OOM
# AND 12/N2/p480 OOM (~11.5GB GGUF-Q4 GPU floor at N2, right at the 12GB edge:
# MARGINAL — a real card may survive via WDDM spill / expandable_segments). 8/N2/
# p480 OOM (floor >> 8GB: GGUF-Q4 NOT viable on 8GB; needs FP8/Int8, out of scope).
# So 12GB is best-effort@p480 and 8GB OOMs gracefully. Sim hard-cap = conservative
# FIT gate, not a real-card guarantee. — see memory.
_INFERENCE_TIERS = (
    # min_vram(GiB floor), block_swap_n, vae,     max_frames, max_px      (card)
    (22,                   2,            "full",  81,         1280 * 720),  # 24GB
    (14,                   2,            "taehv", 49,         1280 * 720),  # 16GB
    (10,                   2,            "taehv", 33,         832 * 480),   # 12GB
    (0,                    2,            "taehv", 17,         832 * 480),   # 8GB-
)
RAM_FOR_PERSISTENT_WORKER_GB = 32  # eager prewarm holds the ~19GB pack in page cache


def inference_tier(vram_gb: int, ram_gb: int = 0) -> dict:
    """Pick the inference recipe for a card. Pure: (vram_gb, ram_gb) ->
    {block_swap_n, vae, max_frames, persistent_worker, tier}.

    The persistent worker + eager prewarm keep the ~19GB GGUF pack resident in
    page cache; on hosts with < 32 GB RAM that thrashes (the Wan2GP lesson + the
    Continue-hang #380), so persistent_worker is False there. ram_gb==0 means
    'unknown' -> leave the worker ON (don't punish a probe failure)."""
    worker = not (0 < ram_gb < RAM_FOR_PERSISTENT_WORKER_GB)
    for min_v, n, vae, max_f, max_px in _INFERENCE_TIERS:
        if vram_gb >= min_v:
            return {"block_swap_n": n, "vae": vae, "max_frames": max_f,
                    "max_px": max_px, "persistent_worker": worker,
                    "tier": f"{min_v}GB"}
    return {"block_swap_n": 2, "vae": "taehv", "max_frames": 17,
            "max_px": 832 * 480, "persistent_worker": False, "tier": "unknown"}


# [S10] Wan2.2-S2V (audio-driven talking character) is ~2x heavier than T2V: it carries
# an audio_injector (25-layer wav2vec2 cross-attn into 12 DiT blocks) + 73 motion frames
# per chunk on top of the base 40-block DiT. The T2V _INFERENCE_TIERS therefore do NOT
# apply -- they'd tell a 16GB user "720p ok" and brick them. MEASURED on a 24GB 4090:
# 480p widescreen (832x480) is reliable at the full per-chunk window (~13GB headroom);
# 720p sits AT the 24GB wall regardless of window (OOMs/thrashes even with the residency
# cap + per-chunk empty_cache + forced bf16 norms) and needs fp8/SageAttention to ship.
# Resident GGUF-Q4 S2V floor is ~9.3GB, so <12GB cards can't hold it. Smaller cards are
# UNMEASURED here (only a 24GB card was available) so their caps are deliberately
# CONSERVATIVE -- the runtime residency-aware per-chunk window cap (run_one_video_gguf
# _do_one_generation_s2v) self-limits further from ACTUAL free VRAM, so these are an
# upper bound the UI enforces, not a fit guarantee.
_S2V_TIERS = (
    # min_vram(GiB floor), max_px,    max_frames(per-chunk window), available, note
    (22, 832 * 480, 80, True,
     "Talking-character runs at 480p (720p needs fp8 -- coming later)."),       # 24GB
    (14, 832 * 480, 48, True,
     "Talking-character: 480p only on a 16GB card."),                           # 16GB
    (10, 832 * 480, 24, True,
     "Talking-character: 480p short clips only on a 12GB card (marginal)."),    # 12GB
    (0,  0,         0,  False,
     "Talking-character (S2V) needs a 12GB+ NVIDIA card."),                     # <12GB
)


def s2v_caps(vram_gb: int) -> dict:
    """S2V capability for a card. Pure: vram_gb -> {available, max_px, max_frames,
    note, tier}. The UI gates the talking-character mode on this so a too-small card
    can't pick a config that bricks (the runtime cap is the second line of defense)."""
    for min_v, max_px, max_f, avail, note in _S2V_TIERS:
        if vram_gb >= min_v:
            return {"available": avail, "max_px": max_px, "max_frames": max_f,
                    "note": note, "tier": f"{min_v}GB"}
    return {"available": False, "max_px": 0, "max_frames": 0,
            "note": "Talking-character (S2V) needs a 12GB+ NVIDIA card.",
            "tier": "unknown"}
