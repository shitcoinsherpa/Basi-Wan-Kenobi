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
        vram_gb=16, blocks_to_swap=20, network_dim=32, network_alpha=32,
        learning_rate=1e-4,
        max_target_frames=33,  # 16GB: 33f stable
    ),
    "24g": Preset(
        name="24G (RTX 3090 / 4090) — recommended default",
        vram_gb=24, blocks_to_swap=10, network_dim=32, network_alpha=32,
        learning_rate=1e-4,
        # 24GB inference frontier at p720_81f only with bf16-norm + auto-swap
        # stack (CURRENT_BEST_2026-06-06). Training adds grad+optimizer state;
        # 49f is the safe default. 81f is opt-in with may-OOM warning.
        max_target_frames=49,
    ),
    "40g_plus": Preset(
        name="40G+ (A100 40GB / 80GB / H100)",
        vram_gb=40, blocks_to_swap=0, network_dim=64, network_alpha=64,
        learning_rate=1e-4, batch_size=2,
        max_target_frames=81,  # 40GB+: full 81f comfortable
        # Dual-expert training viable here
        extra_args=["--offload_inactive_dit"],
    ),
}


def auto_select(vram_gb: int) -> Preset:
    """Auto-pick a preset based on detected VRAM."""
    if vram_gb >= 40:
        return PRESETS["40g_plus"]
    if vram_gb >= 22:  # leave headroom for system
        return PRESETS["24g"]
    if vram_gb >= 14:
        return PRESETS["16g"]
    if vram_gb >= 10:
        return PRESETS["12g"]
    return PRESETS["8g"]


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
