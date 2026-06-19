"""ensure_weights.py -- ONE place that guarantees every model weight BASIWAN needs
is present, downloading only what's missing. Idempotent + resumable (huggingface_hub
resumes partial blobs and no-ops when complete), visible byte progress, portable
(BASIWAN_CKPT_DIR or <repo>/checkpoints -- NO dev-box paths). Consolidates the old
per-mode fetchers (_fetch_vace_gguf.py, _s2v_fetch.py, prefetch_vlms.py) into a single
manifest so "is the weight there?" has one answer for the whole app.

GROUPS (download only what a mode needs):
  inference  -- Studio out-of-box: GGUF Q4 DiT pairs (T2V/I2V/VACE/S2V) + the base
                T5/VAE/tokenizer/config SUBSET (NOT the ~28GB bf16 DiT shards -- GGUF
                replaces them) + Lightning LoRAs + S2V encoder + depth model. ~40GB.
  wan-train  -- Gym (Wan LoRA): the FULL Wan2.2-T2V-A14B base incl. bf16 DiT (~50GB).
  mova       -- MOVA A/V training base (~78GB).  [C3 -- repo id verified at wire time]

Usage:
  python tools/ensure_weights.py                      # inference group (default)
  python tools/ensure_weights.py --groups inference,wan-train
  python tools/ensure_weights.py --check              # report presence only, no download
ASCII only.
"""
import argparse
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
CKPT = Path(os.environ.get("BASIWAN_CKPT_DIR", str(_REPO_ROOT / "checkpoints")))
GGUF = CKPT / "gguf"


def log(*a):
    print("[ensure-weights]", *a, flush=True)


# Each artifact: a dict describing how to fetch it + how to tell it's already present.
#   group   : which --groups bucket it belongs to
#   kind    : "file" (hf_hub_download) | "snapshot" (snapshot_download)
#   repo    : HF repo id
#   present : path/glob (relative to CKPT) that, if it matches an existing file,
#             means we can SKIP the network call entirely (fast idempotency)
#   ...     : kind-specific args (filename / allow_patterns / ignore_patterns / dest)
# dest "gguf"  -> cache_dir=GGUF (HF-cache layout the resolver globs for)
# dest "<rel>" -> local_dir=CKPT/<rel> (flat layout)
MANIFEST = [
    # ---- inference: GGUF Q4 DiT pairs (HF-cache layout under checkpoints/gguf) ----
    {"name": "T2V GGUF pair", "group": "inference", "kind": "snapshot",
     "repo": "QuantStack/Wan2.2-T2V-A14B-GGUF", "dest": "gguf",
     "allow_patterns": ["HighNoise/*Q4_K_M.gguf", "LowNoise/*Q4_K_M.gguf"],
     "present": ["gguf/models--QuantStack--Wan2.2-T2V-A14B-GGUF/snapshots/*/*Noise/*Q4_K_M.gguf",
                 "gguf/Wan2.2-T2V-A14B-GGUF/*Noise/*Q4_K_M.gguf"]},
    {"name": "I2V GGUF pair", "group": "inference", "kind": "snapshot",
     "repo": "QuantStack/Wan2.2-I2V-A14B-GGUF", "dest": "gguf",
     "allow_patterns": ["HighNoise/*Q4_K_M.gguf", "LowNoise/*Q4_K_M.gguf"],
     "present": ["gguf/models--QuantStack--Wan2.2-I2V-A14B-GGUF/snapshots/*/*Noise/*Q4_K_M.gguf",
                 "gguf/Wan2.2-I2V-A14B-GGUF/*Noise/*Q4_K_M.gguf"]},
    {"name": "VACE-Fun GGUF pair", "group": "inference", "kind": "snapshot",
     "repo": "QuantStack/Wan2.2-VACE-Fun-A14B-GGUF", "dest": "gguf",
     "allow_patterns": ["HighNoise/*Q4_K_M.gguf", "LowNoise/*Q4_K_M.gguf"],
     "present": ["gguf/models--QuantStack--Wan2.2-VACE-Fun-A14B-GGUF/snapshots/*/*Noise/*Q4_K_M.gguf",
                 "gguf/Wan2.2-VACE-Fun-A14B-GGUF/*Noise/*Q4_K_M.gguf"]},
    {"name": "S2V GGUF (single)", "group": "inference", "kind": "file",
     "repo": "QuantStack/Wan2.2-S2V-14B-GGUF", "filename": "Wan2.2-S2V-14B-Q4_K_M.gguf", "dest": "gguf",
     "present": ["gguf/models--QuantStack--Wan2.2-S2V-14B-GGUF/snapshots/*/Wan2.2-S2V-14B-Q4_K_M.gguf",
                 "gguf/Wan2.2-S2V-14B-GGUF/Wan2.2-S2V-14B-Q4_K_M.gguf",
                 "gguf/Wan2.2-S2V-14B-Q4_K_M.gguf"]},

    # ---- inference: base T5 + VAE + tokenizer + expert configs (NOT the DiT shards) ----
    {"name": "T2V base (T5+VAE+tokenizer+configs, no DiT shards)", "group": "inference",
     "kind": "snapshot", "repo": "Wan-AI/Wan2.2-T2V-A14B", "dest": "Wan2.2-T2V-A14B",
     "ignore_patterns": ["*.safetensors", "assets/*", "*.png", "*.mp4", "nohup.out"],
     "present": "Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth"},

    # ---- inference: Lightning 4-step LoRAs (only the shipped variant's subfolder) ----
    {"name": "Lightning T2V 4-step LoRA", "group": "inference", "kind": "snapshot",
     "repo": "lightx2v/Wan2.2-Lightning", "dest": "Wan2.2-Lightning",
     "allow_patterns": ["Wan2.2-T2V-A14B-4steps-lora-250928/*.safetensors"],
     "present": "Wan2.2-Lightning/Wan2.2-T2V-A14B-4steps-lora-250928/*.safetensors"},
    {"name": "Lightning I2V 4-step LoRA", "group": "inference", "kind": "snapshot",
     "repo": "lightx2v/Wan2.2-Lightning", "dest": "Wan2.2-Lightning",
     "allow_patterns": ["Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/*.safetensors"],
     "present": "Wan2.2-Lightning/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/*.safetensors"},

    # ---- inference: S2V config + wav2vec2 audio encoder (T5/VAE come from T2V base) ----
    {"name": "S2V config.json", "group": "inference", "kind": "file",
     "repo": "Wan-AI/Wan2.2-S2V-14B", "filename": "config.json", "dest": "Wan2.2-S2V-14B",
     "present": "Wan2.2-S2V-14B/config.json"},
    {"name": "wav2vec2 audio encoder", "group": "inference", "kind": "snapshot",
     "repo": "jonatasgrosman/wav2vec2-large-xlsr-53-english",
     "dest": "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english",
     "allow_patterns": ["model.safetensors", "config.json", "preprocessor_config.json",
                         "vocab.json", "special_tokens_map.json", "alphabet.json"],
     "present": "Wan2.2-S2V-14B/wav2vec2-large-xlsr-53-english/model.safetensors"},

    # ---- inference: depth model for VACE depth-lock restyle ----
    {"name": "Depth-Anything-V2-Small", "group": "inference", "kind": "snapshot",
     "repo": "depth-anything/Depth-Anything-V2-Small-hf", "dest": "cache_ckpt",
     "present": "models--depth-anything--Depth-Anything-V2-Small-hf/snapshots/*/*.safetensors"},

    # ---- wan-train: the FULL base incl. bf16 DiT shards (musubi LoRA training) ----
    {"name": "Wan2.2-T2V-A14B FULL base (training)", "group": "wan-train", "kind": "snapshot",
     "repo": "Wan-AI/Wan2.2-T2V-A14B", "dest": "Wan2.2-T2V-A14B",
     "ignore_patterns": ["assets/*", "*.png", "*.mp4", "nohup.out"],
     "present": "Wan2.2-T2V-A14B/high_noise_model/diffusion_pytorch_model-00006-of-00006.safetensors"},
    # ---- mova: the MOVA-360p joint A/V base (~77.7GB: two video experts + audio_dit + audio/video
    # VAEs + dual_tower_bridge + UMT5 + tokenizer + scheduler). Used by BOTH MOVA A/V LoRA training
    # (Gym) and MOVA A/V generation (Studio). Heavy + optional -> fetched on first MOVA use, never in
    # the default 'inference' set. Present-check = the last video_dit shard (only exists once the bulk
    # download finished). Repo id verified 2026-06-19 (HuggingFace OpenMOSS-Team/MOVA-360p). ----
    {"name": "MOVA-360p joint A/V base", "group": "mova", "kind": "snapshot",
     "repo": "OpenMOSS-Team/MOVA-360p", "dest": "MOVA-360p",
     "ignore_patterns": ["*.mp4", "*.png", "assets/*"],
     "present": "MOVA-360p/video_dit/diffusion_pytorch_model-00003-of-00003.safetensors"},
]


def _present(spec):
    """True if ANY of the artifact's present-globs matches a file under CKPT -> skip the
    fetch. `present` may be a str or list; we accept multiple globs because the GGUF
    resolver itself accepts multiple on-disk layouts (HF-cache models--*/snapshots OR a
    flat repo-short dir OR bare), so the idempotency check must match all of them too."""
    pats = spec.get("present")
    if not pats:
        return False
    if isinstance(pats, str):
        pats = [pats]
    for pat in pats:
        if list(CKPT.glob(pat)):
            return True
    return False


def ensure(groups, check_only=False):
    from huggingface_hub import hf_hub_download, snapshot_download
    CKPT.mkdir(parents=True, exist_ok=True)
    GGUF.mkdir(parents=True, exist_ok=True)
    todo = [s for s in MANIFEST if s["group"] in groups]
    log(f"groups={sorted(groups)} -> {len(todo)} artifact(s); ckpt={CKPT}")
    missing = 0
    for s in todo:
        here = _present(s)
        mark = "present" if here else "MISSING"
        log(f"[{mark}] {s['name']} ({s['repo']})")
        if here or check_only:
            missing += 0 if here else 1
            continue
        dest = s["dest"]
        kw = {}
        if dest == "gguf":
            kw["cache_dir"] = str(GGUF)
        elif dest == "cache_ckpt":
            kw["cache_dir"] = str(CKPT)   # HF-cache layout under checkpoints (depth loader)
        else:
            kw["local_dir"] = str(CKPT / dest)
        try:
            if s["kind"] == "file":
                p = hf_hub_download(repo_id=s["repo"], filename=s["filename"], **kw)
            else:
                for k in ("allow_patterns", "ignore_patterns"):
                    if k in s:
                        kw[k] = s[k]
                p = snapshot_download(repo_id=s["repo"], **kw)
            log(f"  -> {p}")
        except Exception as e:
            log(f"  FAILED {s['name']}: {type(e).__name__} {str(e)[:160]}")
            missing += 1
    if check_only:
        log(f"CHECK: {missing} missing of {len(todo)}")
    else:
        log("ENSURE_WEIGHTS_DONE" if missing == 0 else f"ENSURE_WEIGHTS_INCOMPLETE ({missing} failed)")
    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="inference",
                    help="comma-separated: inference, wan-train, mova (default: inference)")
    ap.add_argument("--check", action="store_true", help="report presence only, no download")
    a = ap.parse_args()
    g = {x.strip() for x in a.groups.split(",") if x.strip()}
    sys.exit(1 if ensure(g, check_only=a.check) and not a.check else 0)
