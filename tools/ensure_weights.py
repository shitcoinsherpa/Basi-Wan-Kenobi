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
  mova       -- MOVA A/V training base (~78GB).  Usage:
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
    # download finished). Repo id verified (HuggingFace OpenMOSS-Team/MOVA-360p). ----
    {"name": "MOVA-360p joint A/V base", "group": "mova", "kind": "snapshot",
     "repo": "OpenMOSS-Team/MOVA-360p", "dest": "MOVA-360p",
     "ignore_patterns": ["*.mp4", "*.png", "assets/*"],
     "present": "MOVA-360p/video_dit/diffusion_pytorch_model-00003-of-00003.safetensors"},
    # ---- mova T2AV style-reference maker: SDXL base for the SDXL+IP-Adapter reference image that
    # generalizes MOVA to any prompt (prompt + a training-clip style frame -> styled scene still ->
    # MOVA I2V ref). Single-file checkpoint (~6.9GB, openrail++ = commercial-OK), loaded by
    # tools/make_style_reference.py via from_single_file. The IP-Adapter weights themselves are
    # prefetched into the HF cache by install_mova.js (diffusers load_ip_adapter reads the HF cache).
    # NOTE: stabilityai/stable-diffusion-xl-base-1.0 is a click-through-license repo; if a token-less
    # fetch is refused at install, the log surfaces it (a mirror can be set via BASIWAN_SDXL_REPO). ----
    {"name": "SDXL base (T2AV style reference)", "group": "mova", "kind": "file",
     "repo": os.environ.get("BASIWAN_SDXL_REPO", "stabilityai/stable-diffusion-xl-base-1.0"),
     "filename": "sd_xl_base_1.0.safetensors", "dest": ".",
     "present": "sd_xl_base_1.0.safetensors",
     # CROSS-APP REUSE: SDXL base is the ONE standard model basiwan shares with the rest of the
     # Pinokio image-gen ecosystem. If the user already has it from comfy/forge/fooocus/a1111,
     # hardlink their copy (free, same drive) instead of re-downloading 6.9GB. These globs are the
     # canonical filename under each app's model dir (read-only; we never write into peer apps).
     "peer_globs": ["app/models/checkpoints/sd_xl_base_1.0.safetensors",
                    "models/checkpoints/sd_xl_base_1.0.safetensors",
                    "app/models/Stable-diffusion/sd_xl_base_1.0.safetensors",
                    "models/Stable-diffusion/sd_xl_base_1.0.safetensors"]},
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


def _api_peer_roots():
    """Sibling Pinokio app dirs (PINOKIO_HOME/api/*), excluding our own repo. Empty list if
    we're not installed under an `api/` parent (dev tree) -- then only BASIWAN_SHARED_DIR is used."""
    parent = _REPO_ROOT.parent
    if parent.name != "api":
        return []
    out = []
    try:
        for d in sorted(parent.iterdir()):
            if d.is_dir() and d.resolve() != _REPO_ROOT:
                out.append(d)
    except OSError:
        pass
    return out


def _find_in_peers(filename, peer_globs):
    """Find an existing copy of `filename` to reuse, WITHOUT downloading. Searches (1) an explicit
    user shared root (BASIWAN_SHARED_DIR, recursive) then (2) each peer Pinokio app under api/ at
    the given peer_globs. Read-only: never writes into a peer. Returns the source Path or None."""
    shared = os.environ.get("BASIWAN_SHARED_DIR")
    if shared:
        sp = Path(shared)
        direct = sp / filename
        if direct.is_file():
            return direct
        try:
            hits = list(sp.glob("**/" + filename))
            if hits:
                return hits[0]
        except OSError:
            pass
    for root in _api_peer_roots():
        for rel in peer_globs:
            try:
                for hit in root.glob(rel):
                    if hit.is_file():
                        return hit
            except OSError:
                continue
    return None


def _link_or_copy(src, dst):
    """Point `dst` at the existing `src` with NO extra disk: hardlink first (works for files on the
    same volume with no admin -- the common case, all apps share one PINOKIO_HOME drive), else
    symlink (cross-volume; needs dev-mode/admin on Windows). Returns True iff a link was made; on
    failure returns False so the caller falls back to a normal download (we never copy -- a copy
    would defeat the whole point)."""
    dst = Path(dst)
    if dst.exists() or dst.is_symlink():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dst))
        return True
    except OSError:
        pass
    try:
        os.symlink(str(src), str(dst))
        return True
    except OSError:
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
        # CROSS-APP REUSE: before any download, try to link an existing copy from a peer Pinokio
        # app (or BASIWAN_SHARED_DIR). Only for flat-file artifacts that declare peer_globs (SDXL).
        if s.get("peer_globs") and s.get("kind") == "file":
            fname = s["filename"]
            target = (CKPT / fname) if s["dest"] == "." else (CKPT / s["dest"] / fname)
            src = _find_in_peers(fname, s["peer_globs"])
            if src is not None and _link_or_copy(src, target):
                log(f"  LINKED existing copy (no download): {src} -> {target}")
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
