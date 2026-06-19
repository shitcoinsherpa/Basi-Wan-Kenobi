"""Run the canonical 5-prompt ship verification.

Reuses the canonical recipe env vars + LoRA + args. Each run produces a .pt
and a .mp4 in outputs/_test/multi_<idx>_<shortname>.{pt,json,mp4}.

Wall-time and any failures are logged to outputs/_test/multi_summary.json.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "tools" / "run_one_video_gguf.py"
CONVERTER = ROOT / "tools" / "pt_to_mp4.py"
PYTHON = str(ROOT / "env" / "Scripts" / "python.exe") if os.name == "nt" else str(ROOT / "env" / "bin" / "python")

# PORTABLE checkpoint resolution: BASIWAN_CKPT_DIR env (canonical, same as
# app.py / basi/preview.py), else <repo_root>/checkpoints. GGUF snapshot dir is
# globbed (the HF cache hash differs per machine) instead of hardcoded.
import glob as _glob
_CKPT = Path(os.environ.get("BASIWAN_CKPT_DIR", str(ROOT / "checkpoints")))
_gguf_hi = sorted(_glob.glob(str(_CKPT / "gguf" / "**" / "HighNoise" /
                             "Wan2.2-T2V-A14B-HighNoise-Q4_K_M.gguf"), recursive=True))
_gguf_lo = sorted(_glob.glob(str(_CKPT / "gguf" / "**" / "LowNoise" /
                             "Wan2.2-T2V-A14B-LowNoise-Q4_K_M.gguf"), recursive=True))
if not _gguf_hi or not _gguf_lo:
    sys.exit(f"[bench] T2V GGUF pair not found under {_CKPT / 'gguf'}; set "
             f"BASIWAN_CKPT_DIR or download the QuantStack T2V-A14B GGUF.")
GGUF_HI = Path(_gguf_hi[0])
GGUF_LO = Path(_gguf_lo[0])
VAE = _CKPT / "Wan2.2-T2V-A14B" / "Wan2.1_VAE.pth"
BASE_DIR = _CKPT / "Wan2.2-T2V-A14B"
LORA_DIR = _CKPT / "Wan2.2-Lightning" / "Wan2.2-T2V-A14B-4steps-lora-250928"

PROMPTS = [
    ("cat_boxing",
     "Two anthropomorphic cats in comfy boxing gear and bright gloves "
     "fight intensely on a spotlighted stage."),
    ("cat_surf",
     "Summer beach vacation style, a white cat wearing sunglasses sits on a "
     "surfboard. The fluffy-furred feline gazes directly at the camera with a "
     "relaxed expression. Blurred beach scenery forms the background featuring "
     "crystal-clear waters, distant green hills, and a blue sky dotted with "
     "white clouds. A close-up shot highlights the feline's intricate details "
     "and the refreshing atmosphere of the seaside."),
    ("nyc_chase",
     "Cinematic NYC alley chase: The camera starts shoulder-height behind a "
     "hooded man steadily tracking forward as he weaves through crowds. Cold "
     "tones, high contrast, neon lights. Smooth glide with intense shake for "
     "immersive pursuer tension. Blurred steam and wet pavement. Lens flare, "
     "shallow depth of field."),
    ("dappled_pan",
     "A low angle shot of a young man in dappled sunlight. Backlighting, warm "
     "low-saturation tones. Slow-motion glide with handheld tremor for dreamy "
     "nostalgia. Blurred foliage for emotional focus. Camera pans left to low "
     "angle shot of a cute girl."),
    ("ironman_dolly",
     "In the style of an American drama promotional poster, Iron Man sits in "
     "a sleek, futuristic metal chair inside a dimly lit industrial setting. "
     "He is fully suited in his iconic red and gold armor, the arc reactor "
     "glowing in his chest. Around him are scattered high-tech gadgets, and "
     "stacks of prototype schematics. He sits still, helmet off, revealing "
     "Tony Stark's face confident, composed, with a subtle smirk. Camera "
     "dollies out. The background shows an abandoned, dim factory with light "
     "filtering through the windows."),
]


def run_one(idx, name, prompt, out_dir):
    out_pt = out_dir / f"multi_{idx}_{name}.pt"
    out_json = out_dir / f"multi_{idx}_{name}.json"
    out_mp4 = out_dir / f"multi_{idx}_{name}.mp4"
    log = out_dir / f"multi_{idx}_{name}.log"

    env = os.environ.copy()
    env.update({
        "BASIWAN_V2": "1",
        "BASIWAN_USE_PACK_CACHE": "1",
        "BASIWAN_NO_POOL": "1",
        # Pack-cache dir: honor an explicit override, else let the runner use its
        # canonical default (~/.cache/marlin_packs) — no hardcoded D:/ path.
        **({"BASIWAN_PACK_CACHE_DIR": os.environ["BASIWAN_PACK_CACHE_DIR"]}
           if os.environ.get("BASIWAN_PACK_CACHE_DIR") else {}),
        "BASIWAN_VAE_BF16": "1",
        "BASIWAN_RMS_BF16": "0",
        "BASIWAN_LN_BF16": "0",
        # Sync swap: BASIWAN_BLOCK_SWAP_ASYNC NOT set on Windows.
    })

    print(f"\n=== [{idx}/{len(PROMPTS)}] {name} ===", flush=True)
    cmd = [
        PYTHON, "-u", str(RUNNER),
        "--gguf-high", str(GGUF_HI),
        "--gguf-low",  str(GGUF_LO),
        "--vae", str(VAE),
        "--base-dir", str(BASE_DIR),
        "--lora-dir", str(LORA_DIR),
        "--lora-mode", "forward",
        "--prompt", prompt,
        "--width", "1280", "--height", "720", "--frames", "17",
        "--block-swap-n", "2", "--ffn-chunk-size", "4096",
        "--out", str(out_pt), "--meta", str(out_json),
    ]
    t0 = time.time()
    with log.open("w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(ROOT))
    elapsed = time.time() - t0

    rec = {"idx": idx, "name": name, "prompt_first40": prompt[:40] + "...",
           "elapsed_total_s": round(elapsed, 1),
           "exit_code": proc.returncode, "log": str(log)}
    if proc.returncode == 0 and out_json.exists():
        meta = json.loads(out_json.read_text())
        rec["wall_s"] = round(meta.get("wall_s", -1), 2)
        try:
            subprocess.run(
                [PYTHON, "-u", str(CONVERTER), str(out_pt), str(out_mp4)],
                capture_output=True, text=True, cwd=str(ROOT), timeout=60,
            )
            rec["mp4"] = str(out_mp4) if out_mp4.exists() else None
            rec["mp4_size"] = out_mp4.stat().st_size if out_mp4.exists() else 0
        except Exception as e:
            rec["mp4_err"] = str(e)
    print(json.dumps(rec, indent=2), flush=True)
    return rec


def main():
    out_dir = ROOT / "outputs" / "_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for idx, (name, prompt) in enumerate(PROMPTS, 1):
        rec = run_one(idx, name, prompt, out_dir)
        summary.append(rec)
        (out_dir / "multi_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FINAL SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
