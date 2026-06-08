"""LoRA format converters for Wan2.2 (musubi-tuner → diffusers / sd-scripts).

musubi-tuner writes LoRAs with the `lora_unet_*` key prefix (sd-scripts
convention). This module does PREFIX-LEVEL conversion, not deep key remapping.

What this handles correctly:
- **sd_scripts / musubi** — passthrough. musubi IS sd-scripts format.
- **diffusers / peft** — prefix swap `lora_unet_` → `transformer.`. peft's
  loader resolves the underscore-separated module path against the actual
  model tree, so this is sufficient.

What this does NOT handle (use a target-native loader instead):
- **ComfyUI (kijai's WanVideoWrapper)** — accepts the raw `lora_unet_*`
  format directly (his loader has a built-in `_` → `.` remapper that knows
  which underscores are hierarchy separators vs part of parameter names
  like `to_q`). Pass the unconverted musubi output to ComfyUI, not the
  output of this script. We expose 'comfyui' as a format here but it just
  copies the file unchanged with a warning.
- **Faster-Wan2.2 inference path** — also handles `lora_unet_*` natively
  via `wan/text2video.py` (BASIWAN_USER_LORA). No conversion needed.

Reference: kohya-ss/sd-scripts loader replaces `.` with `_` in module paths
at save time. Round-tripping requires knowledge of the model tree to know
which underscores were originally dots — peft handles this for diffusers,
ComfyUI-WanVideoWrapper handles it for ComfyUI, and Faster-Wan2.2 handles
it for our own inference. So this script's job is just the prefix swap.

Usage (CLI):
    python -m basi.export --in workspace/my_lora.safetensors \\
        --format diffusers --out my_lora_diffusers.safetensors
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import safetensors.torch as st


# The musubi-tuner output prefix. All key conversions translate FROM this.
MUSUBI_PREFIX = "lora_unet_"

# Target format → (new prefix or None, note).
# None means passthrough; comfyui maps to None with a note that the wrapper
# handles the format natively.
FORMATS = {
    "musubi":     (None, "passthrough (musubi-tuner native)"),
    "sd_scripts": (None, "passthrough (same as musubi)"),
    "diffusers":  ("transformer.", "peft loader resolves module path"),
    "peft":       ("transformer.", "same as diffusers"),
    "comfyui":    (None, "passthrough — kijai's WanVideoWrapper handles musubi format natively"),
}


def _convert_state_dict(sd: dict, target: str) -> dict:
    if target not in FORMATS:
        raise ValueError(f"unknown format: {target}. Choices: {list(FORMATS)}")
    new_prefix, _note = FORMATS[target]
    if new_prefix is None:
        return sd
    out = {}
    for k, v in sd.items():
        if k.startswith(MUSUBI_PREFIX):
            new_key = new_prefix + k[len(MUSUBI_PREFIX):]
        else:
            new_key = k
        out[new_key] = v
    return out


def convert(src: Path, target_format: str, dst: Path) -> dict:
    """Read src LoRA, convert to target_format, write to dst.

    Returns a small dict summary (key count, format, etc.) for logging.
    """
    sd = st.load_file(str(src))
    out_sd = _convert_state_dict(sd, target_format)
    st.save_file(out_sd, str(dst))
    _new_prefix, note = FORMATS[target_format]
    return {
        "src": str(src),
        "dst": str(dst),
        "format": target_format,
        "n_keys": len(out_sd),
        "sample_key": next(iter(out_sd)) if out_sd else None,
        "note": note,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert musubi-tuner Wan2.2 LoRA to a target format.")
    p.add_argument("--in", dest="src", required=True, type=Path,
                   help="source .safetensors (musubi-tuner output)")
    p.add_argument("--format", choices=list(FORMATS), required=True,
                   help="target format")
    p.add_argument("--out", dest="dst", required=True, type=Path,
                   help="destination .safetensors path")
    args = p.parse_args(argv)
    if not args.src.exists():
        print(f"error: input not found: {args.src}", file=sys.stderr)
        return 1
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    summary = convert(args.src, args.format, args.dst)
    print(f"converted: {summary['src']}")
    print(f"      -->: {summary['dst']}")
    print(f"   format: {summary['format']}  ({summary['n_keys']} keys)")
    print(f"     note: {summary['note']}")
    if summary['sample_key']:
        print(f"  sample : {summary['sample_key']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
