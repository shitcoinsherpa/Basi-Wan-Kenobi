"""BASIWAN — inline patch for musubi-tuner pyproject.toml.

Upstream musubi-tuner pins `transformers==4.56.1`. The Qwen3-VL captioner
requires transformers >= 4.57, so we loosen the constraint to
`>=4.57.0,<5`. Run from the BASIWAN app root.

Extracted from install.js inline `python -c "..."` because Pinokio's
Windows cmd.exe shell escaping made the inline form brittle (quoting bugs
hung the install task on 2026-06-08).
"""
from pathlib import Path
import sys

p = Path("ext/musubi-tuner/pyproject.toml")
if not p.exists():
    print(f"[basiwan-patch] not found: {p}")
    sys.exit(0)
t = p.read_text(encoding="utf-8")
old = '"transformers==4.56.1"'
new = '"transformers>=4.57.0,<5"'
if old in t:
    t = t.replace(old, new)
    p.write_text(t, encoding="utf-8")
    print("[basiwan-patch] loosened transformers pin: 4.56.1 -> >=4.57.0,<5")
else:
    print("[basiwan-patch] already patched (no 'transformers==4.56.1' found)")
