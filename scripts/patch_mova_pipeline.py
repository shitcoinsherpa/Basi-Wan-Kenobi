"""BASIWAN — make the vendored OpenMOSS/MOVA package import on Windows/macOS.

MOVA's inference pipeline runs on plain torch + SDPA, but three modules import the
Linux-only `yunchang` (multi-GPU sequence/context-parallel attention) UNCONDITIONALLY at
module top level, so the whole package fails to import on a box without yunchang — even
though single-GPU inference (cp_mesh=None) never uses it (AttnType is only referenced as a
default arg in replace_attention(), the multi-GPU path).

This patch guards those imports with try/except + a complete AttnType fallback, exactly
mirroring the repo's own (incomplete) guard in wan_video_dit.py. It is IDEMPOTENT and
TOLERANT (missing file / already-patched -> no-op, exit 0), so install.js can run it after
cloning OpenMOSS/MOVA into ext/mova. Verified 2026-06-19: with these three guards the
pipeline imports + runs single-GPU inference with yunchang absent (SDPA attention).

Run from the BASIWAN app root.
"""
from pathlib import Path
import sys

ROOT = Path("ext/mova")

# A complete AttnType fallback — must carry every member referenced as a default arg at
# import/class-build time (USPAttention.__init__ and replace_attention default to AttnType.FA).
_FALLBACK = (
    "    class AttnType:  # type: ignore  [basiwan] yunchang-absent fallback\n"
    "        FA = None\n"
    "        FA3 = None\n"
    "        TORCH = None\n"
)

# (relative path, the exact upstream import line, the guarded replacement)
_GUARD_IMPORT = [
    "mova/diffusion/pipelines/pipeline_mova.py",
    "mova/diffusion/pipelines/mova_train.py",
]


def _guard_yunchang_import(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        return f"[basiwan-mova] skip (not found): {rel}"
    t = p.read_text(encoding="utf-8")
    if "[basiwan] yunchang-absent fallback" in t:
        return f"[basiwan-mova] already patched: {rel}"
    needle = "from yunchang.kernels import AttnType"
    if needle not in t:
        return f"[basiwan-mova] no yunchang import to guard in {rel} (upstream changed?)"
    guarded = (
        "try:\n"
        "    from yunchang.kernels import AttnType\n"
        "except Exception:\n"
        "    # [basiwan] yunchang is Linux-only (multi-GPU CP attention) and unused by single-GPU\n"
        "    # inference; guard so the package imports on Windows/macOS (plain torch + SDPA).\n"
        + _FALLBACK
    )
    t = t.replace(needle, guarded, 1)
    p.write_text(t, encoding="utf-8")
    return f"[basiwan-mova] guarded yunchang import: {rel}"


def _fix_wan_dit_fallback() -> str:
    """The repo's OWN fallback in wan_video_dit.py defines only AttnType.FA3, but
    USPAttention.__init__ defaults to AttnType.FA -> AttributeError at import when yunchang
    is absent. Complete the fallback."""
    p = ROOT / "mova/diffusion/models/wan_video_dit.py"
    if not p.exists():
        return "[basiwan-mova] skip (not found): wan_video_dit.py"
    t = p.read_text(encoding="utf-8")
    if "[basiwan] yunchang-absent fallback" in t:
        return "[basiwan-mova] already patched: wan_video_dit.py"
    needle = "    class AttnType:  # type: ignore\n        FA3 = None\n"
    if needle not in t:
        return "[basiwan-mova] wan_video_dit fallback shape changed (upstream changed?) — review manually"
    t = t.replace(needle, _FALLBACK, 1)
    p.write_text(t, encoding="utf-8")
    return "[basiwan-mova] completed AttnType fallback (added FA/TORCH): wan_video_dit.py"


def main():
    if not ROOT.exists():
        print(f"[basiwan-mova] {ROOT} not present — nothing to patch (MOVA not vendored).")
        return
    for rel in _GUARD_IMPORT:
        print(_guard_yunchang_import(rel))
    print(_fix_wan_dit_fallback())


if __name__ == "__main__":
    main()
    sys.exit(0)
