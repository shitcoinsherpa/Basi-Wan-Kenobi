"""Download the VLM models at INSTALL time with visible progress.

Called by install.js (and harmless to re-run — snapshot_download resumes
partial blobs and no-ops when complete). Downloads:
  - Qwen3-VL-4B  — the Studio "Suggest next prompt" model (all tiers)
  - the VRAM-tier captioner (8B at 24G, 7B-AWQ at 16G, 4B at 12G)

Rationale (2026-06-10): both models previously downloaded SILENTLY on
first button click — 8-17 GB behind a frozen progress bar looked like a
dead app. Downloads belong in install, where the Pinokio terminal shows
huggingface_hub's live byte progress. The app's launch-time background
prefetch remains as the safety net for installs done before this script
existed (or aborted downloads).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download  # noqa: E402

from basi.caption import MODEL_TIERS, select_model_for_vram  # noqa: E402

models = list(dict.fromkeys([MODEL_TIERS[12], select_model_for_vram()]))
print(f"[basiwan] prefetching {len(models)} VLM model(s): {models}", flush=True)
for mid in models:
    print(f"[basiwan] downloading {mid} …", flush=True)
    snapshot_download(mid)
    print(f"[basiwan] {mid} complete", flush=True)
print("[basiwan] VLM prefetch done", flush=True)
