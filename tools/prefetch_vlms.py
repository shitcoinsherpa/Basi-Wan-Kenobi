"""Download the VLM models at INSTALL time with visible progress.

Called by install.js (and harmless to re-run — snapshot_download resumes
partial blobs and no-ops when complete). Downloads:
  - Qwen3-VL-4B  — the Studio "Suggest next prompt" model (all tiers)
  - the VRAM-tier captioner (8B at 24G, 7B-AWQ at 16G, 4B at 12G)

Rationale: fetch these at install time, not on first button click —
otherwise 8-17 GB downloads SILENTLY behind a frozen progress bar, which
looks like a dead app. In install, the Pinokio terminal shows
huggingface_hub's live byte progress. The app's launch-time background
prefetch remains as a safety net for installs that skipped or aborted
these downloads.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import snapshot_download  # noqa: E402

from basi.caption import MODEL_TIERS, select_model_for_vram  # noqa: E402

models = list(dict.fromkeys([MODEL_TIERS[12], select_model_for_vram()]))
print(f"[basiwan] prefetching {len(models)} VLM model(s): {models}", flush=True)
for mid in models:
    # Skip cleanly if already cached: local_files_only never touches the network, so a re-run
    # (e.g. Update) doesn't re-scan/re-pull a model we already have. Only fetch what's missing.
    try:
        snapshot_download(mid, local_files_only=True)
        print(f"[basiwan] {mid} already cached — skipping", flush=True)
        continue
    except Exception:
        pass  # not fully cached -> fetch below
    print(f"[basiwan] downloading {mid} …", flush=True)
    try:
        snapshot_download(mid)
        print(f"[basiwan] {mid} complete", flush=True)
    except Exception as e:
        # Non-fatal + isolated: one model's hiccup must not abort the others or the install. The
        # app's launch-time background prefetch (and first-use download) is the safety net.
        print(f"[basiwan] WARN: {mid} prefetch failed ({type(e).__name__}: {e}); "
              f"it will download on first use in the app", flush=True)
print("[basiwan] VLM prefetch done", flush=True)
