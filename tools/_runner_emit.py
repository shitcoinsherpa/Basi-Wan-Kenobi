"""Structured event emitter for run_one_video_gguf.py.

Writes JSON events to stdout for the persistent-worker IPC path AND keeps the
existing human-readable lines for the legacy subprocess-per-click path.

Use:
    from _runner_emit import emit
    emit("phase", name="step_loop", wall_s=68.4, n_steps=4)

The default behavior is BOTH: a `[BASIWAN-EVENT] {json}` line and the existing
human-readable line. The persistent worker enables BASIWAN_RUNNER_JSON_ONLY=1
to suppress the human prefix and emit only `{json}` lines for easier parsing.
"""
from __future__ import annotations
import json
import os
import sys


_JSON_ONLY = os.environ.get("BASIWAN_RUNNER_JSON_ONLY") == "1"


def emit(event: str, **kw) -> None:
    """Emit one structured event.

    Args:
        event: short event kind (e.g. "phase", "step", "ready", "result", "error")
        **kw: event-specific fields
    """
    rec = {"_basiwan_event": True, "event": event, **kw}
    line = json.dumps(rec, separators=(",", ":"))
    if _JSON_ONLY:
        sys.stdout.write(line + "\n")
    else:
        sys.stdout.write(f"[BASIWAN-EVENT] {line}\n")
    sys.stdout.flush()


def is_json_only() -> bool:
    return _JSON_ONLY
