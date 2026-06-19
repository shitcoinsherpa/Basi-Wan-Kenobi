"""Whisper audio judge for the MOVA A/V training loop -- transcribe a generated clip's audio to
MEASURE intelligibility + content (detected language + probability + transcript), so the loop can
self-judge audio quality (cpCER-style) without a human. Optionally score against an expected line.

Usage:
  python tools/whisper_judge.py <clip.mp4> [<clip2.mp4> ...] [--expect "the spoken line"] [--model base]

Prints one JSON line per clip: {file, language, language_prob, text, duration, [cer_vs_expect]}.
Uses faster-whisper on CPU (int8) -- short clips, no GPU contention. ASCII.
"""
import argparse
import json
import sys
from pathlib import Path


def _cer(a: str, b: str) -> float:
    """char error rate of a vs reference b (Levenshtein/len), lowercased+stripped. 0=perfect."""
    a = "".join(a.lower().split()); b = "".join(b.lower().split())
    if not b:
        return 0.0 if not a else 1.0
    # classic DP edit distance
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return round(prev[-1] / len(b), 3)


def judge(mp4: str, model, expect: str | None = None) -> dict:
    segs, info = model.transcribe(str(mp4), beam_size=5)
    text = " ".join(s.text for s in segs).strip()
    out = {"file": Path(mp4).name, "language": info.language,
           "language_prob": round(float(info.language_probability), 3),
           "duration": round(float(info.duration), 2), "text": text}
    if expect is not None:
        out["cer_vs_expect"] = _cer(text, expect)
        out["intelligible_english"] = (info.language == "en"
                                       and float(info.language_probability) > 0.5
                                       and len(text) > 0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--expect", default=None, help="expected spoken line (for CER scoring)")
    ap.add_argument("--model", default="base", help="faster-whisper model size (tiny/base/small)")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    m = WhisperModel(a.model, device="cpu", compute_type="int8")
    for c in a.clips:
        try:
            print(json.dumps(judge(c, m, a.expect), ensure_ascii=False), flush=True)
        except Exception as e:
            print(json.dumps({"file": Path(c).name, "error": f"{type(e).__name__}: {e}"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
