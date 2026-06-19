"""Re-caption a MOVA A/V dataset with ASR dialogue -- the #1 fix for intelligible generated speech
(MOVA is conditioned on the verbatim spoken words; measured 2026-06-17: dialogue-in-prompt -> CER
0.083, generic 'he speaks' -> word-salad). For each clip, Whisper-transcribe the audio; if it holds
intelligible English speech, append the line in MOVA's format -- He says, in English, "...". Clips
without clear speech (music/SFX/silence) keep their caption. Idempotent + reversible: the clean
original is backed up to <clip>.txt.orig and every run rebuilds from it.

Usage:
  python tools/mova_recaption_asr.py <dataset_dir> [--model base] [--min-prob 0.6] [--apply]
Audit (no --apply) prints per-clip speech/lang/text; --apply writes the .txt files.
CPU faster-whisper. ASCII.
"""
import argparse
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--model", default="base", help="faster-whisper size (base good; small=better/slower)")
    ap.add_argument("--min-prob", type=float, default=0.6, help="min language probability to accept as speech")
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--apply", action="store_true", help="write .txt files (else audit only)")
    a = ap.parse_args()
    from faster_whisper import WhisperModel
    m = WhisperModel(a.model, device="cpu", compute_type="int8")
    clips = [c for c in sorted(Path(a.dataset).glob("**/*.mp4")) if "_dropped" not in c.parts]
    n_speech = n_none = 0
    for i, c in enumerate(clips):
        try:
            segs, info = m.transcribe(str(c), beam_size=5)
            text = " ".join(s.text for s in segs).strip()
        except Exception as e:
            print(f"[{i+1}/{len(clips)}] {c.name}: ASR ERROR {type(e).__name__}", flush=True)
            continue
        has_speech = (info.language == "en" and float(info.language_probability) > a.min_prob
                      and len(text.split()) >= a.min_words)
        txt = c.with_suffix(".txt")
        orig = txt.parent / (txt.name + ".orig")
        base_cap = (orig.read_text(encoding="utf-8").strip() if orig.exists()
                    else (txt.read_text(encoding="utf-8").strip() if txt.exists() else ""))
        if has_speech:
            # avoid double-quoting; clean the transcript
            line = text.strip().strip('"')
            newcap = f'{base_cap} He says, in English, "{line}"'
            n_speech += 1
        else:
            newcap = base_cap
            n_none += 1
        print(f"[{i+1}/{len(clips)}] {c.name}: speech={has_speech} "
              f"{info.language}/{float(info.language_probability):.2f} :: {text[:70]}", flush=True)
        if a.apply:
            if not orig.exists() and txt.exists():
                orig.write_text(base_cap, encoding="utf-8")   # back up clean original once
            txt.write_text(newcap, encoding="utf-8")
    print(f"\n== {n_speech}/{len(clips)} clips got dialogue, {n_none} kept as-is. "
          f"apply={a.apply} (re-run with --apply to write) ==", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
