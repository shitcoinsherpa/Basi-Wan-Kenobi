"""MOVA A/V dataset curation CLI -- audit / apply, reversible. The "correct for all users"
curation pass over a built dataset dir (.mp4 + .txt pairs), using basi.mova_data's research-backed
(deep-research 2026-06-16) flag-not-delete primitives.

  audit (default): analyze every clip, write _curation_report.json, print what WOULD be dropped.
                   Read-only -- touches nothing on disk except the report.
  --apply        : MOVE flagged clips (.mp4 + .txt) to <dataset>/_dropped/ (reversible; never
                   deletes). gen_mova_metadata then naturally excludes them; per-clip latent cache
                   for KEPT clips is untouched, so no re-cache is needed.

Drop policy: ALWAYS drop silent/no-audio (MOVA needs real audio) + near-duplicates (pHash). OPT-IN:
--drop-music (music-dominated) / --drop-static. Thresholds are calibration GUESS defaults -- run an
audit FIRST and sanity-check the counts before --apply. CPU only, no GPU, no model download.

Usage:
  python tools/mova_curate.py <dataset_dir> [--drop-music] [--drop-static] [--apply]
                              [--music-thresh 0.70] [--static-thresh 2.0] [--hamming 5] [--no-dedup]
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basi.mova_data import (audio_content_flags, classify_audio_content,  # noqa: E402
                            clip_motion_score, frame_phashes, find_near_duplicates,
                            curate_clip_set)


def analyze(ds: Path, music_thresh: float, static_thresh: float) -> tuple[list[dict], dict]:
    clips = [c for c in sorted(ds.glob("**/*.mp4")) if "_dropped" not in c.parts]
    out: list[dict] = []
    hashes: dict[str, list] = {}
    for i, c in enumerate(clips):
        audio = audio_content_flags(c)
        aclass = classify_audio_content(c)
        motion = clip_motion_score(c)
        flags = list(audio.get("flags", []))
        ms = aclass.get("music_score")
        if aclass.get("available") and ms is not None and ms >= music_thresh:
            flags.append("music")
        if motion is not None and motion < static_thresh:
            flags.append("static")
        drop = any(f in flags for f in ("silent", "no_audio"))
        out.append({"path": str(c), "flags": flags, "drop_recommended": drop,
                    "motion": motion, "audio_class": aclass})
        hashes[str(c)] = frame_phashes(c)
        print(f"  [{i + 1}/{len(clips)}] {c.name}: flags={flags or '-'} "
              f"motion={motion} music={ms}", flush=True)
    return out, hashes


def main() -> int:
    ap = argparse.ArgumentParser(description="MOVA A/V dataset curation (audit/apply, reversible)")
    ap.add_argument("dataset")
    ap.add_argument("--apply", action="store_true", help="move flagged clips to _dropped/ (else audit-only)")
    ap.add_argument("--drop-music", action="store_true")
    ap.add_argument("--drop-static", action="store_true")
    ap.add_argument("--drop-edges", action="store_true",
                    help="drop the first+last clip of each episode (intro/title cards + end credits)")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--music-thresh", type=float, default=0.70)
    ap.add_argument("--static-thresh", type=float, default=2.0)
    ap.add_argument("--hamming", type=int, default=5)
    a = ap.parse_args()
    ds = Path(a.dataset)
    if not ds.is_dir():
        print(f"ERROR: not a directory: {ds}")
        return 2

    print(f"== analyzing {ds} ==", flush=True)
    clips, hashes = analyze(ds, a.music_thresh, a.static_thresh)
    if not clips:
        print("no clips found.")
        return 1

    # Reuse the library's cross-clip curation for the drop decision, but feed it the pHashes we
    # already computed (so dedup doesn't recompute them). curate_clip_set recomputes hashes itself,
    # so to avoid a second pass we replicate its decision here using find_near_duplicates directly.
    rep = curate_clip_set(clips, auto_curate=False, drop_music=a.drop_music,
                          drop_static=a.drop_static, drop_edges=a.drop_edges,
                          dedup=False)   # dedup handled below with our hashes
    drop = {d["path"]: d["reason"] for d in rep["drop"]}
    n_edges = len(rep.get("edges", []))
    dedup_groups = 0
    if not a.no_dedup:
        survivors = {p: h for p, h in hashes.items() if p not in drop}
        for grp in find_near_duplicates(survivors, max_hamming=a.hamming):
            dedup_groups += 1
            for dup in grp[1:]:
                drop.setdefault(dup, f"near-duplicate of {Path(grp[0]).name}")

    tally: Counter = Counter()
    for c in clips:
        for f in c["flags"]:
            tally[f] += 1

    print("\n== REPORT ==")
    print(f"  total clips         : {len(clips)}")
    print(f"  flag tally          : {dict(tally)}")
    print(f"  near-dup groups     : {dedup_groups}")
    print(f"  intro/outro edges   : {n_edges} ({'dropping' if a.drop_edges else 'flagged — use --drop-edges'})")
    print(f"  would-DROP          : {len(drop)}  ->  would-KEEP: {len(clips) - len(drop)}")
    print(f"  thresholds          : music>={a.music_thresh} static<{a.static_thresh} "
          f"hamming<={a.hamming} drop_music={a.drop_music} drop_static={a.drop_static} "
          f"drop_edges={a.drop_edges}")
    for p, why in list(drop.items())[:30]:
        print(f"    - {Path(p).name}: {why}")
    if len(drop) > 30:
        print(f"    ... and {len(drop) - 30} more")

    report = {"dataset": str(ds), "total": len(clips), "tally": dict(tally),
              "dedup_groups": dedup_groups, "edges": n_edges, "would_drop": len(drop),
              "drop": [{"path": p, "reason": w} for p, w in drop.items()],
              "thresholds": {"music": a.music_thresh, "static": a.static_thresh,
                             "hamming": a.hamming, "drop_music": a.drop_music,
                             "drop_static": a.drop_static, "drop_edges": a.drop_edges}}
    (ds / "_curation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  report -> {ds / '_curation_report.json'}")

    if a.apply:
        dropdir = ds / "_dropped"
        dropdir.mkdir(exist_ok=True)
        moved = 0
        for p in drop:
            src = Path(p)
            for ext in (".mp4", ".txt"):
                f = src.with_suffix(ext)
                if f.exists():
                    shutil.move(str(f), str(dropdir / f.name))
                    moved += 1
        print(f"\n== APPLIED: moved {moved} files ({len(drop)} clips) to {dropdir} "
              f"(reversible -- move back to undo). Re-run training to use the curated set. ==")
    else:
        print("\n(AUDIT ONLY -- sanity-check the counts, then re-run with --apply to move "
              "flagged clips to _dropped/)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
