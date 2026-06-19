"""Caption a MOVA A/V dataset with the GYM captioner (basi.caption, Qwen3-VL tier-auto,
mova_av_mode = the A/V prompt: content + transient sound event + trigger). Writes .txt next
to each clip (the gym .mp4+.txt convention); basi.mova_train.gen_mova_metadata then folds the
.txt into MOVA's metadata.json. Runs in the basiwan gym env (has transformers + qwen_vl_utils).

Usage: python tools/mova_caption.py <dataset_dir> [trigger_word] [limit]
  limit>0 captions only the first N clips (smoke). 0 = all. skip_existing is on (re-runnable).
ASCII.
"""
import sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from basi.caption import caption_dataset


def main():
    if len(sys.argv) < 2:
        print("usage: mova_caption.py <dataset_dir> [trigger_word] [limit]"); sys.exit(1)
    ddir = sys.argv[1].rstrip("/\\")
    trigger = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    vids = sorted(glob.glob(ddir + "/videos/*.mp4"))
    if limit:
        vids = vids[:limit]
    print(f"[mova-caption] {len(vids)} clips | trigger={trigger} | mova_av_mode=True", flush=True)
    caption_dataset(vids, trigger_word=trigger, mova_av_mode=True)
    print("MOVA_CAPTION_DONE", flush=True)


if __name__ == "__main__":
    main()
