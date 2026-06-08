"""Auto-captioning for Wan2.2 LoRA datasets.

Tier-aware model selection (research dated 2026-05, supersedes Flux-Gym's Florence-2):
  - 24G+  Qwen/Qwen3-VL-8B-Instruct          — Oct-2025 release, 256K ctx, native bilingual
  - 16G   Qwen/Qwen2.5-VL-7B-Instruct-AWQ    — INT4 weight-quant, fits in <10GB
  - 12G   Qwen/Qwen3-VL-4B-Instruct          — smaller dense, low-VRAM tier

Requires transformers>=4.57 for Qwen3-VL. musubi-tuner's pin to ==4.56.1 was a stale
lower-bound (research 2026-05-28): Qwen3ForCausalLM landed in transformers 4.51 and
4.56→4.57 has zero breaking changes affecting musubi's API surface. The pin was
loosened to >=4.57,<5 in ext/musubi-tuner/pyproject.toml.

Tarsier2-Recap-7B (omni-research) is video-description-specialized and beats GPT-4o
on DREAM-1K F1 — opt-in via model_id="omni-research/Tarsier2-Recap-7B".

Loaded on-demand, unloaded after batch (Flux-Gym pattern).
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Iterable

from .presets import detect_vram_gb

MODEL_TIERS = {
    24: "Qwen/Qwen3-VL-8B-Instruct",
    16: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    12: "Qwen/Qwen3-VL-4B-Instruct",
}
DEFAULT_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

DEFAULT_PROMPT_TEMPLATE = (
    "Describe this video in 100-200 words. Begin with motion and action, then "
    "scene/setting, lighting, and camera motion. Be specific about visible "
    "subjects, their movement, and environmental details. Avoid negations. "
    "Do not include phrases like 'this video shows' — describe directly."
)

# Override via env (Pinokio + CLI install both honor this); falls back to default.
# Lets users tune the captioner prompt without forking the gym.
PROMPT_TEMPLATE = os.environ.get("BASIWAN_CAPTION_PROMPT", DEFAULT_PROMPT_TEMPLATE)


def select_model_for_vram(vram_gb: int | None = None) -> str:
    """Pick a captioner matched to VRAM. Falls back to 12G tier if unknown."""
    if vram_gb is None:
        vram_gb = detect_vram_gb()
    if vram_gb >= 22:
        return MODEL_TIERS[24]
    if vram_gb >= 14:
        return MODEL_TIERS[16]
    return MODEL_TIERS[12]


def _load_vlm(model_id: str):
    """Load Qwen2-VL / Qwen2.5-VL / Qwen3-VL. Returns (model, processor)."""
    import torch
    from transformers import AutoProcessor

    mid = model_id.lower()
    if "qwen3-vl" in mid:
        from transformers import Qwen3VLForConditionalGeneration as ModelCls
    elif "qwen2.5-vl" in mid:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    else:
        from transformers import Qwen2VLForConditionalGeneration as ModelCls

    model = ModelCls.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def _unload(model):
    import torch
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def caption_video(model, processor, video_path: str, trigger_word: str | None = None,
                  max_new_tokens: int = 300) -> str:
    """Single-video caption. Caller manages model lifecycle."""
    import torch
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
            {"type": "text", "text": PROMPT_TEMPLATE},
        ],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    from qwen_vl_utils import process_vision_info
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    caption = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    if trigger_word:
        caption = f"{trigger_word}, {caption}"
    return caption


def caption_dataset(video_paths: Iterable[str], trigger_word: str | None = None,
                    output_dir: str | Path | None = None,
                    model_id: str | None = None,
                    skip_existing: bool = True) -> dict[str, str]:
    """Caption a batch of videos. Writes .txt next to each video (sd-scripts convention).

    model_id=None → tier-aware auto-select via detected VRAM.
    Returns {video_path: caption}. Skips videos with existing .txt (configurable).
    Loads model once, captions all, unloads.
    """
    if model_id is None:
        model_id = select_model_for_vram()
    paths = [Path(p) for p in video_paths]
    todo = []
    skipped = {}
    for p in paths:
        cap_path = p.with_suffix(".txt")
        if skip_existing and cap_path.exists():
            skipped[str(p)] = cap_path.read_text(encoding="utf-8").strip()
            continue
        todo.append(p)
    if not todo:
        return skipped

    print(f"[caption] loading {model_id}…")
    model, processor = _load_vlm(model_id)
    try:
        results = dict(skipped)
        for i, p in enumerate(todo):
            print(f"[caption] {i+1}/{len(todo)} {p.name}…")
            cap = caption_video(model, processor, str(p), trigger_word=trigger_word)
            (p.with_suffix(".txt")).write_text(cap + "\n", encoding="utf-8")
            results[str(p)] = cap
    finally:
        _unload(model)
        print("[caption] model unloaded")
    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: caption.py <dataset_dir> [trigger_word] [model_id]")
        sys.exit(1)
    dataset_dir = Path(sys.argv[1])
    trigger = sys.argv[2] if len(sys.argv) > 2 else None
    mid = sys.argv[3] if len(sys.argv) > 3 else None
    videos = [p for p in sorted(dataset_dir.iterdir())
              if p.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv"}]
    print(f"Found {len(videos)} videos")
    results = caption_dataset([str(p) for p in videos], trigger_word=trigger, model_id=mid)
    for path, cap in results.items():
        print(f"\n{Path(path).name}:\n  {cap[:200]}…")
