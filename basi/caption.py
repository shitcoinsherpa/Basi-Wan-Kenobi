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
    # [2026-06-09] Tightened from 100-200 → 60-100 words to match Wan's own
    # prompt_extend.py target range (80-100 words). UMT5-XXL parses syntax,
    # so we ask for natural language sentences (NOT booru tags) and require
    # the caption to start with the subject — the trigger word (if any) is
    # prepended programmatically by the gym after captioning, so the VLM
    # doesn't need to invent it. See
    # memory/wan22_lora_training_brief_2026-06-09.md.
    "Describe this video clip in 60-100 words of natural language. "
    "Include: the main subject's appearance and clothing, the specific "
    "action or motion happening, the background setting, the camera angle "
    "or shot type (close-up, medium shot, wide shot, etc.), and the overall "
    "visual style. Use complete sentences, not comma-separated tags. "
    "Start with the subject. Do not describe audio. Do not use phrases "
    "like 'this video shows' — describe directly."
)

# [2026-06-10] STYLE-LoRA variant: content-only captions. With a dedicated
# trigger word, describing the constant style in every caption splits the
# signal between the trigger and the style words ("claymation, stop-motion"),
# weakening trigger binding and worsening bleed. Consensus prescription
# (musubi #182 constant-attribute rule applied to style; civitai 8487);
# no published video A/B — Flux image diaries (civitai 6792/7203) show
# style-in-caption also "works" but binds to the words instead of the
# trigger. We chose a trigger, so captions stay content-only.
STYLE_PROMPT_TEMPLATE = (
    "Describe this video clip in 60-100 words of natural language. "
    "Include: the main subject's appearance and clothing, the specific "
    "action or motion happening, the background setting, and the camera "
    "angle or shot type (close-up, medium shot, wide shot, etc.). Do NOT "
    "mention the art style, medium, or animation technique — never use "
    "words like claymation, stop-motion, animated, puppet, clay, or "
    "cartoon. Describe the content as if it were live-action footage. "
    "Use complete sentences, not comma-separated tags. Start with the "
    "subject. Do not describe audio. Do not use phrases like 'this video "
    "shows' — describe directly."
)

# [2026-06-15] MOVA / joint-AUDIO-VIDEO style variant. MOVA's text conditions BOTH the video
# AND audio towers (via the cross-attention bridge), so unlike the video-only STYLE_PROMPT
# (which says "Do not describe audio"), here we ADD one short clause for the salient TRANSIENT
# diegetic sound event -- that variable signal helps the bridge bind audio<->video. But the
# CONSTANT visual AND audio style stay owned by the trigger (never described), and audio is held
# to ONE clause so it doesn't split the trigger signal. See memory/mova_av_dataset_gym_spec.
MOVA_AV_PROMPT_TEMPLATE = (
    "Describe this video clip in 50-90 words of natural language. "
    "Include: the main subject's appearance and clothing, the specific action or motion, the "
    "background setting, and the camera angle or shot type (close-up, medium, wide). Then add "
    "ONE short clause naming the most salient diegetic SOUND EVENT if any (e.g. 'he speaks', "
    "'a door slams', 'footsteps'). Do NOT mention the art style, medium, or animation technique "
    "— never use words like claymation, stop-motion, animated, puppet, clay, or cartoon — and "
    "do NOT describe the constant background music or the overall sonic mix. Describe the content "
    "as if it were live-action. Use complete sentences, not tags. Start with the subject. Do not "
    "use phrases like 'this video shows' — describe directly."
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
                  max_new_tokens: int = 300,
                  prompt_template: str | None = None) -> str:
    """Single-video caption. Caller manages model lifecycle."""
    import torch
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": video_path, "max_pixels": 360 * 420, "fps": 1.0},
            {"type": "text", "text": prompt_template or PROMPT_TEMPLATE},
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


# [2026-06-09 #372] Continuation-prompt system prompt, adapted from the
# vendored wan/utils/system_prompt.py I2V_A14B_EN_SYS_PROMPT style — the
# style the UMT5/Wan2.2 I2V training pipeline was calibrated against.
# DELIBERATELY the opposite of DEFAULT_PROMPT_TEMPLATE above: I2V wants
# short motion-first prompts with static scene description stripped,
# because the conditioning frame already carries the appearance.
CONTINUATION_SYS_PROMPT = (
    "You are an expert at writing continuation prompts for an "
    "image-to-video model. You are given the last frame of the previous "
    "clip, the prompt that generated that clip, and optionally a "
    "direction from the user. Write the prompt for the next clip, which "
    "starts exactly at this frame. Requirements: Emphasize motion - "
    "describe what the subject does next and how the camera moves. Do "
    "not describe static elements already visible in the frame. Copy "
    "identity and style wording from the previous prompt verbatim where "
    "needed for consistency (character appearance, clothing, art style); "
    "change only the action. If a user direction is given, the new "
    "action must follow it; otherwise continue the previous action to "
    "its natural next beat. One paragraph, under 100 words, English "
    "only. Output only the prompt."
)


def caption_image(model, processor, image_path: str, prompt_text: str,
                  system_prompt: str | None = None,
                  max_new_tokens: int = 200) -> str:
    """Single-image VLM call. Caller manages model lifecycle (same contract
    as caption_video)."""
    import torch
    messages = []
    if system_prompt:
        messages.append({"role": "system",
                         "content": [{"type": "text", "text": system_prompt}]})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_path, "max_pixels": 512 * 512},
            {"type": "text", "text": prompt_text},
        ],
    })
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
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()


def suggest_continuation_prompt(last_frame_png: str, previous_prompt: str,
                                user_direction: str = "") -> str:
    """One-shot: load VLM, write a continuation prompt from the last frame,
    unload. Uses the 12G-tier model (Qwen3-VL-4B) regardless of VRAM — this
    is an interactive button while the ~14B video worker may hold reserved
    VRAM, and 4B loads in seconds vs the 8B's tens; prompt-rewriting does
    not need the 8B's captioning ceiling."""
    model_id = MODEL_TIERS[12]
    user_text = (f"Previous prompt: {previous_prompt}\n"
                 f"User direction: {user_direction.strip() or 'none - continue naturally'}")
    print(f"[continue] loading {model_id} for prompt suggestion…")
    model, processor = _load_vlm(model_id)
    try:
        return caption_image(model, processor, last_frame_png, user_text,
                             system_prompt=CONTINUATION_SYS_PROMPT)
    finally:
        _unload(model)
        print("[continue] VLM unloaded")


# [2026-06-10] Frame-picking variant: the VLM sees ALL candidate tail
# frames and both chooses the best continuation point and writes the
# prompt for it — one model load, both jobs. The Laplacian-sharpness
# ranking is a proxy; the VLM can also judge eyes-mid-blink, motion
# smear direction, and compositional usefulness, which pixel variance
# cannot.
CONTINUATION_PICK_SYS_PROMPT = (
    "You are an expert at continuing AI-generated videos with an "
    "image-to-video model. You are given several candidate frames taken "
    "from the end of the previous clip (numbered in the order shown), "
    "the prompt that generated that clip, and optionally a direction "
    "from the user. First choose the single best frame to continue "
    "from: prefer sharp frames with the subject clearly visible, eyes "
    "open, limbs in a continuable pose; avoid motion blur, mid-blink, "
    "or transitional poses. Then write the prompt for the next clip, "
    "which starts exactly at the chosen frame. Prompt requirements: "
    "emphasize motion - what the subject does next and how the camera "
    "moves; do not describe static elements already visible in the "
    "frame; copy identity and style wording from the previous prompt "
    "verbatim where needed for consistency; change only the action; if "
    "a user direction is given, follow it; one paragraph, under 100 "
    "words, English only. Respond with ONLY a JSON object: "
    '{"frame": <number of chosen frame>, "why": "<one short sentence>", '
    '"prompt": "<the continuation prompt>"}'
)


def suggest_continuation_with_pick(tail_pngs: list[str], previous_prompt: str,
                                   user_direction: str = "") -> tuple[int, str, str]:
    """VLM picks the best tail frame AND writes the continuation prompt.

    Returns (frame_index_into_tail_pngs, prompt, why). Falls back to
    (0, raw_text, "") if the JSON contract is violated — index 0 is the
    Laplacian-sharpest frame, so the fallback is the previous behavior."""
    import torch
    model_id = MODEL_TIERS[12]
    user_text = (f"Previous prompt: {previous_prompt}\n"
                 f"User direction: {user_direction.strip() or 'none - continue naturally'}\n"
                 f"The {len(tail_pngs)} candidate frames are attached in order "
                 f"(frame 1 first).")
    print(f"[continue] loading {model_id} for frame pick + prompt…")
    model, processor = _load_vlm(model_id)
    try:
        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": CONTINUATION_PICK_SYS_PROMPT}]},
            {"role": "user",
             "content": ([{"type": "image", "image": p, "max_pixels": 448 * 448}
                          for p in tail_pngs]
                         + [{"type": "text", "text": user_text}])},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=300)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
    finally:
        _unload(model)
        print("[continue] VLM unloaded")

    import json as _json
    import re as _re
    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            idx = max(0, min(int(obj.get("frame", 1)) - 1, len(tail_pngs) - 1))
            prompt = str(obj.get("prompt", "")).strip()
            why = str(obj.get("why", "")).strip()
            if prompt:
                return idx, prompt, why
        except (ValueError, KeyError, _json.JSONDecodeError):
            pass
    # Contract violated — sharpest frame + whole response as prompt draft.
    return 0, raw, ""


def caption_dataset(video_paths: Iterable[str], trigger_word: str | None = None,
                    output_dir: str | Path | None = None,
                    model_id: str | None = None,
                    skip_existing: bool = True,
                    progress_cb=None,
                    style_mode: bool = False,
                    mova_av_mode: bool = False) -> dict[str, str]:
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
            if progress_cb is not None:
                try:
                    progress_cb(i, len(todo), p.name)
                except Exception:
                    pass  # UI callback must never kill the batch
            cap = caption_video(
                model, processor, str(p), trigger_word=trigger_word,
                prompt_template=(MOVA_AV_PROMPT_TEMPLATE if mova_av_mode
                                 else STYLE_PROMPT_TEMPLATE if style_mode else None))
            (p.with_suffix(".txt")).write_text(cap + "\n", encoding="utf-8")
            results[str(p)] = cap
    finally:
        _unload(model)
        print("[caption] model unloaded")
    return results


def asr_dialogue_recaption(dataset_dir, model: str = "base", min_prob: float = 0.6,
                           min_words: int = 2, apply: bool = True, progress_cb=None) -> dict:
    """Append VERBATIM spoken dialogue to MOVA captions -- THE #1 fix for intelligible generated
    speech. MOVA is text-conditioned on the literal spoken words (measured 2026-06-18: dialogue-in-
    caption -> Whisper CER 0.0-0.06; generic 'he speaks' -> non-English word-salad). For each clip,
    CPU-Whisper the audio; if it holds intelligible English speech, append -- He says, in English,
    "...". Music/SFX/silence clips keep their caption. Idempotent + reversible: the clean caption is
    backed up to <clip>.txt.orig once and every run rebuilds from it. Shared by the Gym UI and
    tools/mova_recaption_asr.py. Returns {"speech": n, "total": n, "applied": bool}.

    Raises ImportError (with an actionable message) if faster-whisper isn't installed -- callers
    degrade gracefully, because skipping dialogue silently would reintroduce the babble bug.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:  # noqa: BLE001 - surface a clear, actionable message
        raise ImportError(
            "faster-whisper is required for MOVA dialogue captions (the #1 audio-intelligibility "
            "fix). Install it (`pip install faster-whisper`) or run tools/mova_recaption_asr.py in "
            "an env that has it. Without verbatim dialogue in captions, generated speech is "
            "non-English babble (measured).") from e
    m = WhisperModel(model, device="cpu", compute_type="int8")
    clips = [c for c in sorted(Path(dataset_dir).glob("**/*.mp4")) if "_dropped" not in c.parts]
    n_speech = 0
    for i, c in enumerate(clips):
        if progress_cb is not None:
            try:
                progress_cb(i, len(clips), c.name)
            except Exception:
                pass
        try:
            segs, info = m.transcribe(str(c), beam_size=5)
            text = " ".join(s.text for s in segs).strip()
        except Exception:
            continue
        has_speech = (info.language == "en" and float(info.language_probability) > min_prob
                      and len(text.split()) >= min_words)
        txt = c.with_suffix(".txt")
        orig = txt.parent / (txt.name + ".orig")
        base_cap = (orig.read_text(encoding="utf-8").strip() if orig.exists()
                    else (txt.read_text(encoding="utf-8").strip() if txt.exists() else ""))
        if has_speech:
            line = text.strip().strip('"')
            newcap = f'{base_cap} He says, in English, "{line}"'
            n_speech += 1
        else:
            newcap = base_cap
        if apply:
            if not orig.exists() and txt.exists():
                orig.write_text(base_cap, encoding="utf-8")  # back up the clean caption once
            txt.write_text(newcap, encoding="utf-8")
    return {"speech": n_speech, "total": len(clips), "applied": apply}


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
