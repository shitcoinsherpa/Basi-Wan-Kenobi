"""Auto-captioning for Wan2.2 LoRA datasets.

Tier-aware model selection:
  - 24G+  Qwen/Qwen3-VL-8B-Instruct          — 256K ctx, native bilingual
  - 16G   Qwen/Qwen2.5-VL-7B-Instruct-AWQ    — INT4 weight-quant, fits in <10GB
  - 12G   Qwen/Qwen3-VL-4B-Instruct          — smaller dense, low-VRAM tier

Requires transformers>=4.57 for Qwen3-VL (pinned >=4.57,<5 in
ext/musubi-tuner/pyproject.toml).

Tarsier2-Recap-7B (omni-research) is video-description-specialized and beats GPT-4o
on DREAM-1K F1 — opt-in via model_id="omni-research/Tarsier2-Recap-7B".

Loaded on-demand, unloaded after batch.
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
    # Caption 60-100 words, Wan2.2's prompt_extend.py target range. UMT5-XXL
    # parses syntax, so we ask for natural language sentences (NOT booru tags)
    # and require the caption to start with the subject — the trigger word (if
    # any) is prepended programmatically by the gym after captioning, so the VLM
    # doesn't need to invent it.
    "Describe this video clip in 60-100 words of natural language. "
    "Include: the main subject's appearance and clothing, the specific "
    "action or motion happening, the background setting, the camera angle "
    "or shot type (close-up, medium shot, wide shot, etc.), and the overall "
    "visual style. Use complete sentences, not comma-separated tags. "
    "Start with the subject. Do not describe audio. Do not use phrases "
    "like 'this video shows' — describe directly."
)

# STYLE-LoRA variant: content-only captions, one subject per LoRA. With a
# dedicated trigger word, describing the constant style in every caption splits
# the signal between the trigger and the style words ("claymation, stop-motion"),
# weakening trigger binding and worsening bleed. Mixed subjects degrade coherence,
# so captions stay content-only and the trigger owns the style.
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

# MOVA / joint-AUDIO-VIDEO style variant. MOVA's text conditions BOTH the video
# AND audio towers (via the cross-attention bridge), so unlike the video-only STYLE_PROMPT
# (which says "Do not describe audio"), here we ADD one short clause for the salient TRANSIENT
# diegetic sound event -- that variable signal helps the bridge bind audio<->video. The
# CONSTANT visual AND audio style stay owned by the trigger (never described), and audio is held
# to ONE clause so it doesn't split the trigger signal. Dialogue is added via Whisper
# (asr_dialogue_recaption) where CER 0.0-0.06 holds; generic "he speaks" alone yields word-salad.
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


# --- MOVA prompt format + LLM "magic rewrite" -------------------------------
# The one-line note shown in the UI (Studio MOVA mode + Gym MOVA samples).
MOVA_PROMPT_GUIDE = (
    'MOVA prompt format: `<trigger>, <one visual sentence — subject + action + '
    'shot/setting>.` then, if the subject speaks, `He says, in English, "<exact '
    'words>".` Keep dialogue verbatim in quotes (MOVA conditions speech on the literal '
    'words); the trigger token carries the trained style, so skip style adjectives.'
)

_MOVA_REWRITE_SYS = (
    "You rewrite a user's idea into ONE MOVA audio+video generation prompt. Rules:\n"
    '1. Begin with the trigger token exactly: "{trigger}, " (with the comma). If the trigger '
    "is <none>, start directly with the subject.\n"
    "2. Then one concise visual sentence: the subject, what they do, and the shot/setting "
    "(e.g. medium shot, kitchen interior), ENDING WITH A PERIOD. Do NOT add art-style "
    "adjectives — the trigger carries the trained style.\n"
    "3. SPEAKER BINDING (critical for lip-sync — MOVA has no speaker selector; the spoken line "
    "attaches to whichever on-screen subject the prose names as speaking). Put the speaker "
    "front-and-center — alone and facing the camera — and describe the speaking MOTION (e.g. "
    "'mouth moving as he speaks'). If the subject speaks, append exactly: He says, in English, "
    '"<the spoken words>". For MULTIPLE subjects, name the speaker instead ("The boy says, in '
    "English, ...\") and describe the other(s) as silent/listening (e.g. '... the dragon listens "
    "silently').\n"
    "4. SPOKEN WORDS: copy the user's quoted words VERBATIM in quotes — never paraphrase, "
    "translate, respell, or alter them (MOVA conditions speech on the literal words). Do NOT "
    "phonetically respell anything: auto-respelling breaks words that were fine (measured net-"
    "negative). If a user wants a specific pronunciation they can spell it themselves. If no "
    "speech, omit this clause.\n"
    "5. Output ONLY the final prompt on one line — no preamble, no surrounding quotes, no notes."
)


def mova_format_prompt(user_text, trigger=None, model_id=None, max_new_tokens=160):
    """LLM 'magic rewrite': turn a rough idea into a correctly-formatted MOVA prompt
    (trigger + one visual sentence + a verbatim `He says, in English, "..."` clause). Uses the
    same Qwen3-VL the captioner / Suggest-prompt features use, text-only. Loads then unloads the
    model. Returns the rewritten one-line prompt; raises on failure (caller surfaces it)."""
    import torch
    mid = model_id or MODEL_TIERS[12]  # Qwen3-VL-4B — small/fast, text-only here
    sys_prompt = _MOVA_REWRITE_SYS.format(trigger=((trigger or "").strip() or "<none>"))
    model, processor = _load_vlm(mid)
    try:
        # One worked example (few-shot) — a 4B model follows the format far more reliably with a
        # concrete pattern than from rules alone. Style words stay OUT (the trigger owns the look);
        # speaker is foregrounded; the speaking motion is described; dialogue is verbatim in quotes.
        _ex_in = 'orel says I love Pinokio while his nose grows longer'
        _ex_out = ('moralorel, a boy in a green shirt and blue shorts stands alone and faces the '
                   'camera, his nose growing in length as he speaks. He says, in English, '
                   '"I love Pinokio".')
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": _ex_in}]},
            {"role": "assistant", "content": [{"type": "text", "text": _ex_out}]},
            {"role": "user", "content": [{"type": "text", "text": (user_text or "").strip()}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[:, inputs.input_ids.shape[1]:]
        result = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
        result = result.strip().strip("`").strip()
        if len(result) > 1 and result[0] == '"' and result[-1] == '"' and result.count('"') == 2:
            result = result[1:-1].strip()   # unwrap if the whole line got quoted
        tg = (trigger or "").strip()
        if tg and not result.lower().startswith(tg.lower()):
            result = f"{tg}, {result}"       # enforce trigger prefix if the model dropped it
        return result
    finally:
        _unload(model)


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


# Continuation-prompt system prompt: the UMT5/Wan2.2 I2V style — a
# continuation-prompt LLM that steers frame diversity. DELIBERATELY the opposite
# of DEFAULT_PROMPT_TEMPLATE above: I2V wants short motion-first prompts with
# static scene description stripped, because the conditioning frame already
# carries the appearance.
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


# Frame-picking variant: the VLM sees ALL candidate tail
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


# --- MOVA T2AV style assets: VLM-picked style reference frame + style descriptor --------------
# For SDXL+IP-Adapter style transfer (the per-prompt MOVA reference), the STYLE source must be a
# clean CLOSE character shot, and the SDXL prompt needs a short STYLE descriptor. Signal heuristics
# (Laplacian/saturation) misfire on text/title cards and wide shots, so we use the VLM (already
# loaded for captioning) to choose semantically. One model load does both jobs.
STYLE_PICK_SYS_PROMPT = (
    "You are selecting a STYLE-REFERENCE frame for image style-transfer from an animated show. "
    "You are given several candidate frames (numbered in the order shown). Choose the SINGLE best "
    "frame to represent the show's VISUAL STYLE: STRONGLY prefer a CLOSE shot of ONE character with "
    "the face and surface material clearly visible, sharp and well-lit. AVOID wide establishing "
    "shots, any frame with on-screen TEXT / credits / title cards / logos, blurry or transitional "
    "frames, and empty scenery. Also identify the ANIMATION MEDIUM precisely by its surface cues, "
    "choosing the closest: CLAYMATION/stop-motion (clay or plasticine figures, visible fingerprints/"
    "sculpt seams, soft 3D-lit physical models) -> say 'claymation, stop-motion clay figures'; "
    "PUPPET/felt stop-motion -> 'stop-motion felt puppets'; 2D HAND-DRAWN/cel (flat ink outlines, "
    "flat color fills) -> '2D cel animation, flat colors'; 3D CGI (smooth rendered surfaces) -> "
    "'3D CGI animation'. Look at TEXTURE: clay figures are matte, hand-sculpted and 3-dimensional, "
    "NOT flat-shaded drawings. Give a 3-5 word style/medium phrase; do NOT name the show or "
    'characters. Respond with ONLY a JSON object: '
    '{"frame": <number of chosen frame>, "style": "<3-5 word medium+look>", "why": "<one short sentence>"}'
)


def pick_style_frame_and_tag_vlm(candidate_pngs: list) -> tuple:
    """VLM picks the best STYLE-reference frame from candidates AND derives a 3-5 word style tag.
    Returns (index_into_candidate_pngs, style_tag, why). Falls back to (0, '', '') if the JSON
    contract is violated. One model load."""
    import torch
    model_id = MODEL_TIERS[12]
    print(f"[style-pick] loading {model_id} for style-frame pick + tag…")
    model, processor = _load_vlm(model_id)
    try:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": STYLE_PICK_SYS_PROMPT}]},
            {"role": "user",
             "content": ([{"type": "image", "image": p, "max_pixels": 448 * 448} for p in candidate_pngs]
                         + [{"type": "text",
                             "text": f"The {len(candidate_pngs)} candidate frames are attached in order (frame 1 first)."}])},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=120)
        raw = processor.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    finally:
        _unload(model)
        print("[style-pick] VLM unloaded")
    import json as _json, re as _re
    m = _re.search(r"\{.*\}", raw, _re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            idx = max(0, min(int(obj.get("frame", 1)) - 1, len(candidate_pngs) - 1))
            return idx, str(obj.get("style", "")).strip(), str(obj.get("why", "")).strip()
        except (ValueError, KeyError, _json.JSONDecodeError):
            pass
    return 0, "", ""


def derive_mova_style_assets(clip_paths: list, workspace_dir, n_candidates: int = 12) -> tuple:
    """Write the MOVA T2AV style assets into a workspace at train time: <ws>/style.png (the VLM-
    chosen close-up style frame) + <ws>/style_tag.txt (the 3-5 word style descriptor). Samples one
    mid-frame from up to n_candidates clips, lets the VLM pick the best + name the style. Returns
    (style_png_path|None, style_tag|''). Best-effort: on any failure returns (None,'') and the
    reference-maker falls back to ref.png. Reuses the gym's captioner VLM (one load)."""
    import tempfile
    from pathlib import Path as _P
    try:
        import av
        from PIL import Image
        import numpy as np
    except Exception:
        return None, ""
    ws = _P(workspace_dir); ws.mkdir(parents=True, exist_ok=True)
    tmp = _P(tempfile.mkdtemp(prefix="stylecand_"))
    cands = []
    for cp in [_P(c) for c in clip_paths][:n_candidates * 3]:   # scan extra; keep n non-flat
        if len(cands) >= n_candidates:
            break
        try:
            with av.open(str(cp)) as c:
                s = c.streams.video[0]
                total = s.frames or 0
                target = (total // 2) if total else 12
                idx = 0
                for f in c.decode(s):
                    if idx >= target:
                        rgb = f.to_ndarray(format="rgb24")
                        if float(rgb.mean(axis=2).std()) >= 18.0:   # skip flat/fade
                            out = tmp / f"cand_{len(cands):02d}.png"
                            Image.fromarray(rgb).save(out)
                            cands.append(str(out))
                        break
                    idx += 1
        except Exception:
            continue
    if not cands:
        return None, ""
    try:
        i, tag, _why = pick_style_frame_and_tag_vlm(cands)
    except Exception:
        i, tag = 0, ""
    import shutil
    style_png = ws / "style.png"
    shutil.copy2(cands[max(0, min(i, len(cands) - 1))], style_png)
    if tag:
        (ws / "style_tag.txt").write_text(tag, encoding="utf-8")
    try:
        shutil.rmtree(tmp)
    except Exception:
        pass
    return str(style_png), tag


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
    speech. MOVA is text-conditioned on the literal spoken words (dialogue-in-caption -> Whisper
    CER 0.0-0.06; generic 'he speaks' -> non-English word-salad). For each clip,
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
            # The Qwen caption ends with a generic sound clause ("...he speaks."); the verbatim
            # line replaces it — strip a trailing speak/talk clause so we don't get "he speaks. He
            # says ...". Non-speech sound clauses (footsteps, a door slams) are left intact.
            import re as _re
            base = _re.sub(
                r"[,.;]\s*(?:and\s+)?(?:he|she|they|the\s+[\w-]+)\s+(?:is\s+|are\s+)?"
                r"(?:speak|speaks|speaking|talk|talks|talking)\b[^.?!]*[.?!]?\s*$",
                "", base_cap, flags=_re.I).rstrip()
            if base and base[-1] not in ".!?":
                base += "."
            newcap = f'{base} He says, in English, "{line}"'
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
