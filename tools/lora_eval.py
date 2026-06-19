"""LoRA evaluation harness (#383) — measure a trained style LoRA instead of
eyeballing it. Three robust, reproducible signals on THIS machine (no fragile
external CSD checkpoint — DINOv2 self-supervised features are the dependency-
light style metric the eval research flagged, less content-biased than CLIP):

  style_sim : DINOv2 cosine of a video's frames vs a prototype built from the
              training-set frames. High = looks like the trained style.
  clip_t    : transformers CLIP text-video alignment (prompt adherence).
  bleed     : style_sim of a NO-trigger render minus the dataset noise floor.
              >floor+eps = the LoRA styles even without its trigger (leak).

Pass 1 (zero new GPU): score the per-epoch sample renders that already exist
(_00 trigger / _01 no-trigger) → style-development + bleed-onset curves.
Pass 2 (GPU): main grid of top checkpoints × strengths under the Lightning
serving condition (separate driver).

Run in the Pinokio venv (has torch + transformers + imageio).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

_DINO = "facebook/dinov2-small"   # 22M params, ~90MB; robust + fast
_CLIP = "openai/clip-vit-base-patch32"


def _load_dino():
    import torch
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(_DINO)
    model = AutoModel.from_pretrained(_DINO).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, proc


def _load_clip():
    import torch
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(_CLIP).eval()
    proc = CLIPProcessor.from_pretrained(_CLIP)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, proc


def _read_frames(path, n=8):
    """Evenly-sample n frames from an mp4 (or load a PNG/JPG as 1 frame).
    Returns list of HxWx3 uint8 RGB arrays."""
    import imageio.v3 as iio
    p = Path(path)
    if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
        im = iio.imread(p)
        return [im[..., :3]]
    frames = iio.imread(p, plugin="pyav")  # (T,H,W,3)
    T = len(frames)
    idx = np.linspace(0, T - 1, min(n, T)).round().astype(int)
    return [frames[i][..., :3] for i in idx]


def _dino_embed(frames, model, proc):
    """Mean-pooled L2-normalized DINOv2 CLS embedding over frames."""
    import torch
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    inp = proc(images=ims, return_tensors="pt")
    dev = next(model.parameters()).device
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp).last_hidden_state[:, 0]  # CLS token per frame
    out = torch.nn.functional.normalize(out, dim=-1)
    v = out.mean(0)
    return torch.nn.functional.normalize(v, dim=0).cpu().numpy()


def _clip_t(frames, prompt, model, proc):
    """Mean text-image cosine (CLIP) over frames — prompt adherence."""
    import torch
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    inp = proc(text=[prompt], images=ims, return_tensors="pt",
               padding=True, truncation=True)
    dev = next(model.parameters()).device
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        out = model(**inp)
    im = torch.nn.functional.normalize(out.image_embeds, dim=-1)
    tx = torch.nn.functional.normalize(out.text_embeds, dim=-1)
    return float((im @ tx.T).mean().item())


def build_style_prototype(frame_sources, model, proc):
    """Mean DINOv2 embedding over many training-set frames = the style anchor.
    Also returns the intra-set noise floor: mean cosine of individual frame
    embeddings to the prototype (a video that merely shares the domain scores
    around here; a styled match scores well above)."""
    embs = []
    for src in frame_sources:
        try:
            embs.append(_dino_embed(_read_frames(src, n=1), model, proc))
        except Exception:
            continue
    E = np.stack(embs)
    proto = E.mean(0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    floor = float(np.mean(E @ proto))  # self-similarity baseline
    return proto, floor, len(embs)


def style_sim(frames, proto, model, proc):
    e = _dino_embed(frames, model, proc)
    return float(np.dot(e, proto))


# [#397/W10] CLIP image-image style as a SECOND style axis. DINOv2 alone
# understates style (the ~0.5 "panic" was a one-axis artifact — memory
# restyle_quality_levers); CLIP's image features capture global appearance/palette
# that DINOv2's content-biased CLS can miss. Two axes that AGREE = trustworthy;
# divergence flags a metric artifact. (Distinct from _clip_t, which is text->image
# prompt adherence — this is image->image style.)
def _clip_embed(frames, model, proc):
    """Mean-pooled L2-normalized CLIP IMAGE embedding over frames."""
    import torch
    from PIL import Image
    ims = [Image.fromarray(f) for f in frames]
    inp = proc(images=ims, return_tensors="pt")
    dev = next(model.parameters()).device
    inp = {k: v.to(dev) for k, v in inp.items()}
    with torch.no_grad():
        feats = model.get_image_features(**inp)
    feats = torch.nn.functional.normalize(feats, dim=-1)
    v = feats.mean(0)
    return torch.nn.functional.normalize(v, dim=0).cpu().numpy()


def build_clip_prototype(frame_sources, model, proc):
    """CLIP-image style anchor + intra-set floor (mirrors build_style_prototype
    but on CLIP image features). Same model/proc as _clip_t — no extra load."""
    embs = []
    for src in frame_sources:
        try:
            embs.append(_clip_embed(_read_frames(src, n=1), model, proc))
        except Exception:
            continue
    if not embs:
        return None, 0.0, 0
    E = np.stack(embs)
    proto = E.mean(0)
    proto = proto / (np.linalg.norm(proto) + 1e-8)
    floor = float(np.mean(E @ proto))
    return proto, floor, len(embs)


def clip_style_sim(frames, clip_proto, model, proc):
    if clip_proto is None:
        return None
    e = _clip_embed(frames, model, proc)
    return float(np.dot(e, clip_proto))


def main():
    """Pass 1: score the existing per-epoch sample renders (zero new GPU)."""
    import torch
    ROOT = Path(r"D:/Pinokio/api/basiwan.git")
    ds_imgs = sorted((ROOT / "outputs" / "moral_oral_1" / "images").glob("*.png"))
    if not ds_imgs:
        print("no dataset frames found"); return
    # prompt the per-epoch samples were rendered with (trigger + base)
    base = ("a man in a sweater vest talks to a young boy in a kitchen, "
            "warm interior lighting")
    print(f"[eval] loading DINOv2 + CLIP …", flush=True)
    dmodel, dproc = _load_dino()
    cmodel, cproc = _load_clip()
    print(f"[eval] building style prototype from {len(ds_imgs)} training frames …",
          flush=True)
    proto, floor, n = build_style_prototype([str(p) for p in ds_imgs], dmodel, dproc)
    # [#397/W10] second style axis: CLIP image-image prototype (same training frames).
    clip_proto, clip_floor, _ = build_clip_prototype([str(p) for p in ds_imgs], cmodel, cproc)
    print(f"[eval] prototypes built (n={n}, DINO floor={floor:.3f}, "
          f"CLIP-style floor={clip_floor:.3f})", flush=True)
    rows = []
    for expert in ("moral_orel_high", "moral_orel_low"):
        sdir = ROOT / "outputs" / expert / "sample"
        for mp4 in sorted(sdir.glob("*.mp4")):
            name = mp4.stem
            ep = name.split("_e")[1][:6] if "_e" in name else "?"
            trig = "_00_" in name  # _00 = with trigger, _01 = no-trigger
            frames = _read_frames(mp4, n=8)
            ss = style_sim(frames, proto, dmodel, dproc)
            cs = clip_style_sim(frames, clip_proto, cmodel, cproc)  # [#397] 2nd axis
            ct = _clip_t(frames, (base if not trig else "_moral_orel_ " + base),
                         cmodel, cproc)
            rows.append({"expert": expert, "epoch": ep, "trigger": trig,
                         "style_sim": round(ss, 4),
                         "clip_style": (round(cs, 4) if cs is not None else None),
                         "clip_t": round(ct, 4), "file": mp4.name})
            print(f"  {expert} e{ep} trig={int(trig)} style_sim={ss:.3f} "
                  f"clip_style={cs:.3f} clip_t={ct:.3f}" if cs is not None else
                  f"  {expert} e{ep} trig={int(trig)} style_sim={ss:.3f} clip_t={ct:.3f}",
                  flush=True)
    out = ROOT / "outputs" / "moral_orel_tests" / "eval_pass1.json"
    out.write_text(json.dumps({"floor": floor, "rows": rows}, indent=2))
    # Summary: per-expert style trajectory + bleed (no-trigger vs floor)
    print(f"\n[eval] floor (dataset self-sim) = {floor:.3f}")
    for expert in ("moral_orel_high", "moral_orel_low"):
        er = [r for r in rows if r["expert"] == expert]
        for trig in (True, False):
            tr = sorted([r for r in er if r["trigger"] == trig],
                        key=lambda r: r["epoch"])
            curve = " ".join(f"e{r['epoch'][-2:]}:{r['style_sim']:.2f}" for r in tr)
            lbl = "trigger  " if trig else "NO-trigger"
            print(f"  {expert} {lbl}: {curve}")
    print(f"\n[eval] wrote {out}")


if __name__ == "__main__":
    main()
