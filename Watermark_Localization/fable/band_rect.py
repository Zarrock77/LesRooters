"""Template correlation inside official RECTANGLE images, in the rectangle's own frame (labels locate the box).

If the template matches there (z as on self-embedded pictures) but not on official full-frame images, the
full-frame pictures were altered globally after embedding; if it matches nowhere, the official embedding itself
differs from the kit embedder (strength, variant, frame). Diagnostic only.
usage: band_rect.py <kit> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wm_template import embed, template  # noqa: E402

kit, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
BANDS = [(1, 5), (5, 17), (17, 65)]


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def z0(x, t, lo, hi):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    return float(((c[:, 0, 0] - flat.mean(1)) / flat.std(1)).median())


rows = json.loads((kit / "release/train/metadata.json").read_text())
rng = np.random.default_rng(4)
records = []
for r in rows:
    if r["method"] != method or r["full_watermark"]:
        continue
    m = np.array(Image.open(kit / "release/train/masks" / (r["image_name"] + ".png")).convert("L")) > 127
    ys, xs = np.where(m)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    if m.sum() != (y1 - y0) * (x1 - x0) or min(x1 - x0, y1 - y0) < 96:
        continue
    img = np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB"))
    crop = torch.from_numpy(img[y0:y1, x0:x1].copy()).permute(2, 0, 1)[None].cuda().float() / 255
    bits = torch.tensor([[int(c) for c in r["message"]]]).cuda()
    acc = float(((wm.logits(crop) > 0) == (bits > 0)).float().mean())
    t = template(wm, crop, bits)
    tw = template(wm, crop, torch.randint_like(bits, 2))
    # control: the same crop content, watermark re-embedded by us on top (upper bound for this content and size)
    again = (embed(wm, crop, 1 - bits) * 255).round() / 255
    rec = {"size": (int(x1 - x0), int(y1 - y0)), "acc": acc}
    for lo, hi in BANDS:
        rec[f"z{lo}-{hi}"] = z0(crop, t, lo, hi)
        rec[f"wrong{lo}-{hi}"] = z0(crop, tw, lo, hi)
        rec[f"ours{lo}-{hi}"] = z0(again, template(wm, again, 1 - bits), lo, hi)
    records.append(rec)
    print("RECT", method, rec["size"], "bit acc", round(acc, 3), {k: round(v, 2) for k, v in rec.items() if k[0] in "zwo"}, flush=True)
if records:
    print("RECT_SUMMARY", method, "n", len(records), "median bit acc", round(float(np.median([r["acc"] for r in records])), 3),
          {k: round(float(np.median([r[k] for r in records])), 2) for k in records[0] if k[0] in "zwo"}, flush=True)
print("RECT_DONE", flush=True)
