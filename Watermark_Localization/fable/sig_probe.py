"""Signatures of re-rendered fresh embeddings versus official full-frame images.

For each variant in <dir>/rerender_<method>.npz (made from <dir>/fresh_<method>.npz): decoder bit accuracy and
margin, and how well the picture still matches the message template (fine band; ratio to the fresh embedding).
The same numbers for official full-frame images (smallest holes) are printed first. A variant that reproduces
all of them for the three methods is a candidate for what the editing tool did to the kept pixels.
usage: sig_probe.py <kit> <dir> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, folder, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
if method != "stegastamp":
    from wm_template import template as fast_template  # noqa: E402


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def tmpl(images_uint8, bits):
    if method != "stegastamp":
        x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).cuda().float() / 255
        return fast_template(wm, x, torch.from_numpy(bits).cuda()).cpu()
    out = []
    for im, b in zip(images_uint8, bits):
        good = wm.embed(im, b.tolist()).astype(np.float32); bad = wm.embed(im, (1 - b).tolist()).astype(np.float32)
        out.append((good - bad) / 2)
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2)


def signature(tag, images_uint8, bits, ref=None):
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255
    lg = torch.cat([torch.as_tensor(wm.logits(x[s:s + 8])).float().cpu() for s in range(0, len(x), 8)])
    sign = torch.from_numpy(bits).float() * 2 - 1
    acc, margin = float(((lg > 0) == (sign > 0)).float().mean()), float((lg * sign).mean())
    t = tmpl(images_uint8, bits)
    res = {}
    for lo, hi in ((1, 5), (5, 17)):
        a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
        res[f"beta{lo}-{hi}"] = float(((a * b).flatten(1).sum(1) / (b * b).flatten(1).sum(1)).median())
    text = {k: round(v, 3) for k, v in res.items()}
    if ref:
        text.update({k + "_vs_fresh": round(res[k] / ref[k], 2) for k in res})
    print("SIG", method, f"{tag:26s}", "bit acc", round(acc, 4), "margin", round(margin, 2), "|", text, flush=True)
    return res


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
sel = sorted(sel, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:24]
with np.load(folder / f"fresh_{method}.npz") as z:
    W, bits = z["W"][:24], z["bits"][:24].astype(np.int64)
ref = signature("fresh embedding", W, bits)
signature("OFFICIAL (smallest holes)", images[sel].numpy(), np.array([[int(c) for c in rows[i]["message"]] for i in sel]), ref)
with np.load(folder / f"rerender_{method}.npz") as z:
    for name in z.files:
        signature(name, z[name], bits, ref)
print("SIG_DONE", flush=True)
