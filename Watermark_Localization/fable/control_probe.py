"""Control for the template statistic on official full-frame images: is the low value due to the official
watermark or to the official picture content?

On each official full-frame image we embed OUR OWN watermark with the complemented message on top, and measure
the template statistic for our message. If ours is high (as on COCO) while the official one is low, the official
watermark really differs from a kit embedding of the released picture. Diagnostic only.
usage: control_probe.py <kit> <frame json> <method> [n]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from wm_template import embed, template  # noqa: E402

kit, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 32
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
BANDS = [(1, 5), (5, 17), (17, 65)]


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def z_each(x, t, lo, hi):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    return ((c[:, 0, 0] - flat.mean(1)) / flat.std(1)).cpu().numpy()


def beta(x, t):
    """Least-squares amplitude of the template in the picture, fine band."""
    a, b = box(x, 1) - box(x, 5), box(t, 1) - box(t, 5)
    return ((a * b).flatten(1).sum(1) / (b * b).flatten(1).sum(1)).cpu().numpy() * 255


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
sel = sorted(sel, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:n]
x = images[sel].permute(0, 3, 1, 2).cuda().float() / 255
bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in sel]).cuda()
again = (embed(wm, x, 1 - bits) * 255).round() / 255
rows_out = {"official mark, official picture": (x, template(wm, x, bits)),
            "our mark added on top": (again, template(wm, again, 1 - bits)),
            "official mark after ours": (again, template(wm, again, bits))}
for tag, (img, t) in rows_out.items():
    print("CONTROL", method, f"{tag:34s}", {f"z{lo}-{hi}": round(float(np.median(z_each(img, t, lo, hi))), 2) for lo, hi in BANDS},
          "beta(1-5) median", round(float(np.median(beta(img, t))), 3), flush=True)
zo, zu = z_each(x, template(wm, x, bits), 1, 5), z_each(again, template(wm, again, 1 - bits), 1, 5)
print("CONTROL", method, "per image z(1-5) official:", np.round(zo[:16], 1).tolist(), flush=True)
print("CONTROL", method, "per image z(1-5) ours    :", np.round(zu[:16], 1).tolist(), flush=True)
print("CONTROL_DONE", flush=True)
