"""Was the official full-frame picture watermarked at a SMALLER size and then enlarged?

The starter kit says full-frame pictures are "the whole image watermarked and a part restored to the clean
original", made with exactly the kit embedders. Their decoder margin is higher than a fresh 512 px embedding
(TrustMark 1.1x) while the kit template computed at 512 px does not match. Embedding at the embedder's own
resolution and enlarging the result would raise the margin. Here: embed at 256 / 384 / 128 px, enlarge to 512 with
several filters, then decoder margin and template match (template computed on the 512 px picture as before).
usage: upscale_probe.py <kit> <dir with fresh_<method>.npz> <method>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
kit, folder, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402
from wm_template import template  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
BANDS = ((1, 5), (5, 17), (17, 65))


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def signature(tag, images_uint8, bits, ref=None):
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).cuda().float() / 255
    b = torch.from_numpy(bits).cuda()
    lg = wm.logits(x).float().cpu()
    sign = torch.from_numpy(bits).float() * 2 - 1
    margin = float((lg * sign).mean())
    t = template(wm, x, b)
    res = {"margin": margin}
    for lo, hi in BANDS:
        a, c = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
        res[f"{lo}-{hi}"] = float(((a * c).flatten(1).sum(1) / (c * c).flatten(1).sum(1)).median())
    line = {"margin": round(margin, 2)}
    if ref:
        line.update({"margin vs fresh": round(margin / ref["margin"], 2), **{f"template {lo}-{hi}": round(res[f"{lo}-{hi}"] / ref[f"{lo}-{hi}"], 2) for lo, hi in BANDS}})
    print("UPSCALE", method, f"{tag:34s}", line, flush=True)
    return res


with np.load(folder / f"fresh_{method}.npz") as z:
    S, W, bits = z["S"][:24], z["W"][:24], z["bits"][:24].astype(np.int64)
ref = signature("fresh at 512", W, bits)
FILTERS = {"nearest": Image.Resampling.NEAREST, "bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC, "lanczos": Image.Resampling.LANCZOS}
for small in (256, 384, 128):
    low = [np.asarray(Image.fromarray(s).resize((small, small), Image.Resampling.LANCZOS)) for s in S]
    marked = [wm.embed(l, b.tolist()) for l, b in zip(low, bits)]
    for name, f in FILTERS.items():
        up = np.stack([np.asarray(Image.fromarray(m).resize((512, 512), f)) for m in marked])
        signature(f"embed at {small}, {name} to 512", up, bits, ref)
print("UPSCALE_DONE (official: TrustMark margin 1.1, template 0.08 / 0.09 / 0.49; PixelSeal 0.89, about 0 / 0.2 / 0.33)", flush=True)
