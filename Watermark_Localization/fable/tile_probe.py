"""Can the frozen kit decoders read the known message from ONE tile alone, at the watermark's native scale?

Occlusion and random masking ask "what changes when a window is removed", which is weak because the rest of the
frame already saturates the decoder. Here every tile is decoded independently (nothing outside the tile reaches
the network), without rescaling the tile, so a hole tile has no watermark to read at all.
Official train images only, labels used for scoring only.
usage: tile_probe.py <kit> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
R = wm.resolution
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(np.float32)))


def net(x):                                   # x in [0,1], any size, no resizing
    if method == "pixelseal":
        return wm.det(x)[:, 1:].float()
    return wm.net(x * 2 - 1).float()


def auc(s, label):
    s, label = np.asarray(s), np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


stats = {}
with torch.no_grad():
    for i, r in enumerate(rows):
        if r["method"] != method:
            continue
        name = r["image_name"]
        m = masks[name]
        ys, xs = np.where(m > 0.5)
        rect = (m > 0.5).sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min())
        group = "full" if frame[name]["acc"] >= 0.8 else ("rect" if rect else "transplant")
        if group == "rect":
            continue
        sign = torch.tensor([1.0 if c == "1" else -1.0 for c in r["message"]], device="cuda")
        x = images[i].permute(2, 0, 1)[None].float().cuda() / 255
        xr = F.interpolate(x, size=(R, R), mode="bilinear", align_corners=False, antialias=True)
        mr = F.interpolate(torch.from_numpy(m)[None, None].cuda(), size=(R, R), mode="area")
        for t in ((64, 96, 128, 192) if method == "pixelseal" else (32, 48, 64, 96, 128)):
            s = t // 2
            tiles = xr.unfold(2, t, s).unfold(3, t, s).permute(0, 2, 3, 1, 4, 5).reshape(-1, 3, t, t)
            frac = mr.unfold(2, t, s).unfold(3, t, s).mean((-1, -2)).reshape(-1).cpu().numpy()
            for variant in ("native", "resized"):
                inp = tiles if variant == "native" else F.interpolate(tiles, size=(R, R), mode="bilinear", align_corners=False)
                lg = torch.cat([net(inp[k:k + 128]) for k in range(0, len(inp), 128)])
                margin = (lg * sign).mean(1).cpu().numpy()
                acc = ((lg > 0) == (sign > 0)).float().mean(1).cpu().numpy()
                d = stats.setdefault((group, t, variant), {"margin": [], "acc": [], "frac": []})
                d["margin"].append(margin); d["acc"].append(acc); d["frac"].append(frac)
for (group, t, variant), d in sorted(stats.items()):
    margin, acc, frac = (np.concatenate(d[k]) for k in ("margin", "acc", "frac"))
    inside, outside = frac >= 0.9, frac <= 0.1
    sel = inside | outside
    print("TILE", method, group, "tile", t, variant, "n_in", int(inside.sum()), "n_out", int(outside.sum()),
          "bit_acc_in", round(float(acc[inside].mean()), 4), "bit_acc_out", round(float(acc[outside].mean()), 4) if outside.any() else None,
          "AUC_margin", round(auc(margin[sel], inside[sel]), 4), flush=True)
print("TILE_DONE", flush=True)
