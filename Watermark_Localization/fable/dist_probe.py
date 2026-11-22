"""Where, relative to the edited hole, does the official full-frame watermark still match the kit template?

If the editing tool re-rendered a context box around each hole (crop, inpaint, paste the crop back), the kept
pixels near a hole are no longer the pristine embedding while pixels far away are. Then the local template
correlation of the OFFICIAL watermark rises with the distance to the hole, while our own watermark embedded on
top of the same picture stays flat. Labels are used to measure distances only. Diagnostic, nothing is fitted.
usage: dist_probe.py <kit> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from wm_template import embed, template  # noqa: E402

kit, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
EDGES = [0, 8, 16, 32, 64, 128, 1000]


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def local_ncc(x, t, win=33):
    a, b = box(x, 1) - box(x, 5), box(t, 1) - box(t, 5)
    ab, aa, bb = (box((u * v).mean(1, keepdim=True), win) for u, v in ((a, b), (a, a), (b, b)))
    return (ab / (aa * bb + 1e-14).sqrt())[:, 0]


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
acc = {tag: [np.zeros(len(EDGES)), np.zeros(len(EDGES))] for tag in ("official", "ours")}       # sums, counts; last bin = inside hole
per_image = []
for s in range(0, len(sel), 12):
    part = sel[s:s + 12]
    x = images[part].permute(0, 3, 1, 2).cuda().float() / 255
    bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in part]).cuda()
    again = (embed(wm, x, 1 - bits) * 255).round() / 255
    maps = {"official": local_ncc(x, template(wm, x, bits)).cpu().numpy(),
            "ours": local_ncc(again, template(wm, again, 1 - bits)).cpu().numpy()}
    for j, i in enumerate(part):
        hole = ~np.kron(masks[rows[i]["image_name"]], np.ones((4, 4), bool))
        if not hole.any():
            continue
        dist = ndimage.distance_transform_edt(~hole)
        row = {}
        for tag, m in maps.items():
            for k in range(len(EDGES) - 1):
                selk = (dist > EDGES[k]) & (dist <= EDGES[k + 1])
                acc[tag][0][k] += m[j][selk].sum(); acc[tag][1][k] += selk.sum()
            acc[tag][0][-1] += m[j][hole].sum(); acc[tag][1][-1] += hole.sum()
            far = dist > 64
            row[tag] = (float(m[j][(dist > 0) & (dist <= 32)].mean()), float(m[j][far].mean()) if far.any() else float("nan"))
        per_image.append((float(hole.mean()), row))
for tag in acc:
    means = acc[tag][0] / np.maximum(acc[tag][1], 1)
    print("DIST", method, f"{tag:9s}", "mean local correlation by distance to the hole (px):",
          {f"{EDGES[k]}-{EDGES[k + 1]}": round(float(means[k]), 4) for k in range(len(EDGES) - 1)}, "| inside hole", round(float(means[-1]), 4), flush=True)
per_image.sort(key=lambda p: p[0])
for hole_share, row in per_image[::max(1, len(per_image) // 14)]:
    print("DIST", method, "hole share", round(hole_share, 3), "official near/far", tuple(round(v, 3) for v in row["official"]),
          "ours near/far", tuple(round(v, 3) for v in row["ours"]), flush=True)
print("DIST_DONE", flush=True)
