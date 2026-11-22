"""Does the message template match inside officially transplanted objects (and inside official rectangles),
when it is computed in the FULL frame of the released picture?

Full-frame official pictures match the kit template only weakly (re-rendered), official rectangles match it in
their own frame. If transplanted objects kept their place and pixels, the full-frame template correlates inside
the object and a message-aware evidence can be pooled per segment. Labels are used for measuring only.
usage: transplant_probe.py <kit> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from wm_template import template  # noqa: E402

kit, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def products(x, t):
    a, b = box(x, 1) - box(x, 5), box(t, 1) - box(t, 5)
    return (a * b).mean(1), (a * a).mean(1), (b * b).mean(1)


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
res = {}
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] < 0.8]
for s in range(0, len(sel), 12):
    part = sel[s:s + 12]
    x = images[part].permute(0, 3, 1, 2).cuda().float() / 255
    bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in part]).cuda()
    ab, aa, bb = products(x, template(wm, x, bits))
    ab2, aa2, bb2 = products(x, template(wm, x, torch.randint_like(bits, 2)))
    for j, i in enumerate(part):
        m = masks[rows[i]["image_name"]]
        ys, xs = np.where(m)
        group = "rect" if m.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "transplant"
        m512 = torch.from_numpy(np.kron(m, np.ones((4, 4), bool))).cuda()
        if m512.sum() < 400 or (~m512).sum() < 400:
            continue

        def ncc(p, q, r, region):
            return float(p[j][region].mean() / (q[j][region].mean() * r[j][region].mean() + 1e-14).sqrt())
        res.setdefault(group, []).append((ncc(ab, aa, bb, m512), ncc(ab, aa, bb, ~m512), ncc(ab2, aa2, bb2, m512), float(m.mean())))
for group, v in res.items():
    v = np.array(v)
    print("TRANSPLANT", method, group, "n", len(v), "| correlation inside the true region: median", round(float(np.median(v[:, 0])), 4),
          "mean", round(float(v[:, 0].mean()), 4), "| outside:", round(float(np.median(v[:, 1])), 4), "| inside with a wrong message:", round(float(np.median(v[:, 2])), 4),
          "| share of pictures with inside > outside:", round(float((v[:, 0] > v[:, 1]).mean()), 3), flush=True)
print("TRANSPLANT_DONE (for reference: our own fresh embedding gives about 0.048, official full-frame about 0.007)", flush=True)
