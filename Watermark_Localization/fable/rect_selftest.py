"""Check the rectangle snapping in isolation on the official train images (no detector involved).

Start from the true mask bounding box moved by a random error on each side (as a detector blob would be) and
measure whether the decoder-guided search recovers the true rectangle. Also records the margin reached on
transplanted objects, where no rectangle frame exists, to choose an acceptance threshold.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rect_refine import MAX_ASPECT, METHODS, Scorer, refine  # noqa: E402

kit, out = Path(sys.argv[1]), Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
rows = json.loads((kit / "release/train/metadata.json").read_text())
rng = np.random.default_rng(20261121)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


records, t0 = [], time.time()
for method in METHODS:
    scorer = Scorer(kit, method, device)
    for r in rows:
        if r["method"] != method or r["full_watermark"]:
            continue
        name = r["image_name"]
        m = np.array(Image.open(kit / "release/train/masks" / (name + ".png")).convert("L")) > 127
        ys, xs = np.where(m)
        gt = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))
        w, h = gt[2] - gt[0], gt[3] - gt[1]
        pure = bool(m.sum() / (w * h) > 0.999)
        image = torch.from_numpy(np.array(Image.open(kit / "release/train/images" / (name + ".png")).convert("RGB")))
        image = image.permute(2, 0, 1)[None].float().div(255).to(device)
        sign = torch.tensor([int(b) for b in r["message"]], device=device, dtype=torch.float32) * 2 - 1
        rec = {"name": name, "method": method, "pure": pure, "w": w, "h": h}
        for jitter in (0.0, 0.1, 0.2, 0.35):
            e = rng.uniform(-jitter, jitter, 4)
            init = (gt[0] + e[0] * w, gt[1] + e[1] * h, gt[2] + e[2] * w, gt[3] + e[3] * h)
            res = refine(scorer, image, sign, init, MAX_ASPECT[method] * 1.5)
            if res is None:
                rec[f"j{jitter}"] = None
                continue
            rec[f"j{jitter}"] = {"iou": box_iou(res["box"], gt), "init_iou": box_iou(init, gt) if jitter else 1.0,
                                 "margin": res["margin"], "acc": res["acc"], "init_margin": res["init_margin"],
                                 "delta": [res["box"][k] - gt[k] for k in range(4)]}
        records.append(rec)
    print("SELFTEST", method, round(time.time() - t0, 1), "s", flush=True)
    del scorer
    torch.cuda.empty_cache()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(records))
for method in METHODS:
    for pure in (True, False):
        sub = [r for r in records if r["method"] == method and r["pure"] == pure]
        print(f"\n{method} {'pure rectangles' if pure else 'transplanted objects'} n={len(sub)}")
        for jitter in (0.0, 0.1, 0.2, 0.35):
            v = [r[f"j{jitter}"] for r in sub if r[f"j{jitter}"]]
            iou = np.array([x["iou"] for x in v]); mg = np.array([x["margin"] for x in v]); ii = np.array([x["init_iou"] for x in v])
            line = f"  jitter {jitter}: init IoU {ii.mean():.3f} -> final IoU mean {iou.mean():.3f} median {np.median(iou):.3f} frac>0.9 {(iou > .9).mean():.2f} | margin mean {mg.mean():.2f} p10 {np.quantile(mg, .1):.2f} p50 {np.median(mg):.2f} p90 {np.quantile(mg, .9):.2f} max {mg.max():.2f}"
            if pure:
                d = np.array([x["delta"] for x in v if x["iou"] > 0.8])
                if len(d):
                    line += f" | side offsets (pred-gt) median {np.round(np.median(d, 0), 1).tolist()}"
            print(line)
print("SELFTEST_DONE", flush=True)
