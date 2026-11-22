"""Where do the remaining errors of a run come from? (out-of-fold, information only)

For every method and regime: share of pictures whose thresholded map is uniform (nothing found), their score,
and for the others the IoU of the minority class by size of the true minority region.
usage: error_profile.py <kit> <frame json> <run dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

from assemble import METHODS, miou

kit, frame_json, run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
rows = json.loads((kit / "release/train/metadata.json").read_text())
meta = {r["image_name"]: r for r in rows}
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
frame = json.loads(frame_json.read_text())["train"]
oof = {}
for k in range(5):
    with np.load(run / f"fold_{k}" / "outer_probs.npz") as z:
        oof.update(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))


def group(n):
    if frame[n]["acc"] >= 0.8:
        return "full"
    m = masks[n]; ys, xs = np.where(m)
    return "rect" if m.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "transplant"


for m in METHODS:
    for g in ("full", "rect", "transplant"):
        names = [n for n in oof if meta[n]["method"] == m and group(n) == g]
        rec = []
        for n in names:
            gt, pred = masks[n], oof[n] > 0.5
            minority = ~gt if g == "full" else gt
            pmin = ~pred if g == "full" else pred
            iou_min = (minority & pmin).sum() / max((minority | pmin).sum(), 1)
            rec.append((miou(pred, gt), float(minority.mean()), bool(pmin.sum() == 0), iou_min, float(pmin.mean())))
        r = np.array(rec, float)
        empty = r[:, 2] > 0
        small, mid, big = r[:, 1] < 0.02, (r[:, 1] >= 0.02) & (r[:, 1] < 0.1), r[:, 1] >= 0.1
        f = lambda sel, col: round(float(r[sel, col].mean()), 3) if sel.any() else None  # noqa: E731
        print("PROFILE", f"{m:10s}", f"{g:10s}", "n", len(r), "mIoU", round(float(r[:, 0].mean()), 3), "| nothing found:", int(empty.sum()),
              "pictures, mIoU", f(empty, 0), "| found: mIoU", f(~empty, 0), "minority IoU", f(~empty, 3),
              "| minority IoU by true size <2%:", f(small, 3), f"(n={int(small.sum())})", "2-10%:", f(mid, 3), f"(n={int(mid.sum())})",
              ">10%:", f(big, 3), f"(n={int(big.sum())})", "| predicted/true minority area", round(float(r[:, 4].mean() / max(r[:, 1].mean(), 1e-9)), 2), flush=True)
