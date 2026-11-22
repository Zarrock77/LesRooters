"""How well could SAM segments reproduce the true masks? (upper bound, uses labels: information only)

For each official train image with SAM proposals: the union of segments that lie at least 60 % inside the true
minority region (watermark for clean pictures, hole for full-frame pictures), scored with the official mIoU.
If this ceiling is far above the current predictions, selecting segments with a learned classifier is worth
building; if not, object outlines are not what limits us.
usage: sam_oracle.py <kit> <frame json> <run dir with fold_k/outer_probs.npz> <sam npz> [<sam npz> ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

from assemble import METHODS, miou
from sam_snap import load_sam

kit, frame_json, run = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sam = {}
for p in sys.argv[4:]:
    if Path(p).exists():
        sam.update(load_sam(p))
rows = json.loads((kit / "release/train/metadata.json").read_text())
meta = {r["image_name"]: r for r in rows}
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
frame = json.loads(frame_json.read_text())["train"]
oof = {}
for k in range(5):
    with np.load(run / f"fold_{k}" / "outer_probs.npz") as z:
        oof.update(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
res = {}
for n, segs in sam.items():
    gt, flag = masks[n], frame[n]["acc"] >= 0.8
    fg = ~gt if flag else gt
    ys, xs = np.where(gt)
    group = "full" if flag else ("rect" if gt.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "transplant")
    area = segs.reshape(len(segs), -1).sum(1)
    inside = (segs & fg).reshape(len(segs), -1).sum(1) / np.maximum(area, 1)
    union = segs[inside >= 0.6].any(0) if (inside >= 0.6).any() else np.zeros_like(gt)
    oracle = ~union if flag else union
    covered = float((union & fg).sum() / max(fg.sum(), 1))
    res.setdefault((meta[n]["method"], group), []).append((miou(oof[n] > 0.5, gt), miou(oracle, gt), covered))
print("ORACLE images", len(sam), flush=True)
for group in ("full", "rect", "transplant"):
    for m in METHODS:
        v = np.array(res.get((m, group), []))
        if len(v):
            print("ORACLE", f"{m:10s}", f"{group:10s}", "n", len(v), "| current (threshold only)", round(float(v[:, 0].mean()), 3),
                  "| SAM-union ceiling", round(float(v[:, 1].mean()), 3), "| share of true region covered by pure segments", round(float(v[:, 2].mean()), 3), flush=True)
allv = np.concatenate([np.array(v) for v in res.values()])
print("ORACLE all: current", round(float(allv[:, 0].mean()), 4), "ceiling", round(float(allv[:, 1].mean()), 4),
      "| best of both per image", round(float(np.maximum(allv[:, 0], allv[:, 1]).mean()), 4), flush=True)
