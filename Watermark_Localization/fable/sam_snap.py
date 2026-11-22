"""Snap predicted blobs to Segment Anything outlines, evaluated out-of-fold.

The official edits are object-centred, so the true region is often exactly a set of object segments, while the
detector draws blurry blobs. For every connected blob of the predicted minority region (watermark in a clean
picture, hole in a full-frame watermark) we take the union of SAM segments that lie mostly inside the predicted
region and touch the blob; the blob is replaced by that union when the two agree well enough (IoU >= theta).
Two settings (rho = how much of a segment must be predicted, theta) are chosen per regime flag on the OTHER four
folds only and applied to the held-out fold: the reported gain is out-of-sample. Rectangles accepted by the
decoder search are left untouched. SAM never sees a label.
usage: sam_snap.py --folds <run> --train-root <release/train> --rect-train <json> --sam <train.npz> --frame-json <json>
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from assemble import METHODS, TAUS, decide, miou

RHOS = (0.5, 0.6, 0.7, 0.8, 0.9)
THETAS = (0.2, 0.35, 0.5, 0.65, 0.8, 2.0)          # 2.0 = never snap


def load_sam(path):
    out = {}
    with np.load(path) as z:
        names, counts, packed = z["names"].tolist(), z["counts"], z["packed"]
    start = 0
    for n, c in zip(names, counts):
        out[n] = np.unpackbits(packed[start:start + c], axis=1)[:, :128 * 128].reshape(c, 128, 128).astype(bool)
        start += c
    return out


def snap(mask, flag, segs, rho, theta):
    if theta > 1 or len(segs) == 0:
        return mask
    fg = ~mask if flag else mask                     # the region whose outline is refined
    area = segs.reshape(len(segs), -1).sum(1)
    inside = (segs & fg).reshape(len(segs), -1).sum(1) / np.maximum(area, 1)
    good = segs[inside >= rho]
    if len(good) == 0:
        return mask
    labels, count = ndimage.label(fg)
    out = fg.copy()
    for c in range(1, count + 1):
        blob = labels == c
        touching = good[(good & blob).reshape(len(good), -1).any(1)]
        if len(touching) == 0:
            continue
        union = touching.any(0)
        if (union & blob).sum() / (union | blob).sum() >= theta:
            out &= ~blob
            out |= union
    return ~out if flag else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, type=Path)
    ap.add_argument("--train-root", required=True, type=Path)
    ap.add_argument("--rect-train", required=True, type=Path)
    ap.add_argument("--sam", required=True, type=Path)
    ap.add_argument("--frame-json", required=True, type=Path)
    args = ap.parse_args()
    rows = json.loads((args.train_root / "metadata.json").read_text())
    meta = {r["image_name"]: r for r in rows}
    with np.load(args.train_root / "masks_128.npz") as z:
        masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
    frame = json.loads(args.frame_json.read_text())["train"]
    flag = {n: int(frame[n]["acc"] >= 0.8) for n in masks}
    oof, fold_of = {}, {}
    for k in range(5):
        with np.load(args.folds / f"fold_{k}" / "outer_probs.npz") as z:
            for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32)):
                oof[n], fold_of[n] = p, k
    names = sorted(oof)
    rect = json.loads(args.rect_train.read_text())
    sam = load_sam(args.sam)
    names = [n for n in names if n in sam]
    print("SNAP images with SAM segments:", len(names), "median segments", int(np.median([len(sam[n]) for n in names])), flush=True)

    # 1. the current pipeline, exactly as assemble.py: rectangle snapping with nested settings
    from PIL import Image
    gt_box = {}
    for n in names:
        m512 = np.array(Image.open(args.train_root / "masks" / (n + ".png")).convert("L")) > 127
        ys, xs = np.where(m512)
        gt_box[n] = (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    def box_iou(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return ix * iy / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy)

    def calibrate(images):
        good = [n for n in images if rect.get(n) and rect[n]["margin"] >= 3.0 and box_iou(rect[n]["box"], gt_box[n]) > 0.6]
        offset = tuple(np.median([[gt_box[n][k] - rect[n]["box"][k] for k in range(4)] for n in good], 0).tolist()) if len(good) >= 5 else (0.0,) * 4
        tau = max(TAUS, key=lambda t: np.mean([miou(decide(oof[n], rect.get(n), t, offset), masks[n]) for n in images]))
        return tau, offset

    current, rect_used = {}, {}
    for m in METHODS:
        for k in range(5):
            tau, offset = calibrate([n for n in names if meta[n]["method"] == m and fold_of[n] != k])
            for n in names:
                if meta[n]["method"] == m and fold_of[n] == k:
                    current[n] = decide(oof[n], rect.get(n), tau, offset)
                    rect_used[n] = bool(rect.get(n) is not None and rect[n]["margin"] >= tau)
    before = {n: miou(current[n], masks[n]) for n in names}

    # 2. segment snapping, settings chosen per regime flag on the other folds
    table = {}                                   # (rho, theta) -> per-image score
    for rho in RHOS:
        for theta in THETAS:
            table[(rho, theta)] = {n: (before[n] if rect_used[n] else miou(snap(current[n], flag[n], sam[n], rho, theta), masks[n])) for n in names}
    after, chosen = {}, {}
    for f in (0, 1):
        for k in range(5):
            others = [n for n in names if flag[n] == f and fold_of[n] != k]
            best = max(table, key=lambda key: np.mean([table[key][n] for n in others]))
            chosen[f"flag{f}/fold{k}"] = list(best)
            for n in names:
                if flag[n] == f and fold_of[n] == k:
                    after[n] = table[best][n]

    def group(n):
        if flag[n]:
            return "full"
        m = masks[n]; ys, xs = np.where(m)
        return "rect" if m.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "transplant"

    def mean(d, sel):
        v = [d[n] for n in names if sel(n)]
        return round(float(np.mean(v)), 4) if v else None

    print("SNAP out-of-fold mIoU: before", mean(before, lambda n: True), "after", mean(after, lambda n: True), flush=True)
    print("SNAP by fold:", [(mean(before, lambda n, k=k: fold_of[n] == k), mean(after, lambda n, k=k: fold_of[n] == k)) for k in range(5)], flush=True)
    for m in METHODS:
        print("SNAP", m, "before/after", mean(before, lambda n: meta[n]["method"] == m), mean(after, lambda n: meta[n]["method"] == m),
              {g: (mean(before, lambda n: meta[n]["method"] == m and group(n) == g), mean(after, lambda n: meta[n]["method"] == m and group(n) == g))
               for g in ("full", "rect", "transplant")}, flush=True)
    print("SNAP settings per fold:", chosen, flush=True)
    diff = np.array([after[n] - before[n] for n in names])
    print("SNAP images improved", int((diff > 1e-9).sum()), "worse", int((diff < -1e-9).sum()), "unchanged", int((np.abs(diff) <= 1e-9).sum()),
          "| best fixed setting (in-sample, for information only):", max(((float(np.mean(list(v.values()))), k) for k, v in table.items()))[::-1], flush=True)


if __name__ == "__main__":
    main()
