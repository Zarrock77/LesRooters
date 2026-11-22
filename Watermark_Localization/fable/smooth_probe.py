"""Does smoothing the probability map before thresholding help, per method and regime? (nested, out-of-fold)

For every outer fold the pair (Gaussian sigma in cells, threshold) is chosen per method and regime flag on the
out-of-fold maps of the four OTHER folds and applied to the held-out fold. Compared with the same nested choice
restricted to sigma = 0 (threshold only). Also tested: removing predicted minority components smaller than a
minimum size chosen the same way. Plain maps only (before rectangles and segment selection).
usage: smooth_probe.py <kit> <run dir> <frame json> <reference dir>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

kit, run, frame_json, reference = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
frame = json.loads(frame_json.read_text())["train"]
meta = {r["image_name"]: r["method"] for r in json.loads((kit / "release/train/metadata.json").read_text())}
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
with np.load(run / "oof_probs.npz") as z:
    names, probs = z["names"].tolist(), z["probs"].astype(np.float32)
folds = json.loads((reference / "protocol.json").read_text())["folds"]
fold_of = {n: k for k, f in enumerate(folds) for n in f}
gt = np.stack([masks[n] for n in names])
group = np.array([f"{meta[n]}/{int(frame[n]['acc'] >= 0.8)}" for n in names])
fold = np.array([fold_of[n] for n in names])
SIGMAS = (0.0, 0.7, 1.2, 2.0, 3.0)
THRS = (0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
SIZES = (0, 4, 9, 16, 30)


def miou(pred, g):
    out = np.zeros(len(pred))
    for c in (False, True):
        p, t = pred == c, g == c
        inter, union = (p & t).sum((1, 2)), (p | t).sum((1, 2))
        out += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


def clean(pred, size):
    if size == 0:
        return pred
    out = pred.copy()
    for i in range(len(pred)):
        for value in (0, 1):                                  # drop small islands of either class
            n, lab, stats, _ = cv2.connectedComponentsWithStats((pred[i] == value).astype(np.uint8), connectivity=4)
            for c in range(1, n):
                if stats[c, cv2.CC_STAT_AREA] < size:
                    out[i][lab == c] = not value
    return out


smoothed = {s: (probs if s == 0 else np.stack([cv2.GaussianBlur(p, (0, 0), s) for p in probs])) for s in SIGMAS}
scores = {}                                                    # (sigma, thr, size) -> per-image score
for s in SIGMAS:
    for t in THRS:
        base = smoothed[s] >= t
        for size in SIZES:
            if size and s not in (0.0, 1.2):
                continue
            scores[(s, t, size)] = miou(clean(base, size), gt)
print("SMOOTH settings evaluated", len(scores), flush=True)


def nested(allowed):
    out = np.zeros(len(names))
    chosen = {}
    for k in range(5):
        for g in np.unique(group):
            fit, held = (fold != k) & (group == g), (fold == k) & (group == g)
            best = max(allowed, key=lambda key: scores[key][fit].mean())
            out[held] = scores[best][held]
            chosen.setdefault(g, []).append(best)
    return out, chosen


base, _ = nested([k for k in scores if k[0] == 0.0 and k[2] == 0])
for label, allowed in (("smoothing", [k for k in scores if k[2] == 0]), ("small components", [k for k in scores if k[0] == 0.0]),
                       ("both", list(scores))):
    got, chosen = nested(allowed)
    d = got - base
    rng = np.random.default_rng(0)
    bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)])
    print("SMOOTH", label, "| nested out-of-fold", round(float(got.mean()), 4), "against threshold only", round(float(base.mean()), 4),
          "| paired gain", round(float(d.mean()), 4), "95 % interval", [round(float(bs[75]), 4), round(float(bs[2925]), 4)],
          "| by fold", [round(float(d[fold == k].mean()), 4) for k in range(5)], flush=True)
    print("   by group", {g: round(float(d[group == g].mean()), 4) for g in np.unique(group)}, flush=True)
    print("   chosen in fold 0", {g: v[0] for g, v in chosen.items()}, flush=True)
print("SMOOTH_DONE", flush=True)
