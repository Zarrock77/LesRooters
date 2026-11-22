"""Full-frame pictures whose predicted mask is uniform (no hole found): is a forced smallest hole better? (nested)

A full-frame picture always has a restored part in the train set, so "all watermarked" scores about 0.49. Here the
x % least watermarked cells of the smoothed map are declared a hole when the map gives none. x is chosen per method
on the four other folds among {0 (do nothing), 0.3, 0.6, 1, 2, 3, 5 %}, with two smoothings, and applied to the
held-out fold. Calibrated maps before rectangles and segment selection (full-frame pictures are not touched by
either). Reported: how many train pictures have a uniform truth, how many predictions are uniform, nested gain.
usage: uniform_probe.py <kit> <run dir> <frame json> <reference dir>
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
method = np.array([meta[n] for n in names])
flag = np.array([int(frame[n]["acc"] >= 0.8) for n in names])
fold = np.array([fold_of[n] for n in names])
print("UNIFORM truth: train pictures with a uniform mask", int(((gt.mean((1, 2)) == 0) | (gt.mean((1, 2)) == 1)).sum()),
      "| smallest hole share among full-frame pictures", round(float((1 - gt[flag == 1].mean((1, 2))).min()), 5), flush=True)


def miou(pred, g):
    s = 0.0
    for c in (False, True):
        p, t = pred == c, g == c
        u = (p | t).sum()
        s += 1.0 if u == 0 else (p & t).sum() / u
    return s / 2


pred = probs > 0.5
target = np.nonzero((flag == 1) & pred.all((1, 2)))[0]                 # full-frame, predicted "all watermarked"
print("UNIFORM predictions: full-frame pictures predicted all watermarked", len(target), "by method",
      {m: int((method[target] == m).sum()) for m in ("pixelseal", "trustmark", "stegastamp")},
      "| their mean score", round(float(np.mean([miou(pred[i], gt[i]) for i in target])), 4), flush=True)
SHARES = (0.0, 0.003, 0.006, 0.01, 0.02, 0.03, 0.05)
SIGMAS = (1.0, 2.5)
score = {}
for sg in SIGMAS:
    for x in SHARES:
        out = np.zeros(len(target))
        for j, i in enumerate(target):
            if x == 0:
                out[j] = miou(pred[i], gt[i])
                continue
            s = cv2.GaussianBlur(probs[i], (0, 0), sg)
            k = max(1, int(round(x * s.size)))
            out[j] = miou(s > np.partition(s.ravel(), k - 1)[k - 1], gt[i])
        score[(sg, x)] = out
base = score[(SIGMAS[0], 0.0)]
got, chosen = base.copy(), {}
for k in range(5):
    for m in ("pixelseal", "trustmark", "stegastamp"):
        fit = (fold[target] != k) & (method[target] == m)
        held = (fold[target] == k) & (method[target] == m)
        if fit.sum() < 5 or not held.any():
            continue
        best = max(score, key=lambda key: score[key][fit].mean())
        if score[best][fit].mean() - base[fit].mean() < 0.01:
            best = (SIGMAS[0], 0.0)
        got[held] = score[best][held]
        chosen.setdefault(m, []).append(best)
d = got - base
rng = np.random.default_rng(0)
bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]) if len(d) else np.zeros(4000)
print("UNIFORM nested gain on these pictures", round(float(d.mean()), 4), "95 %", [round(float(bs[100]), 4), round(float(bs[3900]), 4)],
      "| better", int((d > 1e-9).sum()), "worse", int((d < -1e-9).sum()), "| gain over all 510 pictures", round(float(d.sum() / 510), 4), flush=True)
print("UNIFORM chosen", chosen, flush=True)
for key in sorted(score):
    print("   fixed", key, "mean on these pictures", round(float(score[key].mean()), 4), flush=True)
