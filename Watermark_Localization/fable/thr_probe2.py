"""Nested decision cut per (method, flag, predicted size of the minority region), against per (method, flag).
Same margin rule as calibrate_thr.py (keep 0.5 unless the cut beats it by 0.004 on the other folds). Plain maps.
usage: thr_probe2.py <kit> <run dir> <frame json> <reference dir>
"""
import json
import sys
from pathlib import Path

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
soft = probs.mean((1, 2))
minority = np.where(flag == 1, 1 - soft, soft)                       # predicted share of the minority region (label-free)
THRS = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
MARGIN = 0.004


def miou(pred, g):
    out = np.zeros(len(pred))
    for c in (False, True):
        p, t = pred == c, g == c
        inter, union = (p & t).sum((1, 2)), (p | t).sum((1, 2))
        out += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


score = {t: miou(probs > t, gt) for t in THRS}
base = score[0.5]
rng = np.random.default_rng(0)
results = {}
for label, edges in (("method x flag", ()), ("x 2 sizes (3 %)", (0.03,)), ("x 3 sizes (2 %, 10 %)", (0.02, 0.10))):
    size = np.digitize(minority, edges)
    group = np.array([f"{m}/{f}/{s}" for m, f, s in zip(method, flag, size)])
    got, chosen = np.zeros(len(names)), {}
    for k in range(5):
        for g in np.unique(group):
            fit, held = (fold != k) & (group == g), (fold == k) & (group == g)
            if fit.sum() < 12:
                best = 0.5
            else:
                best = max(THRS, key=lambda t: score[t][fit].mean())
                if score[best][fit].mean() - base[fit].mean() < MARGIN:
                    best = 0.5
            got[held] = score[best][held]
            chosen.setdefault(g, []).append(best)
    results[label] = got
    d = got - base
    bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    print("THR2", label, "| nested", round(float(got.mean()), 4), "fixed 0.5", round(float(base.mean()), 4), "| gain", round(float(d.mean()), 4),
          "95 %", [round(float(bs[100]), 4), round(float(bs[3900]), 4)], "| by fold", [round(float(d[fold == k].mean()), 4) for k in range(5)],
          "| by method", {m: round(float(d[method == m].mean()), 4) for m in ("pixelseal", "trustmark", "stegastamp")}, flush=True)
    print("   cuts", {g: v for g, v in chosen.items() if any(x != 0.5 for x in v)}, flush=True)
print("THR2_DONE", flush=True)
