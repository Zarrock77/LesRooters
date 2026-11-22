"""Nested per-(method, regime flag) thresholds against the fixed 0.5 used by assemble.py. Plain maps, out-of-fold.
usage: thr_probe.py <kit> <run dir> <frame json> <reference dir>
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
THRS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)


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
for label, group in (("method x flag", np.array([f"{m}/{f}" for m, f in zip(method, flag)])), ("flag only", flag.astype(str)), ("method only", method)):
    got, chosen = np.zeros(len(names)), {}
    for k in range(5):
        for g in np.unique(group):
            fit, held = (fold != k) & (group == g), (fold == k) & (group == g)
            best = max(THRS, key=lambda t: score[t][fit].mean())
            got[held] = score[best][held]
            chosen.setdefault(g, []).append(best)
    d = got - base
    bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)])
    print("THR", label, "| nested", round(float(got.mean()), 4), "fixed 0.5", round(float(base.mean()), 4), "| paired gain", round(float(d.mean()), 4),
          "95 % interval", [round(float(bs[100]), 4), round(float(bs[3900]), 4)], "| by fold", [round(float(d[fold == k].mean()), 4) for k in range(5)], flush=True)
    print("   by group", {g: (round(float(d[group == g].mean()), 4), chosen[g]) for g in np.unique(group)}, flush=True)
print("THR_DONE", flush=True)
