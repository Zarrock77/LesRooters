"""Build a folds-like directory that takes each watermark method from one run (same frozen folds).

usage: combine_methods.py <runs dir> <out name> pixelseal=<run>[+<run>...] trustmark=... stegastamp=...
A candidate can be one run or an average of runs written runA*runB (mean of their probabilities, still
out-of-fold). When several candidates are given for a method, the one with the best out-of-fold mIoU is kept. That is a small selection on the development folds (a few candidates per method) and is recorded in
combination.json together with every candidate's score.
Per fold k: outer_probs.npz and test_probs.npz are rebuilt image by image from the run chosen for the image's method.
"""
import json
import sys
from pathlib import Path

import numpy as np

runs, out_name = Path(sys.argv[1]), sys.argv[2]
candidates = {a.split("=")[0]: a.split("=")[1].split("+") for a in sys.argv[3:]}
kit = Path("/srv/lesrooters/Hackathon_Watermark_Localisation_Task")
method_train = {r["image_name"]: r["method"] for r in json.loads((kit / "release/train/metadata.json").read_text())}
method_test = {r["image_name"]: r["method"] for r in json.loads((kit / "release/test/metadata.json").read_text())}
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def miou(pred, gt):
    s = 0.0
    for c in (0, 1):
        p, g = pred == c, gt == c
        u = (p | g).sum()
        s += 1.0 if u == 0 else (p & g).sum() / u
    return s / 2


def load_probs(candidate, k, which):
    """Mean probability over the runs of a candidate for fold k ('outer' or 'test')."""
    acc, names = None, None
    for run in candidate.split("*"):
        with np.load(runs / run / f"fold_{k}" / f"{which}_probs.npz") as z:
            n, p = z["names"].tolist(), z["probs"].astype(np.float32)
        assert names is None or names == n
        names, acc = n, (p if acc is None else acc + p)
    return names, acc / len(candidate.split("*"))


frame = json.loads((runs / "frame" / "frame_decode.json").read_text())["train"]
CUTS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8)
MARGIN = 0.004


def oof_scores(candidate, method):
    """(out-of-fold mIoU at the fixed 0.5 cut, the same with the cut chosen per regime flag on the other folds).

    The final decision cut is calibrated per regime (calibrate_thr.py). Ranking candidates at 0.5 would prefer maps
    that happen to be calibrated at 0.5 over maps that rank the cells better, so the ranking uses the same nested
    rule as the calibration: per regime flag, the cut is chosen on the four other folds and kept at 0.5 unless it
    gains MARGIN there."""
    fold, flag, rows = [], [], []
    for k in range(5):
        names, probs = load_probs(candidate, k, "outer")
        for n, p in zip(names, probs):
            if method_train[n] == method:
                fold.append(k); flag.append(int(frame[n]["acc"] >= 0.8)); rows.append([miou(p > c, masks[n]) for c in CUTS])
    fold, flag, rows = np.array(fold), np.array(flag), np.array(rows)
    half = CUTS.index(0.5)
    nested = np.zeros(len(rows))
    for k in range(5):
        for f in (0, 1):
            fit, held = (fold != k) & (flag == f), (fold == k) & (flag == f)
            if not held.any():
                continue
            mean = rows[fit].mean(0)
            best = int(mean.argmax())
            if mean[best] - mean[half] < MARGIN:
                best = half
            nested[held] = rows[held, best]
    return float(rows[:, half].mean()), float(nested.mean())


both = {m: {run: oof_scores(run, m) for run in runs_m} for m, runs_m in candidates.items()}
table = {m: {run: v[1] for run, v in t.items()} for m, t in both.items()}
plain = {m: {run: v[0] for run, v in t.items()} for m, t in both.items()}
choice = {m: max(t, key=t.get) for m, t in table.items()}
for k in range(5):
    outer, test = {}, {}
    for m, candidate in choice.items():
        names, probs = load_probs(candidate, k, "outer")
        for n, p in zip(names, probs):
            if method_train[n] == m:
                outer[n] = p.astype(np.float16)
        test_names, probs = load_probs(candidate, k, "test")
        for n, p in zip(test_names, probs):
            if method_test[n] == m:
                test[n] = p.astype(np.float16)
    assert len(outer) == len(names) and len(test) == 1020
    d = runs / out_name / f"fold_{k}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "outer_probs.npz", names=np.array(names), probs=np.stack([outer[n] for n in names]))
    np.savez_compressed(d / "test_probs.npz", names=np.array(test_names), probs=np.stack([test[n] for n in test_names]))
(runs / out_name / "combination.json").write_text(json.dumps({"choice": choice, "oof_by_candidate_nested_cut": table, "oof_by_candidate_at_0.5": plain}, indent=1))
(runs / out_name / "components.txt").write_text(",".join(f"{m}={r}" for m, r in choice.items()))
print("COMBINED", out_name, json.dumps(choice), "| nested cut:", json.dumps({m: {r: round(v, 4) for r, v in t.items()} for m, t in table.items()}),
      "| at 0.5:", json.dumps({m: {r: round(v, 4) for r, v in t.items()} for m, t in plain.items()}), flush=True)
