"""Nested decision threshold per (method, regime flag), written as a shift of the probability maps.

assemble.py and sam_select.py cut every map at 0.5. Out-of-fold, the best cut differs by regime: for PixelSeal it is
about 0.25 when the message does not decode in the full frame (the watermark is the minority) and about 0.7 when
it does (the hole is the minority), the same in every fold. This script chooses, for each outer fold, the cut of
each (method, flag) group on the out-of-fold maps of the four OTHER folds, and keeps 0.5 unless the chosen cut beats
it there by MARGIN. The maps are then shifted in logit space so that "p > cut" becomes "p > 0.5", and written as a
new folds-like directory: everything downstream (rectangles, segment selection, replay record) stays unchanged and
its out-of-fold report is honest by construction. Test maps use the cut chosen on all five folds; the five test
files hold the shifted fold average, so the average cut at 0.5 is exactly "mean probability > cut".
With size edges (6th argument, e.g. 0.02,0.10) a second level is added: within each (method, flag) group the
pictures are split by the PREDICTED share of the minority region (mean probability of the map, label-free), and a
size group gets its own cut only if it beats the (method, flag) cut on the other folds by MARGIN and by 1.64
standard errors of the paired difference (small groups are noisy); otherwise it keeps the (method, flag) cut.
usage: calibrate_thr.py <runs dir> <base run> <out run> <frame json> <reference dir> [size edges]
"""
import json
import sys
from pathlib import Path

import numpy as np

runs, base, out_name, frame_json, reference = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5])
kit = Path("/srv/lesrooters/Hackathon_Watermark_Localisation_Task")
EDGES = tuple(float(v) for v in sys.argv[6].split(",")) if len(sys.argv) > 6 else ()
THRS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8) if not EDGES else     (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85)
MARGIN = 0.004            # fixed before looking at any outer-fold result of this rule
frame = json.loads(frame_json.read_text())
method = {s: {r["image_name"]: r["method"] for r in json.loads((kit / f"release/{s}/metadata.json").read_text())} for s in ("train", "test")}
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def group(split, n):
    return f"{method[split][n]}/{int(frame[split][n]['acc'] >= 0.8)}"


def size_bin(p, g):
    """Predicted share of the minority region: watermark when the frame does not decode, hole when it does."""
    share = float(p.mean())
    return int(np.digitize(1 - share if g.endswith("/1") else share, EDGES))


def miou(pred, g):
    s = 0.0
    for c in (False, True):
        p, t = pred == c, g == c
        u = (p | t).sum()
        s += 1.0 if u == 0 else (p & t).sum() / u
    return s / 2


def shift(p, t):
    if t == 0.5:
        return p.astype(np.float32)
    p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
    return (1 / (1 + np.exp(-(np.log(p / (1 - p)) - np.log(t / (1 - t)))))).astype(np.float32)


fold_names, fold_probs, test_names, test_probs = [], [], None, []
for k in range(5):
    with np.load(runs / base / f"fold_{k}" / "outer_probs.npz") as z:
        fold_names.append(z["names"].tolist()); fold_probs.append(z["probs"].astype(np.float32))
    with np.load(runs / base / f"fold_{k}" / "test_probs.npz") as z:
        test_names = z["names"].tolist() if test_names is None else test_names
        assert test_names == z["names"].tolist()
        test_probs.append(z["probs"].astype(np.float32))
score = [{t: np.array([miou(p > t, masks[n]) for n, p in zip(fold_names[k], fold_probs[k])]) for t in THRS} for k in range(5)]
groups = [np.array([group("train", n) for n in fold_names[k]]) for k in range(5)]
all_groups = sorted(set(np.concatenate(groups).tolist()))


sizes = [np.array([size_bin(p, g) for p, g in zip(fold_probs[k], groups[k])]) for k in range(5)]


def choose(folds_used, g):
    mean = {t: np.concatenate([score[k][t][groups[k] == g] for k in folds_used]).mean() for t in THRS}
    best = max(THRS, key=lambda t: mean[t])
    return (best, float(mean[best] - mean[0.5])) if mean[best] - mean[0.5] >= MARGIN else (0.5, float(mean[best] - mean[0.5]))


def choose_size(folds_used, g, b, parent):
    """Cut of size group b inside (method, flag) group g, or the parent cut when the evidence is weak."""
    col = {t: np.concatenate([score[k][t][(groups[k] == g) & (sizes[k] == b)] for k in folds_used]) for t in THRS}
    if len(col[0.5]) < 8:
        return parent
    best = max(THRS, key=lambda t: col[t].mean())
    d = col[best] - col[parent]
    return best if d.mean() >= MARGIN and d.mean() >= 1.64 * d.std(ddof=1) / len(d) ** 0.5 else parent


record = {"margin": MARGIN, "per_fold": {}, "test": {}}
out = runs / out_name
before, after = [], []
for k in range(5):
    others = [j for j in range(5) if j != k]
    cut = {g: choose(others, g) for g in all_groups}
    record["per_fold"][k] = {g: list(v) for g, v in cut.items()}
    fine = {(g, b): choose_size(others, g, b, cut[g][0]) for g in all_groups for b in range(len(EDGES) + 1)} if EDGES else {}
    record.setdefault("per_fold_sizes", {})[k] = {f"{g}/{b}": v for (g, b), v in fine.items()}
    used = [fine[(g, b)] if EDGES else cut[g][0] for g, b in zip(groups[k], sizes[k])]
    probs = np.stack([shift(p, t) for p, t in zip(fold_probs[k], used)])
    (out / f"fold_{k}").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"fold_{k}" / "outer_probs.npz", names=np.array(fold_names[k]), probs=probs)
    before.append(score[k][0.5]); after.append(np.array([score[k][t][i] for i, t in enumerate(used)]))
final = {g: choose(range(5), g) for g in all_groups}
record["test"] = {g: list(v) for g, v in final.items()}
final_fine = {(g, b): choose_size(range(5), g, b, final[g][0]) for g in all_groups for b in range(len(EDGES) + 1)} if EDGES else {}
record["test_sizes"] = {f"{g}/{b}": v for (g, b), v in final_fine.items()}
# Same rounding as merge_folds.py (float16): a group whose cut stays at 0.5 then keeps byte-identical test maps, and
# the rectangle search can take over its earlier results (rect_refine.py, exact reuse).
mean_test = np.mean(test_probs, 0).astype(np.float16).astype(np.float32)


def test_cut(n, p):
    g = group("test", n)
    return final_fine[(g, size_bin(p, g))] if EDGES else final[g][0]


shifted = np.stack([shift(p, test_cut(n, p)) for n, p in zip(test_names, mean_test)])
for k in range(5):
    np.savez_compressed(out / f"fold_{k}" / "test_probs.npz", names=np.array(test_names), probs=shifted)
b, a = np.concatenate(before), np.concatenate(after)
record["oof_plain_before"], record["oof_plain_after"] = float(b.mean()), float(a.mean())
record["by_fold"] = [[float(x.mean()), float(y.mean())] for x, y in zip(before, after)]
(out / "calibration.json").write_text(json.dumps(record, indent=1))
print("CALIBRATED", out_name, "plain out-of-fold", round(float(b.mean()), 4), "->", round(float(a.mean()), 4),
      "| by fold", [(round(float(x.mean()), 4), round(float(y.mean()), 4)) for x, y in zip(before, after)], flush=True)
print("CALIBRATED cuts per fold", {g: [record["per_fold"][k][g][0] for k in range(5)] for g in all_groups}, "| test", {g: v[0] for g, v in final.items()}, flush=True)
if EDGES:
    print("CALIBRATED size cuts per fold", {key: [record["per_fold_sizes"][k][key] for k in range(5)] for key in record["test_sizes"]}, "| test", record["test_sizes"], flush=True)
