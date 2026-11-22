"""Learned selection of Segment Anything segments, out-of-fold.

The edits of this dataset are object-centred. In transplant pictures the watermarked region is a set of object
segments; in full-frame pictures part of the holes are added or removed objects. The detector's probability map
is pooled over every SAM segment (mean, contrast with the surrounding ring, rank among the picture's segments
...) and a small network predicts whether the segment belongs to the minority region (watermark in a clean
picture, hole in a full-frame watermark). The region is the union of accepted segments.

Rigour: the probability maps are out-of-fold; for outer fold k the segment classifier is trained on the other
four folds only; the acceptance threshold and the way the union is combined with the plain thresholded map are
chosen per regime and method on those four folds with cross-fitted classifier outputs (never in-sample) and only
adopted when they beat "keep" by a clear margin; then they are applied to fold k. Pictures whose rectangle was
accepted by the decoder search, and box-like blobs, are untouched. SAM never sees a label.
usage: sam_select.py --folds <run> --train-root <release/train> --rect-train <json> --frame-json <json> --sam <npz> [<npz> ...]
       [--out <json>] [--rect-test <json> --test-out <npz>] [--regimes clean,full]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from torch import nn

from assemble import METHODS, TAUS, decide, miou

THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
VARIANTS = ("segments_or_blob", "segments_plus_uncovered_blob", "segments_plus_blob")
MARGIN = 0.01          # a rule is adopted only if it beats "keep" by this much on the other folds (cross-fitted)
# StegaStamp is excluded: its probability maps are already sharp, and on the first out-of-fold evaluation the
# selection moved it by -0.001 overall (+0.012 on transplants, -0.037 on unconfirmed rectangles).
SELECT_METHODS = ("pixelseal", "trustmark")
KEEP = ("keep", 0.0)


def load_sam(paths):
    out = {}
    for path in paths:
        if not Path(path).exists():
            continue
        with np.load(path) as z:
            names, counts, packed, quality = z["names"].tolist(), z["counts"], z["packed"], z["quality"]
        start = 0
        for n, c in zip(names, counts):
            out[n] = (np.unpackbits(packed[start:start + c], axis=1)[:, :128 * 128].reshape(c, 128, 128).astype(bool), quality[start:start + c])
            start += c
    return out


LINEAR = False          # set by --linear: logistic regression instead of the small network (tested, see CLAUDE.md)
GEOMETRY = False        # set by --geometry: where the segment sits and how object-like it is (a transplant is usually
                        # a prominent object away from the border); helps when the detector saw nothing at all


def geometry(segs):
    """Position and shape of each segment: centre, distance to the picture centre, border contact, box fill, nesting."""
    n = len(segs)
    flat = segs.reshape(n, -1)
    area = flat.sum(1)
    out = []
    for j, s in enumerate(segs):
        ys, xs = np.where(s)
        cy, cx = ys.mean() / 127, xs.mean() / 127
        border = float(ys.min() == 0) + float(xs.min() == 0) + float(ys.max() == 127) + float(xs.max() == 127)
        fill = area[j] / ((ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()))
        inside = (flat & flat[j]).sum(1)
        children = float(((inside >= 0.9 * area) & (area < 0.9 * area[j])).sum())      # smaller segments mostly inside this one
        parents = float(((inside >= 0.9 * area[j]) & (area > 1.1 * area[j])).sum())    # larger segments containing this one
        out.append([cx, cy, abs(cx - 0.5), abs(cy - 0.5), border, fill, np.log1p(children), np.log1p(parents)])
    return out


def features(prob, segs, quality, method, extra=()):
    """Per-segment evidence; `extra` = probability maps of other runs for the same picture (more opinions)."""
    geo = geometry(segs) if GEOMETRY else None
    blob = prob > 0.5
    labels, count = ndimage.label(blob)
    img = [prob.mean(), blob.mean(), prob.max()]
    means = np.array([prob[s].mean() for s in segs])
    rows = []
    for j, s in enumerate(segs):
        p = prob[s]
        ring = ndimage.binary_dilation(s, iterations=3) & ~s
        ring_mean = prob[ring].mean() if ring.any() else prob.mean()
        best_iou, covered = 0.0, 0.0
        if count:
            hit = np.unique(labels[s & blob])
            for c in hit[hit > 0]:
                comp = labels == c
                best_iou = max(best_iou, (comp & s).sum() / (comp | s).sum())
            covered = (blob & s).sum() / max(blob.sum(), 1)
        rows.append([means[j], np.quantile(p, 0.95), (p > 0.5).mean(), (p > 0.2).mean(), s.mean(), ring_mean, means[j] - ring_mean,
                     *img, (means[j] - means.mean()) / (means.std() + 1e-6), (means < means[j]).mean(), best_iou, covered,
                     *[float(method == m) for m in METHODS], quality[j][1], quality[j][2], np.log(s.sum())])
        for e in extra:
            rows[-1] += [e[s].mean(), e[s].mean() - (e[ring].mean() if ring.any() else e.mean()), (e[s] > 0.5).mean()]
        if geo is not None:
            rows[-1] += geo[j]
    return np.array(rows, np.float32)


def fit_predict(xa, ya, xb, seeds=3):
    """Small network on standardised features; returns the mean prediction of a few seeds for xb."""
    mu, sd = xa.mean(0), xa.std(0) + 1e-6
    xa_t, xb_t = torch.from_numpy((xa - mu) / sd), torch.from_numpy((xb - mu) / sd)
    ya_t = torch.from_numpy(ya.astype(np.float32))
    out = 0
    for seed in range(seeds):
        torch.manual_seed(seed)
        net = (nn.Linear(xa.shape[1], 1) if LINEAR else
               nn.Sequential(nn.Linear(xa.shape[1], 48), nn.GELU(), nn.Dropout(0.1), nn.Linear(48, 48), nn.GELU(), nn.Linear(48, 1)))
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-3)
        for _ in range(300):
            loss = nn.functional.binary_cross_entropy_with_logits(net(xa_t)[:, 0], ya_t)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            out = out + net(xb_t)[:, 0].sigmoid().numpy() / seeds
    return out


def rectangle_like(prob):
    """True when the thresholded map is one box-filling blob: a rectangle, not an object."""
    blob = prob > 0.5
    if blob.sum() < 16:
        return False
    ys, xs = np.where(blob)
    return blob.sum() / ((ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min())) >= 0.9


def choose(options, score):
    """Best option, but 'keep' unless the gain is clear: small apparent gains on a few dozen pictures are noise."""
    scores = {o: score(o) for o in options}
    best = max(scores, key=scores.get)
    return best if scores[best] - scores[KEEP] >= MARGIN else KEEP


def build(prob, segs, q, t, variant):
    blob = prob > 0.5
    chosen = segs[q >= t]
    if len(chosen) == 0:
        return blob
    union = chosen.any(0)
    if variant == "segments_or_blob":
        return union
    if variant == "segments_plus_blob":
        return union | blob
    return union | (blob & ~segs.any(0))


def options_for(method):
    return [KEEP] + [(v, t) for v in VARIANTS for t in THRESHOLDS if method in SELECT_METHODS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, type=Path)
    ap.add_argument("--train-root", required=True, type=Path)
    ap.add_argument("--rect-train", required=True, type=Path)
    ap.add_argument("--frame-json", required=True, type=Path)
    ap.add_argument("--sam", required=True, nargs="+")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rect-test", type=Path)
    ap.add_argument("--test-out", type=Path, help="write the masks of the selected test pictures here (needs --rect-test)")
    ap.add_argument("--regimes", default="clean", help="clean = pictures without full-frame watermark, full = holes of full-frame pictures")
    ap.add_argument("--extra", nargs="*", default=[], type=Path, help="other runs (same folds) whose probability maps are added as segment evidence")
    ap.add_argument("--linear", action="store_true", help="logistic regression instead of the small network")
    ap.add_argument("--geometry", action="store_true", help="add position / shape / nesting of each segment to its features")
    args = ap.parse_args()
    global GEOMETRY, LINEAR
    GEOMETRY, LINEAR = args.geometry, args.linear
    torch.set_num_threads(8)
    regimes = args.regimes.split(",")
    rows = json.loads((args.train_root / "metadata.json").read_text())
    meta = {r["image_name"]: r for r in rows}
    with np.load(args.train_root / "masks_128.npz") as z:
        masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
    frames = json.loads(args.frame_json.read_text())
    frame = frames["train"]
    oof, fold_of = {}, {}
    for k in range(5):
        with np.load(args.folds / f"fold_{k}" / "outer_probs.npz") as z:
            for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32)):
                oof[n], fold_of[n] = p, k
    names = sorted(oof)
    extra_oof = []                      # out-of-fold maps of the extra runs: fold k's model never saw fold k
    for run in args.extra:
        d = {}
        for k in range(5):
            with np.load(run / f"fold_{k}" / "outer_probs.npz") as z:
                d.update(dict(zip(z["names"].tolist(), z["probs"].astype(np.float32))))
        extra_oof.append(d)
    rect = json.loads(args.rect_train.read_text())
    sam = load_sam(args.sam)

    # the current pipeline (rectangle snapping with nested settings), exactly as assemble.py
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

    def is_full(acc):
        return acc >= 0.8

    def view(prob, full):
        """Probability of the minority region: the watermark in a clean picture, the hole in a full-frame one."""
        return 1 - prob if full else prob

    def final_mask(region, full):
        return ~region if full else region

    after, chosen, state = dict(before), {}, {}
    for regime in regimes:
        full = regime == "full"
        cand = [n for n in names if is_full(frame[n]["acc"]) == full and not rect_used[n] and n in sam and len(sam[n][0])
                and not rectangle_like(view(oof[n], full))]
        X = {n: features(view(oof[n], full), sam[n][0], sam[n][1], meta[n]["method"], [view(e[n], full) for e in extra_oof]) for n in cand}
        truth = {n: (~masks[n] if full else masks[n]) for n in cand}
        Y = {n: ((sam[n][0] & truth[n]).reshape(len(sam[n][0]), -1).sum(1) / sam[n][0].reshape(len(sam[n][0]), -1).sum(1) >= 0.6) for n in cand}
        print("SELECT", regime, "candidate pictures", len(cand), "segments", sum(len(v) for v in X.values()), "positive share",
              round(float(np.concatenate(list(Y.values())).mean()), 3), flush=True)

        def predict(train_names, test_names, X=X, Y=Y):
            q = fit_predict(np.concatenate([X[n] for n in train_names]), np.concatenate([Y[n] for n in train_names]),
                            np.concatenate([X[n] for n in test_names]))
            out, start = {}, 0
            for n in test_names:
                out[n] = q[start:start + len(X[n])]; start += len(X[n])
            return out

        def score(opt, images, q, full=full):
            if opt == KEEP:
                return float(np.mean([before[n] for n in images]))
            return float(np.mean([miou(final_mask(build(view(oof[n], full), sam[n][0], q[n], opt[1], opt[0]), full), masks[n]) for n in images]))

        for k in range(5):
            others = [n for n in cand if fold_of[n] != k]
            mine = [n for n in cand if fold_of[n] == k]
            if not mine:
                continue
            cross = {}                 # cross-fitted outputs on the other folds: settings are chosen without fold k
            for j in range(5):
                if j != k:
                    part = [n for n in others if fold_of[n] == j]
                    if part:
                        cross.update(predict([n for n in others if fold_of[n] != j], part))
            q_mine = predict(others, mine)
            for m in METHODS:
                pool = [n for n in others if meta[n]["method"] == m]
                best = choose(options_for(m), lambda o: score(o, pool, cross)) if pool else KEEP
                chosen[f"{regime}/{m}/fold{k}"] = list(best)
                for n in mine:
                    if meta[n]["method"] == m and best != KEEP:
                        after[n] = miou(final_mask(build(view(oof[n], full), sam[n][0], q_mine[n], best[1], best[0]), full), masks[n])
        state[regime] = (cand, X, Y, predict, score)

    def group(n):
        if is_full(frame[n]["acc"]):
            return "full"
        m = masks[n]; ys, xs = np.where(m)
        return "rect" if m.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "transplant"

    def mean(d, sel):
        v = [d[n] for n in names if sel(n)]
        return round(float(np.mean(v)), 4) if v else None

    print("SELECT out-of-fold mIoU over all 510 pictures: before", mean(before, lambda n: True), "after", mean(after, lambda n: True), flush=True)
    print("SELECT by fold (all pictures):", [(mean(before, lambda n, k=k: fold_of[n] == k), mean(after, lambda n, k=k: fold_of[n] == k)) for k in range(5)], flush=True)
    for m in METHODS:
        print("SELECT", m, "before/after", mean(before, lambda n: meta[n]["method"] == m), mean(after, lambda n: meta[n]["method"] == m),
              {g: (mean(before, lambda n: meta[n]["method"] == m and group(n) == g), mean(after, lambda n: meta[n]["method"] == m and group(n) == g))
               for g in ("full", "rect", "transplant")}, flush=True)
    print("SELECT settings per fold:", chosen, flush=True)
    diff = np.array([after[n] - before[n] for n in names])
    print("SELECT pictures improved", int((diff > 1e-9).sum()), "worse", int((diff < -1e-9).sum()), "unchanged", int((np.abs(diff) <= 1e-9).sum()), flush=True)
    if args.out:
        args.out.write_text(json.dumps({"before": before, "after": after, "settings": chosen}))

    if args.test_out:
        # Test rule: classifier trained on all train candidates (their probability maps are out-of-fold), threshold
        # and combination per regime and method chosen on cross-fitted outputs of those pictures. No test label exists.
        test_root = args.train_root.parent / "test"
        tmeta = {r["image_name"]: r for r in json.loads((test_root / "metadata.json").read_text())}
        tframe = frames["test"]
        probs, tnames = [], None
        for k in range(5):
            with np.load(args.folds / f"fold_{k}" / "test_probs.npz") as z:
                tnames = z["names"].tolist()
                probs.append(z["probs"].astype(np.float32))
        tprob = dict(zip(tnames, np.mean(probs, 0)))
        extra_test = []
        for run in args.extra:
            parts = []
            for k in range(5):
                with np.load(run / f"fold_{k}" / "test_probs.npz") as z:
                    assert z["names"].tolist() == tnames
                    parts.append(z["probs"].astype(np.float32))
            extra_test.append(dict(zip(tnames, np.mean(parts, 0))))
        rect_test = json.loads(args.rect_test.read_text())
        tau = {m: calibrate([n for n in names if meta[n]["method"] == m])[0] for m in METHODS}
        out_names, out_masks, rule = [], [], {}
        for regime in regimes:
            full = regime == "full"
            cand, X, Y, predict, score = state[regime]
            tcand = [n for n in tnames if is_full(tframe[n]["acc"]) == full and n in sam and len(sam[n][0]) and not rectangle_like(view(tprob[n], full))
                     and not (rect_test.get(n) is not None and rect_test[n]["margin"] >= tau[tmeta[n]["method"]])]
            cross = {}
            for j in range(5):
                part = [n for n in cand if fold_of[n] == j]
                if part:
                    cross.update(predict([n for n in cand if fold_of[n] != j], part))
            XT = {n: features(view(tprob[n], full), sam[n][0], sam[n][1], tmeta[n]["method"], [view(e[n], full) for e in extra_test]) for n in tcand}
            q = fit_predict(np.concatenate([X[n] for n in cand]), np.concatenate([Y[n] for n in cand]), np.concatenate([XT[n] for n in tcand]))
            qt, start = {}, 0
            for n in tcand:
                qt[n] = q[start:start + len(XT[n])]; start += len(XT[n])
            for m in METHODS:
                pool = [n for n in cand if meta[n]["method"] == m]
                best = choose(options_for(m), lambda o: score(o, pool, cross)) if pool else KEEP
                rule[f"{regime}/{m}"] = list(best)
                if best == KEEP:
                    continue
                for n in tcand:
                    if tmeta[n]["method"] == m:
                        out_names.append(n)
                        out_masks.append(final_mask(build(view(tprob[n], full), sam[n][0], qt[n], best[1], best[0]), full))
            print("SELECT test", regime, "candidates", len(tcand), "of", sum(1 for n in tnames if is_full(tframe[n]["acc"]) == full), flush=True)
        np.savez_compressed(args.test_out, names=np.array(out_names), masks=np.array(out_masks, bool).reshape(-1, 128, 128), rule=json.dumps(rule))
        changed = sum(int((mk != (tprob[n] > 0.5)).any()) for n, mk in zip(out_names, out_masks))
        print("SELECT test: masks written", len(out_names), "changed", changed, "| rule", rule, flush=True)


if __name__ == "__main__":
    main()
