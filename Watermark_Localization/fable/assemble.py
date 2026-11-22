"""Score out-of-fold predictions honestly and assemble the test submission.

OOF = every official train image predicted by the fold model that never saw it (fit or tune).
Optional rectangle snapping (rect_refine.py output) has two small global settings per method (acceptance margin,
inset in pixels); they are chosen for each outer fold on the OTHER four folds only, so the reported OOF stays
out-of-sample. The test rule uses settings chosen on all OOF images.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

METHODS = ("pixelseal", "trustmark", "stegastamp")
TAUS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 1e9)


def miou(pred, gt):
    s = 0.0
    for c in (0, 1):
        p, g = pred == c, gt == c
        u = (p | g).sum()
        s += 1.0 if u == 0 else (p & g).sum() / u
    return s / 2


def rasterize(box, offset):
    """Rectangle in 512-px edge coordinates (+ per-side correction) -> 128 grid with the official >=50 % rule."""
    x0, y0, x1, y1 = box[0] + offset[0], box[1] + offset[1], box[2] + offset[2], box[3] + offset[3]
    edges = np.arange(129) * 4.0
    cx = np.clip(np.minimum(edges[1:], x1) - np.maximum(edges[:-1], x0), 0, 4)
    cy = np.clip(np.minimum(edges[1:], y1) - np.maximum(edges[:-1], y0), 0, 4)
    return (cy[:, None] * cx[None, :]) / 16.0 >= 0.5


def decide(prob, rect, tau, offset, threshold=0.5):
    if rect is not None and rect["margin"] >= tau:
        m = rasterize(rect["box"], offset)
        if m.any():
            return m
    return prob > threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, type=Path, help="directory with fold_0..fold_4")
    ap.add_argument("--train-root", required=True, type=Path, help="release/train")
    ap.add_argument("--reference-scores", type=Path, help="image_scores.json of the incumbent (paired comparison)")
    ap.add_argument("--rect-train", type=Path)
    ap.add_argument("--rect-test", type=Path)
    ap.add_argument("--submission", type=Path, help="write submission .pt + manifest here (needs torch)")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--replay", type=Path, help="replay.json of the fold models that produced the probabilities")
    ap.add_argument("--candidate", default="fable_detector")
    ap.add_argument("--sam-scores", type=Path, help="sam_select.py --out: out-of-fold scores after nested segment selection")
    ap.add_argument("--sam-test", type=Path, help="sam_select.py --test-out: masks of the selected test pictures")
    args = ap.parse_args()

    rows = json.loads((args.train_root / "metadata.json").read_text())
    with np.load(args.train_root / "masks_128.npz") as z:
        masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
    meta = {r["image_name"]: r for r in rows}
    oof, fold_of, test_probs, test_names = {}, {}, [], None
    for k in range(5):
        with np.load(args.folds / f"fold_{k}" / "outer_probs.npz") as z:
            for n, p in zip(z["names"].tolist(), z["probs"].astype(np.float32)):
                assert n not in oof
                oof[n], fold_of[n] = p, k
        tp = args.folds / f"fold_{k}" / "test_probs.npz"
        if tp.exists():
            with np.load(tp) as z:
                test_names = z["names"].tolist() if test_names is None else test_names
                assert test_names == z["names"].tolist()
                test_probs.append(z["probs"].astype(np.float32))
    assert set(oof) == set(masks) and len(oof) == 510
    names = sorted(oof)
    rect = json.loads(args.rect_train.read_text()) if args.rect_train else {}

    base = {n: miou(oof[n] > 0.5, masks[n]) for n in names}
    # True bounding boxes (512 px) of the train masks, used only to calibrate the snapped boxes.
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
        """Median per-side correction (true - snapped) over confident, roughly right snaps; then the best margin."""
        good = [n for n in images if rect.get(n) and rect[n]["margin"] >= 3.0 and box_iou(rect[n]["box"], gt_box[n]) > 0.6]
        offset = tuple(np.median([[gt_box[n][k] - rect[n]["box"][k] for k in range(4)] for n in good], 0).tolist()) if len(good) >= 5 else (0.0,) * 4
        tau = max(TAUS, key=lambda t: np.mean([miou(decide(oof[n], rect.get(n), t, offset), masks[n]) for n in images]))
        return tau, offset

    final, chosen = dict(base), {}
    if rect:
        for m in METHODS:
            for k in range(5):
                others = [n for n in names if meta[n]["method"] == m and fold_of[n] != k]
                tau, offset = calibrate(others)       # settings never see the fold they are applied to
                chosen[f"{m}/fold{k}"] = [tau, list(offset)]
                for n in names:
                    if meta[n]["method"] == m and fold_of[n] == k:
                        final[n] = miou(decide(oof[n], rect.get(n), tau, offset), masks[n])
    sam_settings = None
    if args.sam_scores:
        sel = json.loads(args.sam_scores.read_text())
        assert max(abs(sel["before"][n] - final[n]) for n in names) < 1e-9, "segment selection started from another pipeline"
        final = {n: float(sel["after"][n]) for n in names}
        sam_settings = sel["settings"]
    report = {"oof_miou_threshold_only": float(np.mean(list(base.values()))), "oof_miou": float(np.mean(list(final.values()))),
              "folds": [float(np.mean([final[n] for n in names if fold_of[n] == k])) for k in range(5)],
              "methods": {m: float(np.mean([final[n] for n in names if meta[n]["method"] == m])) for m in METHODS},
              "full_watermark": {str(f): float(np.mean([final[n] for n in names if bool(meta[n]["full_watermark"]) == f])) for f in (True, False)},
              "snapping_settings_per_fold": chosen}
    if sam_settings is not None:
        report["segment_selection_settings_per_fold"] = sam_settings
    if args.reference_scores:
        ref = json.loads(args.reference_scores.read_text())
        diff = np.array([final[n] - ref[n] for n in names])
        rng = np.random.default_rng(20261126)
        boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(20000)])
        report["reference_oof"] = float(np.mean([ref[n] for n in names]))
        report["paired_gain"] = float(diff.mean())
        report["paired_gain_99pct"] = [float(np.quantile(boot, .005)), float(np.quantile(boot, .995))]
        report["fold_gain"] = [float(np.mean([final[n] - ref[n] for n in names if fold_of[n] == k])) for k in range(5)]
        report["improved_images"] = int((diff > 0).sum()); report["worse_images"] = int((diff < 0).sum())
    print(json.dumps(report, indent=1))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({**report, "image_scores": final}, indent=1))

    if args.submission:
        import torch
        assert len(test_probs) == 5 and len(test_names) == 1020 and len(set(test_names)) == 1020
        prob = np.mean(test_probs, 0)
        rect_test = json.loads(args.rect_test.read_text()) if args.rect_test else {}
        test_meta = {r["image_name"]: r for r in json.loads((args.train_root.parent / "test" / "metadata.json").read_text())}
        assert [r for r in test_meta] == test_names
        settings = {}
        if rect and rect_test:
            for m in METHODS:
                settings[m] = calibrate([n for n in names if meta[n]["method"] == m])
        out = np.zeros((1020, 128, 128), bool)
        snapped = 0
        for j, n in enumerate(test_names):
            m = test_meta[n]["method"]
            if m in settings:
                out[j] = decide(prob[j], rect_test.get(n), *settings[m])
                snapped += int(rect_test.get(n) is not None and rect_test[n]["margin"] >= settings[m][0])
            else:
                out[j] = prob[j] > 0.5
        selected = 0
        if args.sam_test:
            position = {n: j for j, n in enumerate(test_names)}
            with np.load(args.sam_test) as z:
                for n, mk in zip(z["names"].tolist(), z["masks"]):
                    out[position[n]] = mk
                    selected += 1
        args.submission.mkdir(parents=True, exist_ok=True)
        artifact = args.submission / "lesrooters_watermark_fable.pt"
        torch.save({"image_names": test_names, "masks": torch.from_numpy(out)}, artifact)
        check = torch.load(artifact, weights_only=True)
        assert check["image_names"] == test_names and check["masks"].shape == (1020, 128, 128) and check["masks"].dtype == torch.bool
        manifest = {"task_id": "34-watermark-localisation-image", "images": 1020, "shape": [1020, 128, 128], "binary": True,
                    "unique_names_match_test_metadata": True, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "bytes": artifact.stat().st_size, "oof_miou": report["oof_miou"], "test_snapped_rectangles": snapped,
                    "test_snapping_settings": {k: [v[0], list(v[1])] for k, v in settings.items()},
                    "test_segment_selected_pictures": selected,
                    "test_positive_fraction": float(out.mean()), "uniform_test_masks": int(sum(o.all() or not o.any() for o in out))}
        if args.replay:
            replay = json.loads(args.replay.read_text())
            exact = sum(r["exact_images"] for r in replay["folds"]); checked = sum(r["checked_images"] for r in replay["folds"])
            manifest.update({"artifact": artifact.name, "dtype": "torch.bool", "candidate": args.candidate,
                             "validation_expected_predictions": 510, "validation_predictions_checked": checked,
                             "validation_predictions_reproduced_exactly": exact,
                             "validation_replay_mode": "exact" if exact == checked == 510 else "bounded_gpu_numerics",
                             "validation_replay_limits": replay["limits"],
                             "validation_reference": "five outer folds; each fold model replayed on images it never saw",
                             "components": [{"fold": r["fold"], "validation_replay": {k: v for k, v in r.items() if k != "fold"}} for r in replay["folds"]],
                             "candidate_oof_metrics": {k: v for k, v in report.items()
                                                       if k not in ("snapping_settings_per_fold", "segment_selection_settings_per_fold")}})
        (args.submission / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print("SUBMISSION", json.dumps(manifest))


if __name__ == "__main__":
    main()
