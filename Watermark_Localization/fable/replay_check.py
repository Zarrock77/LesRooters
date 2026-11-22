"""Replay every fold model on its outer images and compare with the stored out-of-fold probabilities.

Same limits as the team's reproduction.py (bounded GPU numerics). Output: replay.json with one report per fold.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from repredict import deterministic, fold_model, predict_names  # noqa: E402
from train_det import METHODS  # noqa: E402

LIMITS = {"pixel_disagreement_fraction": 0.001, "max_image_pixel_disagreement_fraction": 0.005,
          "probability_mean_absolute_difference": 0.002, "absolute_miou_difference": 0.001, "max_image_miou_difference": 0.01}


def scores(pred, gt):
    out = np.zeros(len(pred))
    for c in (0, 1):
        p, g = pred == c, gt == c
        u = (p | g).sum((1, 2)); i = (p & g).sum((1, 2))
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True, type=Path)
ap.add_argument("--folds", required=True, type=Path)
ap.add_argument("--frame-json", required=True, type=Path)
ap.add_argument("--components", default="", help="method=run,...: each method's images are replayed by that run's fold models; runA*runB = mean")
args = ap.parse_args()
comp = dict(a.split("=") for a in args.components.split(",")) if args.components else None
deterministic()
frame = json.loads(args.frame_json.read_text())["train"]
rows, images = load_split(args.kit, "train")
index = {r["image_name"]: i for i, r in enumerate(rows)}
with np.load(args.kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
reports = []
for k in range(5):
    with np.load(args.folds / f"fold_{k}" / "outer_probs.npz") as z:
        names, stored = z["names"].tolist(), z["probs"].astype(np.float32)
    methods = [rows[index[n]]["method"] for n in names]
    run_of = {m: ([args.folds.name] if comp is None else comp[m].split("*")) for m in METHODS}
    replay = np.zeros_like(stored)
    for run_name in sorted({r for v in run_of.values() for r in v}):
        # the whole outer fold goes through the same function and batches as when the predictions were stored
        run_dir = args.folds.parent / run_name
        tta = (run_dir / "TTA").exists()          # shift-averaged sibling run (repredict.py --tta-out)
        # A fold model replayed once is not run again for the next combination: its replayed predictions are kept
        # next to it, tied to the checkpoint file (path, size, modification time) and to the list of pictures.
        weights = os.path.realpath(run_dir / f"fold_{k}" / "best.pt")
        st = os.stat(weights)
        signature = f"{weights}:{st.st_size}:{st.st_mtime_ns}:{int(tta)}"
        cache, part = run_dir / f"fold_{k}" / "replay_part.npz", None
        if cache.exists():
            with np.load(cache) as z:
                if z["names"].tolist() == names and str(z["signature"]) == signature:
                    part = z["probs"].astype(np.float32)
        if part is None:
            model = fold_model(run_dir, k, args.kit)
            part = predict_names(model, rows, images, index, names, frame, tta).astype(np.float32)
            np.savez(cache, names=np.array(names), probs=part, signature=np.array(signature))
            del model
        for j, m in enumerate(methods):
            if run_name in run_of[m]:
                replay[j] += part[j] / len(run_of[m])
    replay = replay.astype(np.float16).astype(np.float32)      # stored predictions are float16
    gt = np.stack([masks[n] for n in names])
    a, b = replay > 0.5, stored > 0.5
    diff = (a != b).mean((1, 2))
    old, new = scores(b, gt), scores(a, gt)
    report = {"fold": k, "checked_images": len(names), "exact_images": int((diff == 0).sum()),
              "pixel_disagreement_fraction": float(diff.mean()), "max_image_pixel_disagreement_fraction": float(diff.max()),
              "probability_mean_absolute_difference": float(np.abs(replay - stored).mean()),
              "probability_max_absolute_difference": float(np.abs(replay - stored).max()),
              "reference_miou": float(old.mean()), "recomputed_miou": float(new.mean()),
              "absolute_miou_difference": float(abs(new.mean() - old.mean())), "max_image_miou_difference": float(np.abs(new - old).max())}
    print("REPLAY", json.dumps(report), flush=True)
    for field, limit in LIMITS.items():
        assert 0 <= report[field] <= limit, (field, report[field], limit)
    reports.append(report)
(args.folds / "replay.json").write_text(json.dumps({"limits": LIMITS, "folds": reports}, indent=1))
print("REPLAY_DONE checked", sum(r["checked_images"] for r in reports), "exact", sum(r["exact_images"] for r in reports), flush=True)
