"""Regenerate a run's stored predictions from its saved fold models through ONE deterministic inference path.

For each fold: outer (out-of-fold) probabilities for the 102 held-out images and probabilities for the 1020 test
images, written over outer_probs.npz / test_probs.npz (the first version is kept as *_asfit.npz). The replay check
uses exactly the same function, batch composition and settings, so stored and replayed predictions can be
compared bit for bit instead of within a numerical tolerance.

--tta-out <dir> writes a sibling run instead, predicted with test-time shifts: the picture is moved by one scoring
cell (4 pixels) in a 3 x 3 grid, the maps are moved back and averaged. Out-of-fold
gain measured with tta_probe.py before use. The sibling run links to the same fold models and carries a TTA marker
that the replay check honours.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split, run  # noqa: E402
from train_det import METHODS, build_from_checkpoint  # noqa: E402

BATCH = 32
# 3 x 3 grid of one-cell shifts. Out-of-fold, against no shift / the 5-shift cross: PixelSeal 0.6495 (0.6393 / 0.6478),
# TrustMark 0.7279 (0.7213 / 0.7242), StegaStamp 0.9084 (0.9048 / 0.9083); larger grids bring nothing more (tta_probe.py).
SHIFTS = tuple((dy, dx) for dy in (-4, 0, 4) for dx in (-4, 0, 4))
CHUNK = 128


def deterministic():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fold_model(run_dir, k, kit):
    state = torch.load(run_dir / f"fold_{k}" / "best.pt", map_location="cuda")
    return build_from_checkpoint(state, kit).cuda().eval().to(memory_format=torch.channels_last)


def shifted(images, dy, dx):
    """Shift [N,H,W,3] uint8 pictures by (dy, dx) pixels, filling with the mirrored border."""
    if dy == dx == 0:
        return images
    x = images.permute(0, 3, 1, 2).float()
    x = F.pad(x, (max(dx, 0), max(-dx, 0), max(dy, 0), max(-dy, 0)), mode="reflect")
    h, w = images.shape[1:3]
    y0, x0 = max(-dy, 0), max(-dx, 0)
    return x[:, :, y0:y0 + h, x0:x0 + w].round().byte().permute(0, 2, 3, 1).contiguous()


def unshift(prob, dy, dx):
    """Move a [N,128,128] map back by the shift (in cells), repeating the border."""
    cy, cx = dy // 4, dx // 4
    if cy == cx == 0:
        return prob
    p = torch.from_numpy(prob.astype(np.float32))[:, None]
    p = F.pad(p, (max(-cx, 0), max(cx, 0), max(-cy, 0), max(cy, 0)), mode="replicate")
    y0, x0 = max(cy, 0), max(cx, 0)
    return p[:, 0, y0:y0 + 128, x0:x0 + 128].numpy()


def predict_names(model, rows, images, index, names, frame, tta=False):
    idx = torch.tensor([index[n] for n in names])
    methods = torch.tensor([METHODS.index(rows[i]["method"]) for i in idx.tolist()])
    flags = torch.tensor([int(frame[n]["acc"] >= 0.8) for n in names])
    if not tta:
        return run(model, images[idx], methods, flags, batch=BATCH)
    out = []
    for s in range(0, len(idx), CHUNK):           # chunks keep the shifted copies small; batches inside stay at BATCH
        part, m, f = images[idx[s:s + CHUNK]], methods[s:s + CHUNK], flags[s:s + CHUNK]
        maps = [unshift(run(model, shifted(part, dy, dx), m, f, batch=BATCH).astype(np.float32), dy, dx) for dy, dx in SHIFTS]
        out.append(np.mean(maps, 0))
    return np.concatenate(out).astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--frame-json", required=True, type=Path)
    ap.add_argument("--tta-out", type=Path, help="write a shift-averaged sibling run here instead of re-storing the run itself")
    args = ap.parse_args()
    deterministic()
    tta = args.tta_out is not None
    target = args.tta_out if tta else args.run
    frame = json.loads(args.frame_json.read_text())
    train_rows, train_images = load_split(args.kit, "train")
    test_rows, test_images = load_split(args.kit, "test")
    train_index = {r["image_name"]: i for i, r in enumerate(train_rows)}
    test_index = {r["image_name"]: i for i, r in enumerate(test_rows)}
    test_names = [r["image_name"] for r in test_rows]
    for k in range(5):
        d, t = args.run / f"fold_{k}", target / f"fold_{k}"
        t.mkdir(parents=True, exist_ok=True)
        with np.load(d / "outer_probs.npz") as z:
            names, before = z["names"].tolist(), z["probs"].astype(np.float32)
        model = fold_model(args.run, k, args.kit)
        outer = predict_names(model, train_rows, train_images, train_index, names, frame["train"], tta)
        test = predict_names(model, test_rows, test_images, test_index, test_names, frame["test"], tta)
        if tta:
            link = t / "best.pt"
            if not link.exists():
                link.symlink_to(d / "best.pt")
        else:
            for name in ("outer_probs", "test_probs"):
                keep = d / f"{name}_asfit.npz"
                if not keep.exists():
                    (d / f"{name}.npz").rename(keep)
        np.savez_compressed(t / "outer_probs.npz", names=np.array(names), probs=outer)
        np.savez_compressed(t / "test_probs.npz", names=np.array(test_names), probs=test)
        flips = float(((outer.astype(np.float32) > 0.5) != (before > 0.5)).mean())
        print("REPREDICT", target.name, k, "cells changed vs stored version:", round(flips, 6), flush=True)
        del model
    if tta:
        (target / "TTA").write_text("shifts " + json.dumps(SHIFTS))
    (target / "REPREDICTED").write_text("deterministic inference path, batch %d%s" % (BATCH, ", shift-averaged" if tta else ""))
    print("REPREDICT_DONE", target.name, flush=True)


if __name__ == "__main__":
    main()
