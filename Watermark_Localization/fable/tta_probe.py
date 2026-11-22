"""Does averaging the detector over small shifts of the picture help, out-of-fold?

Each fold model predicts its outer images as usual and with test-time shifts of one scoring cell (4 pixels) in
the four directions; the shifted maps are moved back and averaged. Nothing is fitted; the outer labels are only
used to score both versions.
usage: tta_probe.py <kit> <run dir> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split, run  # noqa: E402
from repredict import deterministic, fold_model  # noqa: E402
from train_det import METHODS, miou_batch  # noqa: E402

kit, run_dir, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
SHIFTS = ((0, 0), (4, 0), (-4, 0), (0, 4), (0, -4))


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


GRID = [(dy, dx) for dy in (-8, -4, 0, 4, 8) for dx in (-8, -4, 0, 4, 8)]
SETS = {"plain": [(0, 0)], "cross5": list(SHIFTS), "square9": [(dy, dx) for dy in (-4, 0, 4) for dx in (-4, 0, 4)],
        "cross9": list(SHIFTS) + [(8, 0), (-8, 0), (0, 8), (0, -8)], "square25": GRID}
deterministic()
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
index = {r["image_name"]: i for i, r in enumerate(rows)}
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
scores = {name: [] for name in SETS}
for k in range(5):
    with np.load(run_dir / f"fold_{k}" / "outer_probs.npz") as z:
        names = [n for n in z["names"].tolist() if rows[index[n]]["method"] == method]
    model = fold_model(run_dir, k, kit)
    idx = torch.tensor([index[n] for n in names])
    methods = torch.tensor([METHODS.index(method)] * len(names))
    flags = torch.tensor([int(frame[n]["acc"] >= 0.8) for n in names])
    gt = np.stack([masks[n] for n in names])
    maps = {(dy, dx): unshift(run(model, shifted(images[idx], dy, dx), methods, flags), dy, dx).astype(np.float32) for dy, dx in GRID}
    for name, shifts in SETS.items():
        scores[name].append(miou_batch(np.mean([maps[sh] for sh in shifts], 0) > 0.5, gt))
    print("TTA", run_dir.name, method, "fold", k, {name: round(float(v[-1].mean()), 4) for name, v in scores.items()}, flush=True)
    del model
print("TTA_DONE", run_dir.name, method, {name: round(float(np.concatenate(v).mean()), 4) for name, v in scores.items()}, flush=True)
