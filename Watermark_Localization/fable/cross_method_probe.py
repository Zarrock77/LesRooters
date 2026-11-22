"""Is there a method-independent cue for the outpainted rectangle / edited regions?

Each fold model predicts its OUTER images while being told the WRONG watermark method. If the mask can still be
found (e.g. PixelSeal rectangles through the StegaStamp head), the cue is not the watermark itself but the edit,
and a shared head could serve the weak methods. Out-of-fold, nothing is trained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split, run  # noqa: E402
from train_det import METHODS, build_from_checkpoint, miou_batch  # noqa: E402

kit, folds, frame_json = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
index = {r["image_name"]: i for i, r in enumerate(rows)}
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
protocol = json.loads((Path("/srv/lesrooters/fable/reference") / "protocol.json").read_text())["folds"]
table = {}
for k in range(5):
    model = build_from_checkpoint(torch.load(folds / f"fold_{k}" / "best.pt", map_location="cuda"), kit).cuda().eval().to(memory_format=torch.channels_last)
    names = protocol[k]
    idx = torch.tensor([index[n] for n in names])
    true = torch.tensor([METHODS.index(rows[i]["method"]) for i in idx.tolist()])
    flags = torch.tensor([int(frame[n]["acc"] >= 0.8) for n in names])
    gt = np.stack([masks[n] for n in names])
    for told in range(3):
        probs = run(model, images[idx], torch.full_like(true, told), flags).astype(np.float32)
        s = miou_batch(probs > 0.5, gt)
        for j, n in enumerate(names):
            m = masks[n]
            ys, xs = np.where(m)
            rect = m.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min())
            group = "full" if flags[j] else ("rect" if rect else "transplant")
            table.setdefault((METHODS[int(true[j])], group, METHODS[told]), []).append(float(s[j]))
for true in METHODS:
    for group in ("full", "rect", "transplant"):
        print("CROSS true=" + true, group, "n=" + str(len(table[(true, group, true)])),
              {"told " + t: round(float(np.mean(table[(true, group, t)])), 3) for t in METHODS}, flush=True)
print("CROSS_DONE", flush=True)
