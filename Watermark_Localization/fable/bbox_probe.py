"""Does the known message decode in the bounding box of an object-shaped watermarked region?

Pictures without a full-frame watermark are either filled rectangles (watermarked in their own frame, the message
decodes in the rectangle) or object-shaped regions ("transplants"). If the object-shaped regions were also
watermarked in a patch frame (a rectangle around them, part of it restored to clean afterwards), the kit decoder
would read the message in a box around the region. Train labels are used for measuring only.
For each picture: bit accuracy of the kit decoder on the tight bounding box of the true region and on the box
enlarged by 10 / 25 / 50 % per side. Reference: the filled rectangles (should decode) and the full frame.
usage: bbox_probe.py <kit> <frame json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from rect_refine import METHODS, Scorer  # noqa: E402

kit, frame_json = Path(sys.argv[1]), Path(sys.argv[2])
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
PADS = (0.0, 0.1, 0.25, 0.5)

for method in METHODS:
    scorer = Scorer(kit, method)
    out = {}
    for i, r in enumerate(rows):
        n = r["image_name"]
        gt = masks[n]
        if r["method"] != method or frame[n]["acc"] >= 0.8 or not gt.any():
            continue
        ys, xs = np.where(gt)
        y0, y1, x0, x1 = ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4
        fill = gt.sum() * 16 / ((y1 - y0) * (x1 - x0))
        regime = "rectangle" if fill > 0.999 else "object-shaped"
        boxes = []
        for p in PADS:
            dy, dx = p * (y1 - y0), p * (x1 - x0)
            boxes.append([max(0, x0 - dx), max(0, y0 - dy), min(512, x1 + dx), min(512, y1 + dy)])
        image = images[i:i + 1].to(scorer.device).permute(0, 3, 1, 2).float() / 255
        sign = torch.tensor([int(c) for c in r["message"]], device=scorer.device).float() * 2 - 1
        _, acc = scorer(image, torch.tensor(boxes, device=scorer.device, dtype=torch.float32), sign)
        out.setdefault(regime, []).append([frame[n]["acc"], fill, min(y1 - y0, x1 - x0)] + acc.cpu().tolist())
    for regime, v in out.items():
        v = np.array(v)
        print("BBOX", method, f"{regime:14s}", "pictures", len(v), "| full-frame acc", round(float(v[:, 0].mean()), 3),
              "| region fills its box by", round(float(v[:, 1].mean()), 2), "| median short side", int(np.median(v[:, 2])), "px",
              "| mean bit acc at pad 0/10/25/50 %:", [round(float(v[:, 3 + k].mean()), 3) for k in range(len(PADS))],
              "| share of pictures with acc >= 0.8 (best pad):", round(float((v[:, 3:].max(1) >= 0.8).mean()), 3), flush=True)
        if regime == "object-shaped":
            big = v[:, 2] >= 96
            if big.any():
                print("BBOX", method, "object-shaped, short side >= 96 px:", int(big.sum()), "pictures | mean bit acc at pad 0/10/25/50 %:",
                      [round(float(v[big, 3 + k].mean()), 3) for k in range(len(PADS))], "| share >= 0.8:", round(float((v[big, 3:].max(1) >= 0.8).mean()), 3), flush=True)
    del scorer
    torch.cuda.empty_cache()
print("BBOX_DONE", flush=True)
