"""Do official pictures share a clean source? (train + test, 1530 pictures)

Every picture is reduced to a 16x16 grey thumbnail; pairs whose thumbnails nearly coincide are listed with the
share of pixels that are exactly identical at full resolution. If the same clean picture was used for several
dataset pictures, the pixels where two siblings differ are exactly where one of them carries a watermark.
usage: twin_probe.py <kit>
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

kit = Path(sys.argv[1])
names, thumbs, split_of = [], [], []
for split in ("train", "test"):
    for r in json.loads((kit / "release" / split / "metadata.json").read_text()):
        im = Image.open(kit / "release" / split / "images" / (r["image_name"] + ".png")).convert("L").resize((16, 16), Image.Resampling.BOX)
        names.append(r["image_name"]); split_of.append(split); thumbs.append(np.asarray(im, np.float32).ravel())
t = np.stack(thumbs)
t = (t - t.mean(1, keepdims=True)) / (t.std(1, keepdims=True) + 1e-6)
sim = t @ t.T / t.shape[1]
np.fill_diagonal(sim, -1)
best = sim.max(1)
print("TWIN pictures", len(names), "| nearest-neighbour thumbnail correlation: quantiles 50/90/99/max", np.round(np.quantile(best, [0.5, 0.9, 0.99, 1.0]), 3).tolist(), flush=True)
pairs = sorted({tuple(sorted((i, int(sim[i].argmax())))) for i in np.nonzero(best > 0.97)[0]})
print("TWIN pairs above 0.97:", len(pairs), flush=True)
for i, j in pairs[:12]:
    a = np.asarray(Image.open(kit / "release" / split_of[i] / "images" / (names[i] + ".png")).convert("RGB"), np.int16)
    b = np.asarray(Image.open(kit / "release" / split_of[j] / "images" / (names[j] + ".png")).convert("RGB"), np.int16)
    d = np.abs(a - b).max(2)
    print("TWIN", names[i], split_of[i], names[j], split_of[j], "corr", round(float(sim[i, j]), 4), "| identical pixels", round(float((d == 0).mean()), 4), "| differ by > 8:", round(float((d > 8).mean()), 4), flush=True)
print("TWIN_DONE", flush=True)
