"""Segment Anything proposals for official images (generator environment, no labels read).

The edits of this dataset are object-centred: a transplanted object, an added object, a removed object. The
detector gives blurry evidence; SAM gives the exact outline of every object and part. This script stores, for
each image, all automatic SAM masks reduced to the 128x128 scoring grid with the kit's rule (cell = 1 if at
least half of its 4x4 pixels are in the mask). Selection of segments happens later, out-of-fold.
usage: sam_masks.py <kit> <split> <checkpoint> <out.npz> [shard index] [shard count] [points per side] [frame json]
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

kit, split, ckpt, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
shard, shards = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
rows = json.loads((kit / "release" / split / "metadata.json").read_text())
if len(sys.argv) > 8:      # only pictures where the message does not decode in the full frame (objects, rectangles)
    frame = json.loads(Path(sys.argv[8]).read_text())[split]
    full = len(sys.argv) > 9 and sys.argv[9] == "full"       # "full": the pictures with a full-frame watermark instead
    rows = [r for r in rows if (frame[r["image_name"]]["acc"] >= 0.8) == full]
if os.environ.get("SAM_METHODS"):                       # optional: only some watermarking methods
    rows = [r for r in rows if r["method"] in os.environ["SAM_METHODS"].split(",")]
rows = rows[shard::shards]
sam = sam_model_registry["vit_h"](checkpoint=ckpt).cuda().eval()
PPS = int(sys.argv[7]) if len(sys.argv) > 7 else 32
generator = SamAutomaticMaskGenerator(sam, points_per_side=PPS, points_per_batch=128, pred_iou_thresh=0.86,
                                      stability_score_thresh=0.90, crop_n_layers=int(os.environ.get("SAM_CROP_LAYERS", "0")),
                                      crop_n_points_downscale_factor=2, min_mask_region_area=0)
names, counts, packed, quality = [], [], [], []
if out.exists():                                        # resume a shard that was interrupted
    with np.load(out) as z:
        names, counts = z["names"].tolist(), z["counts"].tolist()
        packed, quality = list(z["packed"]), [tuple(q) for q in z["quality"]]
    rows = [r for r in rows if r["image_name"] not in set(names)]
    print("SAM resume", split, shard, "done", len(names), "left", len(rows), flush=True)
t0 = time.time()
for k, r in enumerate(rows):
    image = np.array(Image.open(kit / "release" / split / "images" / (r["image_name"] + ".png")).convert("RGB"))
    with torch.no_grad():
        masks = generator.generate(image)
    kept = 0
    for m in sorted(masks, key=lambda m: -m["area"]):
        small = m["segmentation"].reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
        if small.sum() < 2:
            continue
        packed.append(np.packbits(small))
        quality.append((m["area"], m["predicted_iou"], m["stability_score"]))
        kept += 1
    names.append(r["image_name"]); counts.append(kept)
    if k % 10 == 0:
        print("SAM", split, shard, k, "/", len(rows), "masks", kept, round(time.time() - t0, 1), "s", flush=True)
    if (k + 1) % 40 == 0 or k + 1 == len(rows):        # partial results are usable while the rest is computed
        tmp = out.with_name(out.stem + ".tmp.npz")
        np.savez_compressed(tmp, names=np.array(names), counts=np.array(counts), packed=np.stack(packed), quality=np.array(quality, np.float32))
        tmp.replace(out)
print("SAM_DONE", split, shard, len(names), "images", int(np.sum(counts)), "masks", round(time.time() - t0, 1), "s", flush=True)
