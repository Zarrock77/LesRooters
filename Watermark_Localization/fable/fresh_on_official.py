"""Decoder margin of a FRESH kit embedding on official-domain pictures, versus the official watermark.

alpha_probe.py compared official margins with fresh embeddings on COCO photographs. The official pictures are a
different kind of content, so the fair reference is a fresh embedding on official pictures themselves: here on
official images of the OTHER methods whose own watermark covers little of the frame. Diagnostic only.
usage: fresh_on_official.py <kit> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(12)
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def stats(images_uint8, bits):
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255
    lg = torch.cat([torch.as_tensor(wm.logits(x[s:s + 8])).float().cpu() for s in range(0, len(x), 8)])
    sign = torch.from_numpy(bits).float() * 2 - 1
    return round(float(((lg > 0) == (sign > 0)).float().mean()), 4), round(float((lg * sign).mean()), 2)


own = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
own = sorted(own, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:32]
print("FRESH", method, "official watermark, smallest holes:", stats(images[own].numpy(), np.array([[int(c) for c in rows[i]["message"]] for i in own])), flush=True)
others = [i for i, r in enumerate(rows) if r["method"] != method and frame[r["image_name"]]["acc"] < 0.8 and masks[r["image_name"]].mean() < 0.05][:32]
S = images[others].numpy()
bits = rng.integers(0, 2, (len(S), wm.nbits))
W = np.stack([wm.embed(S[j], bits[j].tolist()) for j in range(len(S))])
print("FRESH", method, "fresh kit embedding on", len(S), "official pictures of other methods (bit acc, margin):", stats(W, bits), flush=True)
print("FRESH_DONE", flush=True)
