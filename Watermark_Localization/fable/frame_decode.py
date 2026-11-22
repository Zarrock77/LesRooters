"""Decode the known message in the FULL frame of every train and test image with the kit decoders.

Output json: {split: {image_name: {"acc": bit accuracy, "margin": mean signed logit}}}. The accuracy tells whether
the image carries a full-frame watermark (edited holes) or not (small rectangle / transplanted object).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

kit, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(kit))
import onnxruntime as ort  # noqa: E402
ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("pixelseal", "trustmark", "stegastamp")
result = {}
for split in ("train", "test"):
    rows = json.loads((kit / "release" / split / "metadata.json").read_text())
    result[split] = {}
    for method in METHODS:
        wm = load_watermark(method, models_dir=kit / "models", device="cuda:0")
        sub = [r for r in rows if r["method"] == method]
        for s in range(0, len(sub), 32):
            batch = sub[s:s + 32]
            x = torch.from_numpy(np.stack([np.array(Image.open(kit / "release" / split / "images" / (r["image_name"] + ".png")).convert("RGB"))
                                           for r in batch])).permute(0, 3, 1, 2).float() / 255
            lg = wm.logits(x).cpu().numpy()
            for r, l in zip(batch, lg):
                bits = np.array([int(b) for b in r["message"]])
                result[split][r["image_name"]] = {"acc": float(((l > 0) == (bits > 0)).mean()),
                                                  "margin": float(((bits * 2 - 1) * l).mean())}
        del wm
        torch.cuda.empty_cache()
        print("FRAME", split, method, len(sub), flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, sort_keys=True))
for split in ("train", "test"):
    acc = np.array([v["acc"] for v in result[split].values()])
    print(split, len(acc), "acc>=0.8:", int((acc >= 0.8).sum()), "0.65<=acc<0.8:", int(((acc >= 0.65) & (acc < 0.8)).sum()), flush=True)
print("FRAME_DONE", flush=True)
