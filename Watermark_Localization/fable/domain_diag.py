"""Diagnostic: does the synthetic-trained detector see a kit-embedded watermark on competition-domain content?

Content = official train images WITHOUT a full-frame watermark and of ANOTHER method (so they are clean for the
method under test). The method's watermark is embedded by the kit in the full frame, blob holes restore the
original pixels, and the detector must find the holes. The same procedure on COCO sources is the reference.
If COCO works and competition content does not, the gap is the image domain. Nothing is trained or tuned.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_det import METHODS, build_from_checkpoint, down128, make_mask, miou_batch, sample_area  # noqa: E402

kit, ckpt, coco = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 60
sys.path.insert(0, str(kit))
import onnxruntime as ort  # noqa: E402
ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402

state = torch.load(ckpt, map_location="cuda")
model = build_from_checkpoint(state, kit).cuda().eval().to(memory_format=torch.channels_last)
rows = json.loads((kit / "release/train/metadata.json").read_text())
coco_clean = np.load(coco, mmap_mode="r")
rng = np.random.default_rng(20261121)

for mi, method in enumerate(METHODS):
    wm = load_watermark(method, models_dir=kit / "models", device="cuda:0")
    sources = {"competition": [np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB"))
                               for r in rows if not r["full_watermark"] and r["method"] != method][:limit],
               "coco": [np.array(coco_clean[len(coco_clean) - 1 - k]) for k in range(limit)]}
    for domain, images in sources.items():
        outs, labels = [], []
        for S in images:
            W = np.asarray(wm.embed(S, rng.integers(0, 2, wm.nbits).tolist()), dtype=np.uint8)
            hole = make_mask(rng, sample_area(rng, 0.01, 0.4, 0.8))
            out = W.copy(); out[hole] = S[hole]
            outs.append(out); labels.append(down128(~hole))
        x = torch.from_numpy(np.stack(outs))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            prob = torch.cat([model(x[s:s + 32].cuda(), torch.full((len(x[s:s + 32]),), mi, device="cuda"),
                                    torch.ones(len(x[s:s + 32]), dtype=torch.long, device="cuda")).float().sigmoid().cpu()
                              for s in range(0, len(x), 32)]).numpy()
        gt = np.stack(labels)
        score = miou_batch(prob > 0.5, gt)
        const = miou_batch(np.ones_like(gt), gt)
        diff = float(np.mean([np.abs(o.astype(np.float32) - s).mean() for o, s in zip(outs, images)]))
        print("DIAG", method, domain, "n", len(images), "mIoU", round(float(score.mean()), 4), "all-ones baseline", round(float(const.mean()), 4),
              "mean |watermark| /255", round(diff, 3), flush=True)
    del wm
    torch.cuda.empty_cache()
print("DIAG_DONE", flush=True)
