"""GPU probe: per-position agreement of the kit extractors with the KNOWN message, as a localisation map.

One forward pass per image in the full frame: logits before pooling, signed by the message and averaged over bits.
Scored by per-image AUC against the true mask. Nothing is trained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

kit = Path(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

rows = json.loads((kit / "release/train/metadata.json").read_text())
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def auc(s, label):
    label = np.asarray(label).ravel().astype(bool); s = np.asarray(s, np.float64).ravel()
    n1 = int(label.sum()); n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


for method in ("pixelseal", "trustmark"):
    wm = load_watermark(method, models_dir=kit / "models", device=device)
    stats = []
    for r in rows:
        if r["method"] != method:
            continue
        x = torch.from_numpy(np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB")))
        x = x.permute(2, 0, 1)[None].float().div(255).to(device)
        sign = torch.tensor([int(b) for b in r["message"]], dtype=torch.float32, device=device) * 2 - 1
        with torch.no_grad():
            pp = wm.features(x)["per_position_logits"].float()          # 1 x nbits x h x w
        signed = pp[0] * sign[:, None, None]
        maps = {"mean": signed.mean(0), "acc": (signed > 0).float().mean(0), "tanh": torch.tanh(signed).mean(0)}
        gt = masks[r["image_name"]]
        rec = {"full": bool(r["full_watermark"]), "shape": list(pp.shape[-2:])}
        for k, v in maps.items():
            up = F.interpolate(v[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            rec[k] = auc(up, gt)
        stats.append(rec)
    for full in (True, False):
        sub = [s for s in stats if s["full"] == full]
        print("POSAGREE", method, "full" if full else "sparse", len(sub), "map", sub[0]["shape"],
              {k: round(float(np.mean([s[k] for s in sub if s[k] is not None])), 3) for k in ("mean", "acc", "tanh")}, flush=True)
    del wm
    torch.cuda.empty_cache()
print("POSAGREE_DONE", flush=True)
