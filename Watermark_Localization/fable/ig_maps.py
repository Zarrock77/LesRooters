"""High-resolution decoder evidence maps by integrated gradients over a blend mask (PixelSeal, TrustMark).

The image is blended cell by cell (4x4 px cells, 128x128 grid) between a watermark-free version (mean colour or
heavy blur) and itself. Integrating the gradient of the kit decoder's message agreement along the blend gives, for
every cell, its share of the agreement: about 1 (relative density) where the watermark is present, about 0 where
it was edited away. Nothing is trained; the known message and the full frame are used.
Output per split/method: npz with names, maps [N,2,128,128] (mean baseline, blur baseline) and full margins.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--methods", default="pixelseal,trustmark")
ap.add_argument("--splits", default="train,test")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--steps", type=int, default=16)
args = ap.parse_args()
sys.path.insert(0, str(args.kit))
from wmloc.watermarks import load_watermark  # noqa: E402

dev = args.device


def gauss(t, sigma):
    k = int(sigma * 3) * 2 + 1
    ax = torch.arange(k, dtype=torch.float32, device=t.device) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    t = F.pad(t, (k // 2,) * 4, mode="reflect")
    t = F.conv2d(t, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(t, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def auc(s, label):
    label = np.asarray(label).ravel().astype(bool); s = np.asarray(s, np.float64).ravel()
    n1 = int(label.sum()); n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


args.out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for method in args.methods.split(","):
    wm = load_watermark(method, models_dir=args.kit / "models", device=dev)
    net = wm.net if method == "trustmark" else wm.det
    for p in net.parameters():
        p.requires_grad_(False)

    def margin(x, sign):
        lg = net(wm._prep(x)) if method == "trustmark" else net(wm._resize(x))[:, 1:]
        return (lg.float() * sign).mean(1)

    for split in args.splits.split(","):
        rows = [r for r in json.loads((args.kit / "release" / split / "metadata.json").read_text()) if r["method"] == method]
        masks = {}
        if split == "train":
            with np.load(args.kit / "release/train/masks_128.npz") as z:
                masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
        names, maps, base, stats = [], [], [], []
        for n, r in enumerate(rows):
            x = torch.from_numpy(np.array(Image.open(args.kit / "release" / split / "images" / (r["image_name"] + ".png")).convert("RGB")))
            x = x.permute(2, 0, 1)[None].float().div(255).to(dev)
            sign = torch.tensor([int(b) for b in r["message"]], dtype=torch.float32, device=dev) * 2 - 1
            with torch.no_grad():
                m_full = float(margin(x, sign)[0])
            channels = []
            for d in (x.mean((2, 3), keepdim=True).expand_as(x), gauss(x, 12.0)):
                grad = torch.zeros(1, 1, 128, 128, device=dev)
                for k in range(args.steps):
                    alpha = torch.full((1, 1, 128, 128), (k + 0.5) / args.steps, device=dev, requires_grad=True)
                    xa = d + F.interpolate(alpha, scale_factor=4, mode="nearest") * (x - d)
                    grad += torch.autograd.grad(margin(xa, sign).sum(), alpha)[0]
                channels.append((grad[0, 0] / args.steps) * 16384 / max(abs(m_full), 0.5))
            a = torch.stack(channels).clamp(-8, 8)
            names.append(r["image_name"]); maps.append(a.cpu().numpy().astype(np.float16)); base.append(m_full)
            if masks:
                gt = masks[r["image_name"]]
                sm = F.avg_pool2d(a[None], 9, 1, 4, count_include_pad=False)[0].cpu().numpy()
                stats.append({"full": bool(r["full_watermark"]), "auc_mean": auc(sm[0], gt), "auc_blur": auc(sm[1], gt),
                              "auc_raw": auc(a[0].cpu().numpy(), gt), "auc_sum": auc(sm[0] + sm[1], gt)})
            if n % 100 == 0:
                print("IG", method, split, n, "/", len(rows), round(time.time() - t0, 1), "s", flush=True)
        np.savez_compressed(args.out / f"{split}_{method}.npz", names=np.array(names), maps=np.stack(maps), base=np.array(base, np.float32))
        if stats:
            for full in (True, False):
                sub = [s for s in stats if s["full"] == full]
                line = {k: round(float(np.mean([s[k] for s in sub if s[k] is not None])), 3) for k in ("auc_mean", "auc_blur", "auc_raw", "auc_sum")}
                print("IG_AUC", method, "full" if full else "sparse", len(sub), json.dumps(line), flush=True)
    del wm, net
    torch.cuda.empty_cache()
(args.out / "IG_DONE").write_text("ok")
print("IG_DONE", round(time.time() - t0, 1), "s", flush=True)
