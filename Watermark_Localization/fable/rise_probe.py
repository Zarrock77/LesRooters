"""GPU probe: decoder evidence per cell from many random half-destroyed versions of the image (group testing).

For one image in the full frame and its KNOWN message: destroy a random half of the cells (heavy blur), read the
kit decoder's agreement with the message, repeat N times, and regress the agreement on which cells were kept.
A cell that carries the watermark contributes positive agreement when kept; an edited or clean cell contributes
nothing. Every forward pass informs all cells, unlike one-window-at-a-time occlusion. Nothing is trained.
Reports per-image AUC against the true mask on official train images.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

kit = Path(sys.argv[1])
device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 30
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

rows = json.loads((kit / "release/train/metadata.json").read_text())
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


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


def miou(p, g):
    s = 0.0
    for c in (0, 1):
        a, b = p == c, g == c
        u = (a | b).sum()
        s += 1.0 if u == 0 else (a & b).sum() / u
    return s / 2


@torch.no_grad()
def evidence(wm, method, x, sign, G, N, keep=0.5, destroyer="blur"):
    d = gauss(x, 12.0) if destroyer == "blur" else x.mean((2, 3), keepdim=True).expand_as(x)
    gen = torch.Generator(device=device).manual_seed(0)
    A = (torch.rand(N, G * G, device=device, generator=gen) < keep).float()
    y = []
    for s in range(0, N, 256):
        m = F.interpolate(A[s:s + 256].view(-1, 1, G, G), size=(512, 512), mode="nearest")
        batch = d + m * (x - d)
        if method == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16):
                lg = wm.net(wm._prep(batch)).float()
        else:
            lg = wm.det(wm._resize(batch))[:, 1:].float()
        y.append((lg * sign).mean(1))
    y = torch.cat(y)
    Ac = A - A.mean(0, keepdim=True)
    w = torch.linalg.solve(Ac.T @ Ac + 1e-2 * N * torch.eye(G * G, device=device), Ac.T @ (y - y.mean()))
    return w.view(G, G)


for method in ("pixelseal", "trustmark"):
    wm = load_watermark(method, models_dir=kit / "models", device=device)
    for group, want_full in (("full", True), ("sparse", False)):
        sub = [r for r in rows if r["method"] == method and bool(r["full_watermark"]) == want_full][:limit]
        res = {}
        t0 = time.time()
        for r in sub:
            x = torch.from_numpy(np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB")))
            x = x.permute(2, 0, 1)[None].float().div(255).to(device)
            sign = torch.tensor([int(b) for b in r["message"]], dtype=torch.float32, device=device) * 2 - 1
            gt = masks[r["image_name"]]
            for G, N, destroyer in ((16, 2048, "blur"), (16, 2048, "mean"), (8, 1024, "blur"), (32, 4096, "blur")):
                w = evidence(wm, method, x, sign, G, N, destroyer=destroyer)
                up = F.interpolate(w[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
                key = f"G{G}_N{N}_{destroyer}"
                res.setdefault(key, {"auc": [], "oracle": []})
                res[key]["auc"].append(auc(up, gt))
                res[key]["oracle"].append(max(miou(up > q, gt) for q in np.quantile(up, np.linspace(0.02, 0.98, 49))))
        const = float(np.mean([max(masks[r["image_name"]].mean(), 1 - masks[r["image_name"]].mean()) / 2 for r in sub]))
        for key, v in res.items():
            print("RISE", method, group, len(sub), key, "AUC", round(float(np.mean([a for a in v["auc"] if a is not None])), 3),
                  "oracle-threshold mIoU", round(float(np.mean(v["oracle"])), 3), "constant", round(const, 3),
                  round(time.time() - t0, 1), "s", flush=True)
    del wm
    torch.cuda.empty_cache()
print("RISE_DONE", flush=True)
