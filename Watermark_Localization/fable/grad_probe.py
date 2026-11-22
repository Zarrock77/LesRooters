"""CPU-only probe: integrated-gradient attribution of the decoder's message margin (full frame, known message).

Idea: where the watermark is present, the high-frequency content of the image pushes the decoder towards the
known message; where it was edited away, it does not. No training, no labels used to build the map.
Writes only under its own output dir. Read-only on the kit.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

kit, out, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
sys.path.insert(0, str(kit))
torch.set_num_threads(2)
from wmloc.watermarks import load_watermark  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
rows = [r for r in json.loads((kit / "release/train/metadata.json").read_text()) if r["method"] == method]
wm = load_watermark(method, models_dir=kit / "models", device="cpu")
for p in (wm.net if method == "trustmark" else wm.det).parameters():
    p.requires_grad_(False)


def logits_fn(xt):
    if method == "trustmark":
        return wm.net(wm._prep(xt)).float()
    return wm.det(wm._resize(xt))[:, 1:].float()


def read(name):
    x = np.array(Image.open(kit / "release/train/images" / (name + ".png")).convert("RGB"))
    m = np.array(Image.open(kit / "release/train/masks" / (name + ".png")).convert("L")) > 127
    return x, m


def gaussian_blur(t, sigma):
    k = int(sigma * 3) * 2 + 1
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    t = F.pad(t, (k // 2,) * 4, mode="reflect")
    t = F.conv2d(t, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(t, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def auc(s, label):
    label = label.ravel().astype(bool)
    s = s.ravel()
    n1 = int(label.sum())
    n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def miou(pred, gt):
    vals = []
    for c in (0, 1):
        p, g = pred == c, gt == c
        u = (p | g).sum()
        vals.append(1.0 if u == 0 else (p & g).sum() / u)
    return float(np.mean(vals))


def best_threshold_miou(score, gt):
    qs = np.quantile(score, np.linspace(0.01, 0.99, 50))
    return max(miou((score > q), gt) for q in qs)


K = 8
recs, maps, names = [], [], []
t0 = time.time()
with open(out / f"{method}.jsonl", "w") as stream:
    for index, r in enumerate(rows[:limit]):
        x, m = read(r["image_name"])
        ys, xs = np.where(m)
        fill = m.sum() / ((ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()))
        group = "full" if r["full_watermark"] else ("rect" if fill > 0.999 else "transplant")
        if group == "rect":
            continue
        bits = torch.tensor([int(b) for b in r["message"]], dtype=torch.float32)
        sign = (bits * 2 - 1)[None]
        xt = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float() / 255
        m128 = m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
        rec = {"name": r["image_name"], "group": group, "area": float(m128.mean())}
        stack = []
        for sigma in (1.5, 4.0):
            xb = gaussian_blur(xt, sigma)
            diff = xt - xb
            grad = torch.zeros_like(xt)
            for k in range(K):
                a = (k + 0.5) / K
                xi = (xb + a * diff).clone().requires_grad_(True)
                mg = (logits_fn(xi) * sign).mean()
                grad += torch.autograd.grad(mg, xi)[0]
            ig = (diff * grad / K).sum(1, keepdim=True)            # 1x1x512x512
            with torch.no_grad():
                m_full = float((logits_fn(xt) * sign).mean()); m_base = float((logits_fn(xb) * sign).mean())
            rec[f"s{sigma}"] = {"M": m_full, "M_base": m_base, "ig_sum": float(ig.sum())}
            for win in (16, 32, 64):
                sm = F.avg_pool2d(ig, win + 1, 1, win // 2, count_include_pad=False)
                v = F.avg_pool2d(sm, 4)[0, 0].numpy()
                rec[f"s{sigma}"][f"w{win}"] = {"auc": auc(v, m128), "best_miou": best_threshold_miou(v, m128),
                                               "in": float(v[m128].mean()), "out": float(v[~m128].mean())}
                stack.append(v.astype(np.float32))
        recs.append(rec)
        maps.append(np.stack(stack)); names.append(r["image_name"])
        stream.write(json.dumps(rec) + "\n"); stream.flush()
        if len(recs) % 5 == 0:
            print("GRAD", method, len(recs), index + 1, "/", len(rows), round(time.time() - t0, 1), "s", flush=True)
np.savez_compressed(out / f"{method}_ig_maps.npz", names=np.array(names), maps=np.stack(maps))

for group in ("full", "transplant"):
    sub = [r for r in recs if r["group"] == group]
    if not sub:
        continue
    base = np.mean([max(r["area"], 1 - r["area"]) / 2 for r in sub])
    print(f"\n== {method} {group} n={len(sub)} constant-prediction mIoU={base:.3f}")
    for sigma in (1.5, 4.0):
        s = f"s{sigma}"
        print(f"  sigma {sigma}: M={np.mean([r[s]['M'] for r in sub]):.2f} M_blurred={np.mean([r[s]['M_base'] for r in sub]):.2f} "
              f"IG sum={np.mean([r[s]['ig_sum'] for r in sub]):.2f}")
        for win in (16, 32, 64):
            w = f"w{win}"
            aucs = [r[s][w]["auc"] for r in sub if r[s][w]["auc"] is not None]
            print(f"     win {win:2d}: AUC mean={np.mean(aucs):.3f} median={np.median(aucs):.3f} p10={np.quantile(aucs, .1):.3f} | "
                  f"oracle-threshold mIoU={np.mean([r[s][w]['best_miou'] for r in sub]):.3f} | "
                  f"in={np.mean([r[s][w]['in'] for r in sub]):.2e} out={np.mean([r[s][w]['out'] for r in sub]):.2e}")
print("GRAD_DONE", method, round(time.time() - t0, 1), "s", flush=True)
