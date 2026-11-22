"""CPU-only probe: does the organisers' decoder, read in the FULL frame with the known message, say WHERE the
watermark is when only one cell is kept (rest destroyed) or only one cell is destroyed?

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
n_full, n_sparse = int(sys.argv[4]), int(sys.argv[5])
sys.path.insert(0, str(kit))
torch.set_num_threads(2)
from wmloc.watermarks import load_watermark  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
rows = [r for r in json.loads((kit / "release/train/metadata.json").read_text()) if r["method"] == method]
wm = load_watermark(method, models_dir=kit / "models", device="cpu")


def read(name):
    x = np.array(Image.open(kit / "release/train/images" / (name + ".png")).convert("RGB"))
    m = np.array(Image.open(kit / "release/train/masks" / (name + ".png")).convert("L")) > 127
    return x, m


def is_pure_rectangle(m):
    ys, xs = np.where(m)
    return m.sum() / ((ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min())) > 0.999


def gaussian_blur(t, sigma):
    k = int(sigma * 3) * 2 + 1
    ax = torch.arange(k, dtype=torch.float32) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    t = F.pad(t, (k // 2,) * 4, mode="reflect")
    t = F.conv2d(t, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(t, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def margins(batch, sign):
    res = []
    for i in range(0, len(batch), 16):
        lg = wm.logits(batch[i:i + 16])
        res.append((lg.cpu() * sign).mean(1))
    return torch.cat(res).numpy()


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


def cells(G):
    c = 512 // G
    return [(gy * c, gx * c, c) for gy in range(G) for gx in range(G)]


chosen, nf, ns = [], 0, 0
for r in rows:
    x, m = read(r["image_name"])
    if r["full_watermark"] and nf < n_full:
        chosen.append((r, x, m, "full")); nf += 1
    elif not r["full_watermark"] and not is_pure_rectangle(m) and ns < n_sparse:
        chosen.append((r, x, m, "transplant")); ns += 1
    if nf >= n_full and ns >= n_sparse:
        break

t0 = time.time()
records, maps = [], {}
with open(out / f"{method}.jsonl", "w") as stream:
    for index, (r, x, m, group) in enumerate(chosen):
        bits = torch.tensor([int(b) for b in r["message"]], dtype=torch.float32)
        sign = (bits * 2 - 1)[None]
        xt = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float() / 255
        destroyed = {"blur": gaussian_blur(xt, 6.0), "mean": xt.mean((2, 3), keepdim=True).expand_as(xt).clone()}
        rec = {"name": r["image_name"], "group": group, "area": float(m.mean()),
               "M_full": float(margins(xt, sign)[0])}
        for dname, xd in destroyed.items():
            rec[f"M0_{dname}"] = float(margins(xd, sign)[0])
        variants = [(8, "blur", "keep"), (8, "blur", "remove"), (8, "mean", "keep"), (8, "mean", "remove"),
                    (16, "blur", "remove"), (4, "blur", "keep"), (4, "blur", "remove")]
        for G, dname, mode in variants:
            xd = destroyed[dname]
            batch = []
            for (y, xx, c) in cells(G):
                if mode == "keep":
                    b = xd.clone(); b[:, :, y:y + c, xx:xx + c] = xt[:, :, y:y + c, xx:xx + c]
                else:
                    b = xt.clone(); b[:, :, y:y + c, xx:xx + c] = xd[:, :, y:y + c, xx:xx + c]
                batch.append(b)
            mg = margins(torch.cat(batch), sign).reshape(G, G)
            a = mg - rec[f"M0_{dname}"] if mode == "keep" else rec["M_full"] - mg
            cov = m.reshape(G, 512 // G, G, 512 // G).mean((1, 3))
            key = f"G{G}_{dname}_{mode}"
            rec[key] = {"auc": auc(a, cov >= 0.5), "corr": float(np.corrcoef(a.ravel(), cov.ravel())[0, 1]),
                        "mean_wm": float(a[cov >= 0.5].mean()) if (cov >= 0.5).any() else None,
                        "mean_clean": float(a[cov < 0.5].mean()) if (cov < 0.5).any() else None,
                        "std": float(a.std())}
            maps[f"{r['image_name']}|{key}"] = a.astype(np.float32)
            maps[f"{r['image_name']}|cov{G}"] = cov.astype(np.float32)
        records.append(rec)
        stream.write(json.dumps(rec) + "\n"); stream.flush()
        print("ATTRIB", method, index + 1, "/", len(chosen), group, round(time.time() - t0, 1), "s", flush=True)
np.savez_compressed(out / f"{method}_maps.npz", **maps)

for group in ("full", "transplant"):
    sub = [r for r in records if r["group"] == group]
    if not sub:
        continue
    print(f"\n== {method} {group} n={len(sub)} M_full={np.mean([r['M_full'] for r in sub]):.2f} "
          f"M0_blur={np.mean([r['M0_blur'] for r in sub]):.2f} M0_mean={np.mean([r['M0_mean'] for r in sub]):.2f}")
    for key in [k for k in sub[0] if k.startswith("G")]:
        def avg(field):
            v = [r[key][field] for r in sub if r[key][field] is not None]
            return float(np.mean(v)) if v else float("nan")
        print(f"   {key:18s} auc={avg('auc'):.3f} corr={avg('corr'):.3f} wm={avg('mean_wm'):.3f} clean={avg('mean_clean'):.3f} std={avg('std'):.3f}")
print("ATTRIB_DONE", method, round(time.time() - t0, 1), "s", flush=True)
