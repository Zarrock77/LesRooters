"""CPU-only probe of how the train set was built (read-only on the kit, writes only under its own output dir).

1. Does the known message decode from the full frame / from the exact mask bounding box, and how fast does the
   signal decay when the crop is misaligned?
2. Does the re-embedded message template correlate locally with the image where the mask says "watermarked"?
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
from scipy import ndimage

kit, out, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
sys.path.insert(0, str(kit))
torch.set_num_threads(2)
from wmloc.watermarks import load_watermark  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
rows = [r for r in json.loads((kit / "release/train/metadata.json").read_text()) if r["method"] == method][:limit]
wm = load_watermark(method, models_dir=kit / "models", device="cpu")


def read(name):
    x = np.array(Image.open(kit / "release/train/images" / (name + ".png")).convert("RGB"))
    m = np.array(Image.open(kit / "release/train/masks" / (name + ".png")).convert("L")) > 127
    return x, m


def score(x, box, bits):
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(512, int(round(x1))), min(512, int(round(y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    t = torch.from_numpy(x[y0:y1, x0:x1].copy()).permute(2, 0, 1)[None].float() / 255
    lg = wm.logits(t)[0].cpu().numpy()
    return [float(((lg > 0) == (bits > 0.5)).mean()), float(((bits * 2 - 1) * lg).mean())]


def kind_of(m):
    ys, xs = np.where(m)
    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    fill = m.sum() / ((y1 - y0) * (x1 - x0))
    if fill > 0.999:
        k = "pure_rectangle"
    elif x1 - x0 == 512 and y1 - y0 == 512:
        k = "fullframe_with_holes"
    elif fill > 0.6:
        k = "rectangle_with_holes"
    else:
        k = "irregular_object"
    return k, (int(x0), int(y0), int(x1), int(y1)), float(fill)


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


def template(x, bits):
    b = [int(v) for v in bits]
    good = wm.embed(x, b).astype(np.float32)
    bad = wm.embed(x, [1 - v for v in b]).astype(np.float32)
    return (good - bad) / 2 / 255, (good - x.astype(np.float32)) / 255


def corr_maps(x, T):
    xf = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float() / 255
    Tf = torch.from_numpy(T.copy()).permute(2, 0, 1)[None].float()
    res = {}
    for hp in (3, 7):
        hx = xf - F.avg_pool2d(xf, hp, 1, hp // 2, count_include_pad=False)
        hT = Tf - F.avg_pool2d(Tf, hp, 1, hp // 2, count_include_pad=False)
        xT, xx, TT = (hx * hT).mean(1, keepdim=True), (hx * hx).mean(1, keepdim=True), (hT * hT).mean(1, keepdim=True)
        for win in (17, 33):
            sm = lambda v: F.avg_pool2d(v, win, 1, win // 2, count_include_pad=False)  # noqa: E731
            cov, vx, vT = sm(xT), sm(xx), sm(TT)
            res[f"ncc_hp{hp}_w{win}"] = (cov / (vx * vT + 1e-14).sqrt())[0, 0].numpy()
            res[f"beta_hp{hp}_w{win}"] = (cov / (vT + 1e-10)).clamp(-4, 4)[0, 0].numpy()
    return res


def down(a):
    h, w = a.shape
    return a[: h // 4 * 4, : w // 4 * 4].reshape(h // 4, 4, w // 4, 4).mean((1, 3))


records, saved_maps, saved_names = [], [], []
t0 = time.time()
with open(out / f"{method}.jsonl", "w") as stream:
    for index, r in enumerate(rows):
        x, m = read(r["image_name"])
        bits = np.array([int(b) for b in r["message"]], dtype=np.float32)
        kind, (x0, y0, x1, y1), fill = kind_of(m)
        w, h = x1 - x0, y1 - y0
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        rec = {"name": r["image_name"], "method": method, "full": bool(r["full_watermark"]), "kind": kind,
               "bbox": [x0, y0, x1, y1], "fill": fill, "area": float(m.mean())}
        dec = {"frame": score(x, (0, 0, 512, 512), bits), "bbox": score(x, (x0, y0, x1, y1), bits)}
        if r["full_watermark"]:
            for s in (0.5, 0.75):
                dec[f"center_{s}"] = score(x, (256 - 256 * s, 256 - 256 * s, 256 + 256 * s, 256 + 256 * s), bits)
            dec["quadrant"] = score(x, (0, 0, 256, 256), bits)
        else:
            for f in (0.1, 0.25, 0.5):
                dec[f"shiftx_{f}"] = score(x, (x0 + f * w, y0, x1 + f * w, y1), bits)
                dec[f"shifty_{f}"] = score(x, (x0, y0 + f * h, x1, y1 + f * h), bits)
            for s in (0.7, 0.85, 1.15, 1.3, 1.6, 2.0, 3.0):
                dec[f"scale_{s}"] = score(x, (cx - w * s / 2, cy - h * s / 2, cx + w * s / 2, cy + h * s / 2), bits)
            for p in (-4, 4, 8, 16):
                dec[f"pad_{p}"] = score(x, (x0 - p, y0 - p, x1 + p, y1 + p), bits)
        rec["decode"] = dec

        tm = {}
        rec["template"] = tm
        records.append(rec)
        stream.write(json.dumps(rec) + "\n")
        stream.flush()
        if index % 10 == 0:
            print("PROBE", method, index + 1, "/", len(rows), round(time.time() - t0, 1), "s", flush=True)


def mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None


print("\n=== DECODE (bit accuracy, margin) ===")
for kind in ("fullframe_with_holes", "pure_rectangle", "rectangle_with_holes", "irregular_object"):
    sub = [r for r in records if r["kind"] == kind]
    if not sub:
        continue
    print(f"{method} {kind} n={len(sub)} area={mean([r['area'] for r in sub])}")
    for key in sub[0]["decode"]:
        vals = [r["decode"].get(key) for r in sub]
        vals = [v for v in vals if v is not None]
        if vals:
            acc = np.array([v[0] for v in vals])
            print(f"   {key:12s} acc mean={acc.mean():.3f} p10={np.quantile(acc, .1):.3f} p50={np.median(acc):.3f} "
                  f"p90={np.quantile(acc, .9):.3f} margin={np.mean([v[1] for v in vals]):.3f}")
print("\n=== TEMPLATE (per-image AUC of local statistic vs mask; mean inside / outside) ===")
for kind in ("fullframe_with_holes", "pure_rectangle", "rectangle_with_holes", "irregular_object"):
    sub = [r for r in records if r["kind"] == kind]
    if not sub:
        continue
    keys = sorted({k for r in sub for k in r["template"]})
    for key in keys:
        vals = [r["template"][key] for r in sub if key in r["template"]]
        print(f"{method} {kind:22s} {key:22s} n={len(vals):3d} auc={mean([v['auc'] for v in vals])} "
              f"in={mean([v['in'] for v in vals])} out={mean([v['out'] for v in vals])}")
print("PROBE_DONE", method, round(time.time() - t0, 1), "s", flush=True)
