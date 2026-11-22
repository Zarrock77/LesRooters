"""CPU-only probe: can a message template (residual of the known message) locate the watermark rectangle by
FFT cross-correlation, even when the template is computed without knowing the rectangle (content independence)?

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
from PIL import Image
from scipy import ndimage

kit, out, method, limit = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
sys.path.insert(0, str(kit))
torch.set_num_threads(2)
from wmloc.watermarks import load_watermark  # noqa: E402

out.mkdir(parents=True, exist_ok=True)
rows = [r for r in json.loads((kit / "release/train/metadata.json").read_text()) if r["method"] == method]
wm = load_watermark(method, models_dir=kit / "models", device="cpu")
R = {"stegastamp": 400, "trustmark": 256, "pixelseal": 256}[method]


def read(name):
    x = np.array(Image.open(kit / "release/train/images" / (name + ".png")).convert("RGB"))
    m = np.array(Image.open(kit / "release/train/masks" / (name + ".png")).convert("L")) > 127
    return x, m


def residual_native(cover_uint8, bits):
    """Message-dependent residual at the embedder's working resolution R x R (float, /255 units)."""
    small = np.asarray(Image.fromarray(cover_uint8).resize((R, R), Image.Resampling.BICUBIC))
    b = [int(v) for v in bits]
    good = wm.embed(small, b).astype(np.float32)
    bad = wm.embed(small, [1 - v for v in b]).astype(np.float32)
    return (good - bad) / 2 / 255, (good - small.astype(np.float32)) / 255


def resize_f(a, w, h):
    return np.stack([np.asarray(Image.fromarray(a[:, :, c].astype(np.float32), mode="F").resize((w, h), Image.Resampling.BILINEAR))
                     for c in range(3)], 2)


def hp(a, sigma):
    return a - ndimage.gaussian_filter(a, sigma=(sigma, sigma, 0))


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def fft_search(I_hp, t_hp):
    """NCC of template at every valid position. I_hp: 512x512x3, t_hp: hxwx3."""
    h, w = t_hp.shape[:2]
    H, W = I_hp.shape[:2]
    corr = np.zeros((H, W), np.float64)
    for c in range(3):
        corr += np.fft.irfft2(np.fft.rfft2(I_hp[:, :, c]) * np.conj(np.fft.rfft2(t_hp[:, :, c], s=(H, W))), s=(H, W))
    e = (I_hp ** 2).sum(2)
    cs = np.pad(e, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    Hv, Wv = H - h + 1, W - w + 1
    energy = cs[h:h + Hv, w:w + Wv] - cs[:Hv, w:w + Wv] - cs[h:h + Hv, :Wv] + cs[:Hv, :Wv]
    return corr[:Hv, :Wv] / np.sqrt(np.maximum(energy, 1e-12) * (t_hp ** 2).sum() + 1e-18)


gray = np.full((R, R, 3), 127, np.uint8)
recs = []
t0 = time.time()
count = 0
with open(out / f"{method}.jsonl", "w") as stream:
    for r in rows:
        x, m = read(r["image_name"])
        ys, xs = np.where(m)
        x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        w, h = x1 - x0, y1 - y0
        pure = m.sum() / (w * h) > 0.999
        if not pure or r["full_watermark"] or w < 12 or h < 12:
            continue
        bits = [int(b) for b in r["message"]]
        crop = x[y0:y1, x0:x1]
        Tm_true, Ta_true = residual_native(crop, bits)
        Tm_gray, Ta_gray = residual_native(gray, bits)
        Tm_full, _ = residual_native(x, bits)
        rec = {"name": r["image_name"], "w": int(w), "h": int(h), "x0": int(x0), "y0": int(y0),
               "ncc_native_true_gray": ncc(Tm_true, Tm_gray), "ncc_native_true_full": ncc(Tm_true, Tm_full),
               "rms_true": float(np.sqrt((Ta_true ** 2).mean()) * 255), "rms_msg": float(np.sqrt((Tm_true ** 2).mean()) * 255)}
        xf = x.astype(np.float32) / 255
        for sigma in (1.0, 2.5):
            I_hp = hp(xf, sigma)
            for tname, T in (("true", Tm_true), ("gray", Tm_gray), ("full", Tm_full)):
                t_hp = hp(resize_f(T, int(w), int(h)), sigma)
                nmap = fft_search(I_hp, t_hp)
                py, px = np.unravel_index(np.argmax(nmap), nmap.shape)
                z = (nmap - nmap.mean()) / (nmap.std() + 1e-12)
                rec[f"s{sigma}_{tname}"] = {"ncc_at_true": float(nmap[y0, x0]), "z_at_true": float(z[y0, x0]),
                                            "peak_z": float(z[py, px]), "peak_err": [int(px - x0), int(py - y0)],
                                            "hit": bool(abs(px - x0) <= 4 and abs(py - y0) <= 4)}
            # size sensitivity of the gray template at the true position neighbourhood
            for f in (0.9, 0.95, 1.05, 1.1, 1.2):
                w2, h2 = max(8, int(round(w * f))), max(8, int(round(h * f)))
                if w2 > 512 or h2 > 512:
                    continue
                nmap = fft_search(I_hp, hp(resize_f(Tm_gray, w2, h2), sigma))
                z = (nmap - nmap.mean()) / (nmap.std() + 1e-12)
                rec[f"s{sigma}_gray_scale{f}"] = float(z.max())
        recs.append(rec)
        stream.write(json.dumps(rec) + "\n"); stream.flush()
        count += 1
        print("FFT", method, count, r["image_name"], w, h, round(time.time() - t0, 1), "s", flush=True)
        if count >= limit:
            break

print(f"\n== {method} n={len(recs)} native ncc true~gray={np.mean([r['ncc_native_true_gray'] for r in recs]):.3f} "
      f"true~full={np.mean([r['ncc_native_true_full'] for r in recs]):.3f} "
      f"rms residual={np.mean([r['rms_true'] for r in recs]):.2f}/255 msg part={np.mean([r['rms_msg'] for r in recs]):.2f}/255")
for sigma in (1.0, 2.5):
    for tname in ("true", "gray", "full"):
        k = f"s{sigma}_{tname}"
        print(f"   {k:12s} ncc@true={np.mean([r[k]['ncc_at_true'] for r in recs]):.3f} z@true={np.median([r[k]['z_at_true'] for r in recs]):.1f} "
              f"peak_z={np.median([r[k]['peak_z'] for r in recs]):.1f} hit(<=4px)={np.mean([r[k]['hit'] for r in recs]):.2f}")
    for f in (0.9, 0.95, 1.05, 1.1, 1.2):
        k = f"s{sigma}_gray_scale{f}"
        v = [r[k] for r in recs if k in r]
        print(f"   {k:22s} median peak z={np.median(v):.1f}")
print("FFT_DONE", method, round(time.time() - t0, 1), "s", flush=True)
