"""GPU probe: which post-processing of a fresh watermark reproduces what is measured on the official images?

Official full-frame images: template coefficient beta about 0.14 (PixelSeal), 0.15 (TrustMark), 0.36 (StegaStamp),
decoder margins about 9.0 / 7.9 / 6.1. Fresh kit embeddings: beta 0.27 / 0.63 / 0.74, margins 10.8 / 8.6 / 7.9.
Each candidate transform is applied to fresh watermarked COCO images, then beta and margin are measured exactly
as on the official images (template from re-embedding the known message on the transformed image).
Optional extra arrays (e.g. VAE round trips made in another environment) are read from <dir>/extra_<method>.npz.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

kit, folder = Path(sys.argv[1]), Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
sys.path.insert(0, str(kit))
import onnxruntime as ort  # noqa: E402
ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().to(device)


def box(t, k):
    return F.avg_pool2d(t, k, 1, k // 2, count_include_pad=False)


def beta(x, T, hp=7, win=33):
    hx, hT = x - box(x, hp), T - box(T, hp)
    cov = box((hx * hT).mean(1, keepdim=True), win); vT = box((hT * hT).mean(1, keepdim=True), win)
    return float((cov / (vT + 1e-10)).clamp(-4, 4).mean())


def jpeg(a, q):
    buf = io.BytesIO(); Image.fromarray(a).save(buf, "JPEG", quality=q); return np.array(Image.open(buf).convert("RGB"))


def resize(a, s, back=Image.Resampling.BICUBIC):
    im = Image.fromarray(a)
    return np.array(im.resize((int(512 * s), int(512 * s)), Image.Resampling.LANCZOS).resize((512, 512), back))


TRANSFORMS = {
    "none": lambda a: a,
    "jpeg95": lambda a: jpeg(a, 95), "jpeg85": lambda a: jpeg(a, 85), "jpeg70": lambda a: jpeg(a, 70),
    "down0.75_up": lambda a: resize(a, 0.75), "down0.5_up": lambda a: resize(a, 0.5), "up2_down": lambda a: resize(a, 2.0, Image.Resampling.LANCZOS),
    "blur0.6": lambda a: np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(0.6))),
    "blur1.0": lambda a: np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(1.0))),
    "shift1px": lambda a: np.roll(a, (1, 1), (0, 1)),
}
for method in ("pixelseal", "trustmark", "stegastamp"):
    wm = load_watermark(method, models_dir=kit / "models", device=device)
    with np.load(folder / f"fresh_{method}.npz") as z:
        S, W, bits = z["S"], z["W"], z["bits"]
    only_extra = len(sys.argv) > 4 and sys.argv[4] == "extra"
    variants = {} if only_extra else {name: [f(w) for w in W] for name, f in TRANSFORMS.items()}
    extra = folder / f"extra_{method}.npz"
    if extra.exists():
        with np.load(extra) as z:
            for key in z.files:
                variants[key] = list(z[key])
    for name, images in variants.items():
        b, m, d = [], [], []
        for x, w, bt in zip(images, W, bits):
            good = np.asarray(wm.embed(x, bt.tolist()), dtype=np.float32)
            bad = np.asarray(wm.embed(x, (1 - bt).tolist()), dtype=np.float32)
            sign = torch.tensor(bt, dtype=torch.float32, device=device) * 2 - 1
            b.append(beta(to_t(x) / 255, to_t(good - bad) / 510))
            m.append(float((wm.logits(to_t(x) / 255).to(device)[0] * sign).mean()))
            d.append(float(np.abs(x.astype(np.float32) - w).mean()))
        print("DEGRADE", method, f"{name:14s}", "beta", round(float(np.mean(b)), 3), "margin", round(float(np.mean(m)), 2),
              "mean |change| /255", round(float(np.mean(d)), 2), flush=True)
    del wm
    torch.cuda.empty_cache()
print("DEGRADE_DONE", flush=True)
