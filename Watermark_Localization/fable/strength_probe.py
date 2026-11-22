"""GPU probe: (1) fresh embedding on competition-domain content, (2) weaker watermark (S + alpha (W - S)).

Compares the template coefficient beta and the decoder margin with the values measured on the official
full-frame images (beta 0.14 / 0.15 / 0.36, margin 9.0 / 7.9 / 6.1 for PixelSeal / TrustMark / StegaStamp).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

kit, folder = Path(sys.argv[1]), Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
sys.path.insert(0, str(kit))
import onnxruntime as ort  # noqa: E402
ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402

rows = json.loads((kit / "release/train/metadata.json").read_text())
rng = np.random.default_rng(11)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().to(device)


def box(t, k):
    return F.avg_pool2d(t, k, 1, k // 2, count_include_pad=False)


def beta(x, T, hp=7, win=33):
    hx, hT = x - box(x, hp), T - box(T, hp)
    cov = box((hx * hT).mean(1, keepdim=True), win); vT = box((hT * hT).mean(1, keepdim=True), win)
    return float((cov / (vT + 1e-10)).clamp(-4, 4).mean())


def measure(wm, x, bits):
    good = np.asarray(wm.embed(x, bits.tolist()), dtype=np.float32)
    bad = np.asarray(wm.embed(x, (1 - bits).tolist()), dtype=np.float32)
    sign = torch.tensor(bits, dtype=torch.float32, device=device) * 2 - 1
    return beta(to_t(x) / 255, to_t(good - bad) / 510), float((wm.logits(to_t(x) / 255).to(device)[0] * sign).mean())


for method in ("pixelseal", "trustmark", "stegastamp"):
    wm = load_watermark(method, models_dir=kit / "models", device=device)
    comp = [np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB"))
            for r in rows if not r["full_watermark"] and r["method"] != method][:24]
    res = [measure(wm, np.asarray(wm.embed(S, b.tolist()), dtype=np.uint8), b) for S in comp for b in [rng.integers(0, 2, wm.nbits)]]
    print("STRENGTH", method, "competition content, fresh", "beta", round(float(np.mean([r[0] for r in res])), 3),
          "margin", round(float(np.mean([r[1] for r in res])), 2), flush=True)
    with np.load(folder / f"fresh_{method}.npz") as z:
        S, W, bits = z["S"].astype(np.float32), z["W"].astype(np.float32), z["bits"]
    for alpha in (0.25, 0.5, 0.75):
        res = [measure(wm, np.clip(np.rint(s + alpha * (w - s)), 0, 255).astype(np.uint8), b) for s, w, b in zip(S, W, bits)]
        print("STRENGTH", method, f"coco, alpha {alpha}", "beta", round(float(np.mean([r[0] for r in res])), 3),
              "margin", round(float(np.mean([r[1] for r in res])), 2), flush=True)
    del wm
    torch.cuda.empty_cache()
print("STRENGTH_DONE", flush=True)
