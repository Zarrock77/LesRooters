"""What happened to the official pictures between embedding and release?

The message template matches the watermark of self-embedded pictures (z about 5) and barely the official ones
(z about 0.8), although the decoder still reads the message. Here the template correlation is measured per
spatial frequency band, for official full-frame images and for self-embedded pictures after candidate
degradations (JPEG, down/up resampling, blur, noise), together with the decoder's bit accuracy. The degradation
whose profile matches the official one is what the synthetic training data lacks.
usage: band_probe.py <kit> <clean.npy> <frame json> <method> [n]
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from wm_template import embed, template  # noqa: E402

kit, clean_path, frame_json, method = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 32
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(2)
BANDS = [(1, 5), (5, 17), (17, 65), (65, 257)]


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def z0(x, t, lo, hi):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    return float(((c[:, 0, 0] - flat.mean(1)) / flat.std(1)).median())


def describe(tag, x, bits):
    t = template(wm, x, bits)
    lg = wm.logits(x)
    acc = float(((lg > 0) == (bits > 0)).float().mean())
    print("BAND", method, f"{tag:28s}", "bit acc", round(acc, 4), "| z per band",
          {f"{lo}-{hi}": round(z0(x, t, lo, hi), 2) for lo, hi in BANDS}, flush=True)


def jpeg(x, q):
    out = []
    for im in (x * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy():
        buf = io.BytesIO(); Image.fromarray(im).save(buf, "JPEG", quality=q)
        out.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).cuda().float() / 255


def resample(x, s, mode="bicubic"):
    size = int(round(512 * s))
    y = F.interpolate(x, size=(size, size), mode=mode, align_corners=False, antialias=True)
    return ((F.interpolate(y, size=(512, 512), mode=mode, align_corners=False, antialias=True).clamp(0, 1)) * 255).round() / 255


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8][:n]
x_real = torch.stack([images[i] for i in sel]).permute(0, 3, 1, 2).cuda().float() / 255
b_real = torch.tensor([[int(c) for c in rows[i]["message"]] for i in sel]).cuda()
clean = np.load(clean_path, mmap_mode="r")
idx = rng.choice(len(clean), n, replace=False)
xc = torch.from_numpy(np.stack([clean[int(i)] for i in idx])).permute(0, 3, 1, 2).cuda().float() / 255
bits = torch.from_numpy(rng.integers(0, 2, (n, wm.nbits))).cuda()
xw = (embed(wm, xc, bits) * 255).round() / 255
describe("official full-frame", x_real, b_real)
describe("self-embedded", xw, bits)
for q in (95, 85, 70, 50):
    describe(f"self + jpeg {q}", jpeg(xw, q), bits)
for s in (2.0, 1.5, 0.75, 0.5):
    describe(f"self + resample x{s}", resample(xw, s), bits)
describe("self + bilinear x0.5", resample(xw, 0.5, "bilinear"), bits)
for sigma in (1.0, 2.0):
    k = int(4 * sigma) * 2 + 1
    g = torch.exp(-(torch.arange(k, device="cuda") - k // 2) ** 2 / (2 * sigma ** 2)); g = (g / g.sum())
    y = F.conv2d(F.conv2d(F.pad(xw, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)
    describe(f"self + blur {sigma}", (y * 255).round() / 255, bits)
for sd in (2.0, 5.0):
    describe(f"self + noise {sd}", ((xw + torch.randn_like(xw) * sd / 255).clamp(0, 1) * 255).round() / 255, bits)
# the watermark may have been written on a larger picture that was then reduced to 512
for s in (2.0, 1.5):
    size = int(round(512 * s))
    big = F.interpolate(xc, size=(size, size), mode="bicubic", align_corners=False).clamp(0, 1)
    wbig = embed(wm, big, bits)
    for mode in ("bicubic", "bilinear", "area"):
        kw = {} if mode == "area" else {"align_corners": False, "antialias": True}
        y = (F.interpolate(wbig, size=(512, 512), mode=mode, **kw).clamp(0, 1) * 255).round() / 255
        describe(f"embed at {size}, {mode} to 512", y, bits)
print("BAND_DONE", flush=True)
