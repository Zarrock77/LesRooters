"""Is the watermark of the official full-frame images written in the frame we assume?

The message template correlates with the residual on pictures we embed ourselves, but the learned matched filter
transfers badly to official pictures. Here: cross-correlation surface between the picture and its template over
all shifts (FFT), for official full-frame images and for self-embedded controls, plus the same statistic when the
template is computed in a rescaled frame. A peak away from zero shift, or only at another scale, means the
official watermark was embedded in a different frame (other resolution, crop or resize after embedding).
usage: xcorr_probe.py <kit> <clean.npy> <frame json> <method> [n]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from wm_template import embed, template  # noqa: E402

kit, clean_path, frame_json, method = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 32
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(2)


def band(v, k=9):
    return v - F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def surface(x, t):
    """Normalised cross-correlation of picture and template for every cyclic shift, [B,H,W] (zero shift at 0,0)."""
    a, b = band(x), band(t)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    return c / (a.flatten(1).norm(dim=1) * b.flatten(1).norm(dim=1) + 1e-12)[:, None, None]


def describe(tag, x, bits):
    t = template(wm, x, bits)
    s = surface(x, t)
    flat = s.flatten(1)
    z0 = ((s[:, 0, 0] - flat.mean(1)) / flat.std(1)).cpu().numpy()
    peak = flat.argmax(1).cpu().numpy()
    py, px = peak // s.shape[-1], peak % s.shape[-1]
    py = np.where(py > s.shape[-2] // 2, py - s.shape[-2], py); px = np.where(px > s.shape[-1] // 2, px - s.shape[-1], px)
    zmax = ((flat.max(1).values - flat.mean(1)) / flat.std(1)).cpu().numpy()
    wrong = surface(x, template(wm, x, torch.randint_like(bits, 2)))
    zw = ((wrong[:, 0, 0] - wrong.flatten(1).mean(1)) / wrong.flatten(1).std(1)).cpu().numpy()
    print("XCORR", method, tag, "n", len(x), "| z at zero shift: median", round(float(np.median(z0)), 2), "mean", round(float(z0.mean()), 2),
          "| wrong message:", round(float(np.median(zw)), 2), "| peak at zero shift in", int(((py == 0) & (px == 0)).sum()), "of", len(x),
          "| median peak z", round(float(np.median(zmax)), 2), "| peaks (dy,dx)", list(zip(py.tolist(), px.tolist()))[:10], flush=True)
    return float(np.median(z0))


# official full-frame train images
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8][:n]
x_real = torch.stack([images[i] for i in sel]).permute(0, 3, 1, 2).cuda().float() / 255
b_real = torch.tensor([[int(c) for c in rows[i]["message"]] for i in sel]).cuda()
# self-embedded controls: batched float embedder, and the kit's own embed() (uint8, PIL path)
clean = np.load(clean_path, mmap_mode="r")
idx = rng.choice(len(clean), n, replace=False)
xc = torch.from_numpy(np.stack([clean[int(i)] for i in idx])).permute(0, 3, 1, 2).cuda().float() / 255
b_syn = torch.from_numpy(rng.integers(0, 2, (n, wm.nbits))).cuda()
x_syn = (embed(wm, xc, b_syn) * 255).round() / 255
x_kit = torch.from_numpy(np.stack([wm.embed(np.asarray(clean[int(i)]), b_syn[j].tolist()) for j, i in enumerate(idx)])).permute(0, 3, 1, 2).cuda().float() / 255
describe("self-embedded (batched)", x_syn, b_syn)
describe("self-embedded (kit embed)", x_kit, b_syn)
describe("clean control", xc, b_syn)
describe("official full-frame", x_real, b_real)
# template computed in a rescaled frame: the official watermark may have been embedded at another size
for scale in (0.5, 0.75, 0.9, 1.1, 1.25, 1.5, 2.0):
    size = int(round(512 * scale))
    xs = F.interpolate(x_real, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    t = F.interpolate(template(wm, xs, b_real), size=(512, 512), mode="bilinear", align_corners=False, antialias=True)
    s = surface(x_real, t)
    z0 = ((s[:, 0, 0] - s.flatten(1).mean(1)) / s.flatten(1).std(1)).cpu().numpy()
    print("XCORR", method, "official, template built at", size, "px: median z at zero shift", round(float(np.median(z0)), 2), flush=True)
print("XCORR_DONE", flush=True)
