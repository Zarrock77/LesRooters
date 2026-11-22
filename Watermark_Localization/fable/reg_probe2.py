"""Is the released full-frame picture a CROP of the frame that was watermarked?

reg_probe.py excluded small rescales, shifts and flips. Here the embedding frame is allowed to be much larger
than the released picture, separately in x and y (a non-square original cropped to a square, or a larger square
cropped): the picture is placed in the centre of the candidate frame, the template is computed in that frame,
and z-scored correlation surfaces are averaged over images. Diagnostic only.
usage: reg_probe2.py <kit> <frame json> <method> [n]
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

kit, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 24
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
WIN = 24
SCALES = (1.0, 1.125, 1.2, 1.25, 1.333, 1.5, 1.6, 1.778, 2.0)


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def zsurface(x, t):
    a, b = box(x, 1) - box(x, 5), box(t, 1) - box(t, 5)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    z = (c - flat.mean(1)[:, None, None]) / flat.std(1)[:, None, None]
    return torch.roll(z, (WIN, WIN), (1, 2))[:, :2 * WIN + 1, :2 * WIN + 1]


def warp(x, sx, sy):
    theta = torch.tensor([[sx, 0, 0], [0, sy, 0]], device=x.device, dtype=torch.float32)[None].repeat(len(x), 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bicubic", padding_mode="reflection", align_corners=False).clamp(0, 1)


def score(x, bits):
    z = zsurface(x, template(wm, x, bits)).mean(0)
    k = int(z.argmax())
    return float(z[WIN, WIN]), float(z.max()), (k // z.shape[1] - WIN, k % z.shape[1] - WIN)


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as zf:
    masks = dict(zip(zf["names"].tolist(), zf["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
sel = sorted(sel, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:n]
x = images[sel].permute(0, 3, 1, 2).cuda().float() / 255
bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in sel]).cuda()
# control: our own mark embedded in a 4:3 frame around the picture, then only the central square kept
big = warp(x, 1.333, 1.0)                              # picture squeezed into the centre of a wider frame
ours = embed(wm, big, 1 - bits)
crop = F.interpolate(ours[:, :, :, 64:448], size=(512, 512), mode="bicubic", align_corners=False).clamp(0, 1)
print("REG2", method, "control (ours, 4:3 frame cropped): assumed square frame", score(crop, 1 - bits),
      "| assumed 4:3 frame", score(warp(crop, 1.333, 1.0), 1 - bits), flush=True)
results = []
for sx in SCALES:
    for sy in SCALES:
        z0, zmax, where = score(x if sx == sy == 1.0 else warp(x, sx, sy), bits)
        results.append((zmax, z0, sx, sy, where))
for zmax, z0, sx, sy, where in sorted(results, reverse=True)[:8]:
    print("REG2", method, "frame x", sx, "y", sy, "| mean z at zero shift", round(z0, 2), "| best", round(zmax, 2), "at", where, flush=True)
print("REG2_DONE", flush=True)
