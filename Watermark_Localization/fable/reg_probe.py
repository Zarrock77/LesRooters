"""Search the frame in which the official full-frame watermark was embedded.

The official watermark decodes at full strength but does not match the kit template computed on the released
picture, while our own embedding on the same pictures does (control_probe.py). If the picture was rescaled or
shifted after embedding, the template matches again once the picture is warped back into the embedding frame.
For every candidate scale (about the centre) the z-scored correlation surfaces are averaged over images, so a
systematic shift shows up as a peak far above 1 / sqrt(n). Diagnostic only.
usage: reg_probe.py <kit> <frame json> <method> [n]
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
WIN = 20


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def zsurface(x, t):
    a, b = box(x, 1) - box(x, 5), box(t, 1) - box(t, 5)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    z = (c - flat.mean(1)[:, None, None]) / flat.std(1)[:, None, None]
    z = torch.roll(z, (WIN, WIN), (1, 2))[:, :2 * WIN + 1, :2 * WIN + 1]       # shifts -WIN..WIN, zero shift at centre
    return z


def warp(x, sx, sy):
    """Resample the picture as if the embedding frame were sx, sy times the released frame (same centre)."""
    theta = torch.tensor([[1 / sx, 0, 0], [0, 1 / sy, 0]], device=x.device, dtype=torch.float32)[None].repeat(len(x), 1, 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bicubic", padding_mode="reflection", align_corners=False).clamp(0, 1)


def report(tag, x, bits):
    z = zsurface(x, template(wm, x, bits)).mean(0)
    k = int(z.argmax())
    dy, dx = k // z.shape[1] - WIN, k % z.shape[1] - WIN
    print("REG", method, f"{tag:26s}", "mean z at zero shift", round(float(z[WIN, WIN]), 2), "| best mean z", round(float(z.max()), 2),
          "at (dy,dx)", (dy, dx), flush=True)
    return float(z.max())


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as zf:
    masks = dict(zip(zf["names"].tolist(), zf["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
sel = sorted(sel, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:n]
x = images[sel].permute(0, 3, 1, 2).cuda().float() / 255
bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in sel]).cuda()
print("REG noise level of a mean surface:", round(1 / np.sqrt(len(sel)), 2), flush=True)
ours = (embed(wm, x, 1 - bits) * 255).round() / 255
report("control: our own mark", ours, 1 - bits)
report("control: ours, warped 1.03", warp(ours, 1.03, 1.03), 1 - bits)
best = []
for s in np.round(np.arange(0.88, 1.1201, 0.01), 2):
    best.append((report(f"official, scale {s}", x if s == 1.0 else warp(x, float(s), float(s)), bits), float(s), float(s)))
for sx, sy in ((1.0, 0.95), (1.0, 1.05), (0.95, 1.0), (1.05, 1.0), (1.0, 0.9), (0.9, 1.0), (1.0, 1.1), (1.1, 1.0)):
    best.append((report(f"official, scale x{sx} y{sy}", warp(x, sx, sy), bits), sx, sy))
# flips, in case a mirrored picture was watermarked or released
for name, f in (("flip lr", lambda v: v.flip(3)), ("flip ud", lambda v: v.flip(2)), ("transpose", lambda v: v.transpose(2, 3))):
    best.append((report(f"official, {name}", f(x), bits), name, name))
print("REG_BEST", method, sorted(best, key=lambda b: -b[0])[:3], flush=True)
print("REG_DONE", flush=True)
