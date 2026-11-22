"""Does a slight smooth spatial drift of the picture reproduce the official full-frame signatures?

Official full-frame pictures keep the decoder margin (TrustMark 1.06x, PixelSeal 0.88x, StegaStamp 0.87x of a fresh
embedding) but lose the match with the kit template (TrustMark 0.08x, PixelSeal about 0, StegaStamp 0.6-0.8x).
A generative model that re-synthesises the kept pixels can move fine detail by a pixel or two. Here fresh
embeddings are warped by a random smooth displacement field of given RMS amplitude and both numbers are measured.
usage: warp_probe.py <kit> <dir with fresh_<method>.npz> <method>
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

kit, folder, method = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
if method != "stegastamp":
    from wm_template import template as fast_template  # noqa: E402
gen = torch.Generator(device="cuda").manual_seed(0)


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def tmpl(x, bits):
    if method != "stegastamp":
        return fast_template(wm, x, bits)
    out = []
    for im, b in zip((x * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy(), bits.cpu().numpy()):
        good = wm.embed(im, b.tolist()).astype(np.float32); bad = wm.embed(im, (1 - b).tolist()).astype(np.float32)
        out.append((good - bad) / 2)
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).cuda()


def warp(x, rms, grid=16):
    """Random smooth displacement: a coarse field of grid x grid vectors, bicubically upsampled, RMS amplitude in pixels."""
    if rms == 0:
        return x
    n, _, h, w = x.shape
    d = torch.randn(n, 2, grid, grid, device=x.device, generator=gen)
    d = F.interpolate(d, size=(h, w), mode="bicubic", align_corners=False)
    d = d / d.pow(2).mean((1, 2, 3), keepdim=True).sqrt() * rms
    ys, xs = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    gx = (xs[None] + d[:, 0] + 0.5) / w * 2 - 1
    gy = (ys[None] + d[:, 1] + 0.5) / h * 2 - 1
    out = F.grid_sample(x, torch.stack([gx, gy], -1), mode="bicubic", padding_mode="reflection", align_corners=False)
    return (out.clamp(0, 1) * 255).round() / 255


def signature(tag, x, bits, ref=None):
    lg = torch.cat([torch.as_tensor(wm.logits(x[s:s + 8])).float().cpu() for s in range(0, len(x), 8)])
    sign = bits.float().cpu() * 2 - 1
    acc, margin = float(((lg > 0) == (sign > 0)).float().mean()), float((lg * sign).mean())
    t = tmpl(x, bits)
    res = {}
    for lo, hi in ((1, 5), (5, 17)):
        a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
        res[f"{lo}-{hi}"] = float(((a * b).flatten(1).sum(1) / (b * b).flatten(1).sum(1)).median())
    line = {"bit acc": round(acc, 4), "margin": round(margin, 2)}
    if ref:
        line.update({"margin vs fresh": round(margin / ref["margin"], 2), **{f"template {k} vs fresh": round(res[k] / ref[k], 2) for k in res}})
    print("WARP", method, f"{tag:22s}", line, flush=True)
    return {"margin": margin, **res}


with np.load(folder / f"fresh_{method}.npz") as z:
    W, bits = z["W"][:24], z["bits"][:24].astype(np.int64)
x = torch.from_numpy(W).permute(0, 3, 1, 2).cuda().float() / 255
b = torch.from_numpy(bits).cuda()
ref = signature("fresh", x, b)
for grid in (16, 48):
    for rms in (0.5, 1.0, 2.0, 3.0):
        signature(f"warp rms {rms} grid {grid}", warp(x, rms, grid), b, ref)
print("WARP_DONE (official: TrustMark margin 1.06 template 0.08 | PixelSeal 0.88 / ~0 | StegaStamp 0.87 / 0.6-0.8)", flush=True)
