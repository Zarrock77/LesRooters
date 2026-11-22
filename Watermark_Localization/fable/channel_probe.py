"""Was the official full-frame watermark computed on a channel-permuted picture (e.g. BGR handed to the embedder)?

A script that reads pictures with OpenCV and passes BGR to an RGB embedder produces a watermark that still decodes
(the residual is close to a luminance pattern) but whose pixel pattern differs from the kit embedder's output on
the RGB picture. For each of the six channel orders: template computed on the permuted picture, permuted back,
correlated with the official full-frame pictures. Our own embedding on top of the same pictures is the control.
usage: channel_probe.py <kit> <frame json> <method> [n]
"""
import itertools
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
n = int(sys.argv[4]) if len(sys.argv) > 4 else 32
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def z(x, t, lo=1, hi=5):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    return round(float(((c[:, 0, 0] - flat.mean(1)) / flat.std(1)).median()), 2)


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as zf:
    masks = dict(zip(zf["names"].tolist(), zf["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
sel = sorted(sel, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:n]
x = images[sel].permute(0, 3, 1, 2).cuda().float() / 255
bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in sel]).cuda()
ours = (embed(wm, x, 1 - bits) * 255).round() / 255
print("CHANNEL", method, "control, our own mark (RGB):", z(ours, template(wm, ours, 1 - bits)), z(ours, template(wm, ours, 1 - bits), 5, 17), flush=True)
for perm in itertools.permutations(range(3)):
    inv = [perm.index(c) for c in range(3)]
    t = template(wm, x[:, list(perm)], bits)[:, inv]
    print("CHANNEL", method, "official, embedder fed channels", "".join("RGB"[c] for c in perm), "| z fine", z(x, t), "| z mid", z(x, t, 5, 17), flush=True)
# other input conventions an editing script could have used
for name, f, g in (("value range inverted (1 - x)", lambda v: 1 - v, lambda t: -t),
                   ("vertical flip", lambda v: v.flip(2), lambda t: t.flip(2)),
                   ("horizontal flip", lambda v: v.flip(3), lambda t: t.flip(3)),
                   ("rotated 180", lambda v: v.flip(2).flip(3), lambda t: t.flip(2).flip(3)),
                   ("transposed", lambda v: v.transpose(2, 3), lambda t: t.transpose(2, 3)),
                   ("rot90", lambda v: torch.rot90(v, 1, (2, 3)), lambda t: torch.rot90(t, -1, (2, 3))),
                   ("rot270", lambda v: torch.rot90(v, -1, (2, 3)), lambda t: torch.rot90(t, 1, (2, 3))),
                   ("grey picture", lambda v: v.mean(1, keepdim=True).repeat(1, 3, 1, 1), lambda t: t)):
    t = g(template(wm, f(x), bits))
    print("CHANNEL", method, "official, embedder fed", name, "| z fine", z(x, t), "| z mid", z(x, t, 5, 17), flush=True)
print("CHANNEL_DONE", flush=True)
