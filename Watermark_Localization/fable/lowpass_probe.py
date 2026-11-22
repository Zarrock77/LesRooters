"""Is the official full-frame watermark the kit watermark with only its COARSE part left?

Per band, official TrustMark full-frame pictures keep about 0.08 of the template amplitude below 17 px but about
0.6 above, while the decoder margin is intact. A generative editor that re-renders the kept pixels re-synthesises
fine texture and preserves coarse structure. Emulation: keep the cover, low-pass only the watermark residual,
x = S + G_sigma * (W - S). For each sigma: decoder margin and template amplitude per band, relative to a fresh
embedding, next to the official numbers measured the same way.
usage: lowpass_probe.py <kit> <dir with fresh_<method>.npz> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, folder, frame_json, method = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
if method != "stegastamp":
    from wm_template import template as fast_template  # noqa: E402
BANDS = ((1, 5), (5, 17), (17, 65))


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


def gauss(v, sigma):
    if sigma == 0:
        return v
    k = int(4 * sigma) * 2 + 1
    g = torch.exp(-(torch.arange(k, device=v.device) - k // 2) ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    v = F.conv2d(F.pad(v, (k // 2, k // 2, 0, 0), mode="reflect"), g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(F.pad(v, (0, 0, k // 2, k // 2), mode="reflect"), g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def signature(tag, x, bits, ref=None):
    lg = torch.cat([torch.as_tensor(wm.logits(x[s:s + 8])).float().cpu() for s in range(0, len(x), 8)])
    sign = bits.float().cpu() * 2 - 1
    acc, margin = float(((lg > 0) == (sign > 0)).float().mean()), float((lg * sign).mean())
    t = tmpl(x, bits)
    res = {"margin": margin}
    for lo, hi in BANDS:
        a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
        res[f"{lo}-{hi}"] = float(((a * b).flatten(1).sum(1) / (b * b).flatten(1).sum(1)).median())
    line = {"bit acc": round(acc, 4), "margin": round(margin, 2)}
    if ref:
        line.update({"margin vs fresh": round(margin / ref["margin"], 2), **{f"template {lo}-{hi}": round(res[f"{lo}-{hi}"] / ref[f"{lo}-{hi}"], 2) for lo, hi in BANDS}})
    print("LOWPASS", method, f"{tag:26s}", line, flush=True)
    return res


with np.load(folder / f"fresh_{method}.npz") as z:
    S, W, bits = z["S"][:24].astype(np.float32), z["W"][:24].astype(np.float32), z["bits"][:24].astype(np.int64)
s = torch.from_numpy(S).permute(0, 3, 1, 2).cuda() / 255
w = torch.from_numpy(W).permute(0, 3, 1, 2).cuda() / 255
b = torch.from_numpy(bits).cuda()
ref = signature("fresh", w, b)
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
sel = sorted(sel, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:24]
signature("OFFICIAL (smallest holes)", images[sel].permute(0, 3, 1, 2).cuda().float() / 255,
          torch.tensor([[int(c) for c in rows[i]["message"]] for i in sel]).cuda(), ref)
for sigma in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
    x = ((s + gauss(w - s, sigma)).clamp(0, 1) * 255).round() / 255
    signature(f"residual low-pass {sigma}", x, b, ref)
# coarse part kept at full strength plus the fine part attenuated
for sigma, keep in ((3.0, 0.1), (4.0, 0.1), (6.0, 0.1)):
    r = w - s
    low = gauss(r, sigma)
    x = ((s + low + keep * (r - low)).clamp(0, 1) * 255).round() / 255
    signature(f"low-pass {sigma} + {keep} fine", x, b, ref)
print("LOWPASS_DONE", flush=True)
