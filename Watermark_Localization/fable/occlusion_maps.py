"""Decoder-occlusion evidence maps for every official train and test image (nothing is trained).

For each image, in the FULL frame and with the known message: destroy one window at a time (mean colour and
heavy blur) and record how much the kit decoder's agreement with the message drops. Windows that carry the
watermark cost agreement; edited or never-watermarked windows cost nothing.
Windows: 64 px (stride 32) and 128 px (stride 64). Output per split: npz with names, maps [N,4,128,128] float16
(64px-mean, 64px-blur, 128px-mean, 128px-blur; each normalised by the image's full-frame margin) and base margins.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--methods", default="pixelseal,trustmark,stegastamp")
ap.add_argument("--splits", default="train,test")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
sys.path.insert(0, str(args.kit))
import onnxruntime as ort  # noqa: E402
ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402

dev = args.device
WINDOWS = [(64, 32), (128, 64)]


def gauss(t, sigma):
    k = int(sigma * 3) * 2 + 1
    ax = torch.arange(k, dtype=torch.float32, device=t.device) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    t = F.pad(t, (k // 2,) * 4, mode="reflect")
    t = F.conv2d(t, g.view(1, 1, 1, k).repeat(3, 1, 1, 1), groups=3)
    return F.conv2d(t, g.view(1, 1, k, 1).repeat(3, 1, 1, 1), groups=3)


def margins(wm, method, batch, sign):
    out = []
    for i in range(0, len(batch), 128):
        b = batch[i:i + 128]
        if method == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16):
                lg = wm.net(wm._prep(b)).float()
        elif method == "pixelseal":
            lg = wm.det(wm._resize(b))[:, 1:].float()
        else:
            lg = wm.logits(b).to(dev)
        out.append((lg * sign).mean(1))
    return torch.cat(out)


args.out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
for method in args.methods.split(","):
    wm = load_watermark(method, models_dir=args.kit / "models", device=dev)
    for split in args.splits.split(","):
        rows = [r for r in json.loads((args.kit / "release" / split / "metadata.json").read_text()) if r["method"] == method]
        names, maps, base = [], [], []
        for n, r in enumerate(rows):
            x = torch.from_numpy(np.array(Image.open(args.kit / "release" / split / "images" / (r["image_name"] + ".png")).convert("RGB")))
            x = x.permute(2, 0, 1)[None].float().div(255).to(dev)
            sign = torch.tensor([int(b) for b in r["message"]], dtype=torch.float32, device=dev) * 2 - 1
            destroyed = [x.mean((2, 3), keepdim=True).expand_as(x), gauss(x, 12.0)]
            with torch.no_grad():
                m_full = float(margins(wm, method, x, sign)[0])
                channels = []
                for size, stride in WINDOWS:
                    pos = [(y, xx) for y in range(0, 512 - size + 1, stride) for xx in range(0, 512 - size + 1, stride)]
                    for d in destroyed:
                        batch = x.repeat(len(pos), 1, 1, 1)
                        for k, (y, xx) in enumerate(pos):
                            batch[k, :, y:y + size, xx:xx + size] = d[0, :, y:y + size, xx:xx + size]
                        drop = m_full - margins(wm, method, batch, sign)
                        acc = torch.zeros(128, 128, device=dev); cnt = torch.zeros(128, 128, device=dev)
                        for k, (y, xx) in enumerate(pos):
                            acc[y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += drop[k]
                            cnt[y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += 1
                        channels.append(acc / cnt / max(abs(m_full), 0.5))
            names.append(r["image_name"]); maps.append(torch.stack(channels).cpu().numpy().astype(np.float16)); base.append(m_full)
            if n % 50 == 0:
                print("OCC", method, split, n, "/", len(rows), round(time.time() - t0, 1), "s", flush=True)
        np.savez_compressed(args.out / f"{split}_{method}.npz", names=np.array(names), maps=np.stack(maps), base=np.array(base, np.float32))
    del wm
    torch.cuda.empty_cache()
(args.out / f"DONE_{args.methods.replace(',', '_')}").write_text("ok")
print("OCC_DONE", args.methods, round(time.time() - t0, 1), "s", flush=True)
