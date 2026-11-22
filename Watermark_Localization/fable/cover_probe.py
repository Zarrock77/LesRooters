"""Was the official watermark computed on one picture and carried by another rendering of it?

Official full-frame images decode at full strength (alpha_probe.py) but their watermark matches the message
template four to six times less than a fresh embedding (band_probe.py), and no degradation of a fresh embedding
reproduces that. Hypothesis: the residual r = embed(S, m) - S was computed on the original picture S, while the
released pixels are another rendering S' of the same picture (e.g. the output of the editing model) plus r.
Test: x = S' + r for several S' and compare margin and template correlation with the official values.
usage: cover_probe.py <kit> <clean.npy> <re-rendered clean.npy, same pictures> <frame json> <method> [n]
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
from wm_template import template  # noqa: E402

kit, clean_path, alt_path, frame_json, method = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4]), sys.argv[5]
n = int(sys.argv[6]) if len(sys.argv) > 6 else 32
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(8)
BANDS = [(1, 5), (5, 17), (17, 65)]


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def z0(x, t, lo, hi):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    return float(((c[:, 0, 0] - flat.mean(1)) / flat.std(1)).median())


def describe(tag, images_uint8, bits):
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).cuda().float() / 255
    b = torch.from_numpy(bits).cuda()
    lg = wm.logits(x)
    sign = b.float() * 2 - 1
    margin = float((lg * sign).mean())
    acc = float(((lg > 0) == (sign > 0)).float().mean())
    zs = {f"{lo}-{hi}": round(z0(x, template(wm, x, b), lo, hi), 2) for lo, hi in BANDS} if method != "stegastamp" else {}
    print("COVER", method, f"{tag:34s}", "bit acc", round(acc, 4), "margin", round(margin, 3), "| template z", zs, flush=True)


def jpeg(a, q):
    out = []
    for im in a:
        buf = io.BytesIO(); Image.fromarray(im).save(buf, "JPEG", quality=q)
        out.append(np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB")))
    return np.stack(out)


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
sel = sorted(sel, key=lambda i: 1 - masks[rows[i]["image_name"]].mean())[:n]          # smallest holes: cleanest reference
describe("official (smallest holes)", images[sel].numpy(), np.array([[int(c) for c in rows[i]["message"]] for i in sel]))

clean, alt = np.load(clean_path, mmap_mode="r"), np.load(alt_path, mmap_mode="r")
idx = rng.choice(len(clean), n, replace=False)
S = np.stack([clean[int(i)] for i in idx]).astype(np.float32)
A = np.stack([alt[int(i)] for i in idx]).astype(np.float32)
bits = rng.integers(0, 2, (n, wm.nbits))
W = np.stack([wm.embed(S[j].astype(np.uint8), bits[j].tolist()) for j in range(n)]).astype(np.float32)
r = W - S
u8 = lambda v: np.clip(np.rint(v), 0, 255).astype(np.uint8)  # noqa: E731
describe("fresh: S + r", u8(W), bits)
describe("re-rendered picture + r", u8(A + r), bits)
describe("jpeg75(S) + r", u8(jpeg(u8(S), 75).astype(np.float32) + r), bits)
describe("S + noise 4 + r", u8(S + rng.normal(0, 4, S.shape) + r), bits)
describe("other picture + r", u8(np.roll(S, 1, axis=0) + r), bits)
# the reverse order: watermark first, then the whole picture re-rendered is covered by other probes (degrade_probe.py)
Wa = np.stack([wm.embed(u8(A[j]), bits[j].tolist()) for j in range(n)]).astype(np.float32)
describe("fresh on the re-rendered picture", u8(Wa), bits)
describe("S + residual of re-rendered", u8(S + (Wa - A)), bits)
print("COVER_DONE", flush=True)
