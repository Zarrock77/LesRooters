"""Is the official watermark simply weaker than a fresh kit embedding?

Earlier measurements: on official full-frame images the watermark's amplitude along the message template is
about half (PixelSeal, StegaStamp) to a quarter (TrustMark) of a fresh embedding, with slightly lower decoder
margins. Here a fresh embedding is attenuated, x + alpha * (embed(x, m) - x), with a hole of about the official
size, and the decoder's bit accuracy and signed margin are compared with the official full-frame images.
The alpha whose margin matches tells how strong the synthetic training watermark should be.
usage: alpha_probe.py <kit> <clean.npy> <frame json> <method> [n]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from train_det import make_mask  # noqa: E402

kit, clean_path, frame_json, method = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
n = int(sys.argv[5]) if len(sys.argv) > 5 else 48
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(6)


def stats(images_uint8, bits):
    x = torch.from_numpy(images_uint8).permute(0, 3, 1, 2).float() / 255
    lg = torch.cat([torch.as_tensor(wm.logits(x[s:s + 16])).float().cpu() for s in range(0, len(x), 16)])
    sign = torch.from_numpy(bits).float() * 2 - 1
    acc = ((lg > 0) == (sign > 0)).float().mean(1)
    margin = (lg * sign).mean(1)
    return float(acc.mean()), float(acc.median()), float(margin.mean()), float(margin.median())


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
hole_share = np.array([1 - masks[rows[i]["image_name"]].mean() for i in sel])
real_bits = np.array([[int(c) for c in rows[i]["message"]] for i in sel])
a = stats(images[sel].numpy(), real_bits)
print("ALPHA", method, "official full-frame n", len(sel), "hole share median", round(float(np.median(hole_share)), 3),
      "| bit acc mean/median", round(a[0], 4), round(a[1], 4), "| margin mean/median", round(a[2], 3), round(a[3], 3), flush=True)
# split the official images by hole size: the margin of images with tiny holes is the cleanest reference
small = hole_share <= np.quantile(hole_share, 0.33)
b = stats(images[sel].numpy()[small], real_bits[small])
print("ALPHA", method, "official, smallest third of holes (median", round(float(np.median(hole_share[small])), 3), ")",
      "| bit acc", round(b[0], 4), "| margin mean/median", round(b[2], 3), round(b[3], 3), flush=True)

clean = np.load(clean_path, mmap_mode="r")
idx = rng.choice(len(clean), n, replace=False)
S = np.stack([clean[int(i)] for i in idx])
bits = rng.integers(0, 2, (n, wm.nbits))
W = np.stack([wm.embed(S[j], bits[j].tolist()) for j in range(n)]).astype(np.float32)
holes = [make_mask(rng, float(np.clip(rng.choice(hole_share), 0.003, 0.6))) for _ in range(n)]
other = np.roll(S, 1, axis=0)
for alpha in (1.0, 0.8, 0.65, 0.5, 0.4, 0.3, 0.2, 0.1):
    out = np.clip(np.rint(S + alpha * (W - S)), 0, 255).astype(np.uint8)
    full = stats(out, bits)
    holed = out.copy()
    for j in range(n):
        holed[j][holes[j]] = other[j][holes[j]]
    h = stats(holed, bits)
    print("ALPHA", method, "fresh, strength", alpha, "| no hole: bit acc", round(full[0], 4), "margin", round(full[2], 3),
          "| with official-size holes: bit acc mean/median", round(h[0], 4), round(h[1], 4), "margin mean/median", round(h[2], 3), round(h[3], 3), flush=True)
print("ALPHA_DONE", flush=True)
