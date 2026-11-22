"""How sharp are the official mask edges, and how strong is the kit residual on the watermarked side?

For pixel pairs placed d pixels on either side of a true mask edge (same row or column), the difference
x(watermarked) - x(clean) is regressed on the kit residual at the watermarked pixel. With a hard composite of the
kit embedding the slope is the same for every d; with a feathered edge it grows with d; if the official residual
is unrelated to the kit's, it stays near zero. Self-made hard composites are the control. Full-frame pictures
only (holes are not object outlines, so the picture's own gradient averages out). Nothing is fitted.
usage: edge_profile.py <kit> <clean.npy> <frame json> <method>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_det import make_mask, sample_area  # noqa: E402

kit, clean_path, frame_json, method = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(19)
DIST = (1, 2, 3, 5, 8, 12)


def residual(img, bits):
    if method == "stegastamp":
        return wm.embed(img, bits).astype(np.float32) - img.astype(np.float32)
    from wm_template import embed
    x = torch.from_numpy(img).permute(2, 0, 1)[None].cuda().float() / 255
    return ((embed(wm, x, torch.tensor([bits]).cuda()) - x)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.float32)


def accumulate(acc, img, mask, bits):
    r, x = residual(img, bits), img.astype(np.float32)
    for axis in (0, 1):
        m = mask if axis == 1 else mask.T
        xx = x if axis == 1 else x.transpose(1, 0, 2)
        rr = r if axis == 1 else r.transpose(1, 0, 2)
        a, b = m[:, :-1], m[:, 1:]
        for sel, sgn in ((a & ~b, 1), (~a & b, -1)):        # sgn 1: watermark on the left of the edge
            ys, xs = np.where(sel)
            for d in DIST:
                xw = xs - (d - 1) if sgn == 1 else xs + d
                xc = xs + d if sgn == 1 else xs - (d - 1)
                ok = (xw >= 0) & (xw < 512) & (xc >= 0) & (xc < 512)
                ok[ok] &= m[ys[ok], xw[ok]] & ~m[ys[ok], xc[ok]]
                jump, ref = xx[ys[ok], xw[ok]] - xx[ys[ok], xc[ok]], rr[ys[ok], xw[ok]]
                acc[d][0] += float((jump * ref).sum()); acc[d][1] += float((ref * ref).sum()); acc[d][2] += int(ok.sum())


def report(tag, acc):
    print("PROFILE", method, f"{tag:28s}", "slope by distance to the edge:", {d: round(acc[d][0] / max(acc[d][1], 1e-9), 3) for d in DIST},
          "| pairs at d=1:", acc[1][2], "| kit residual RMS on those pixels", round(float(np.sqrt(acc[1][1] / max(acc[1][2] * 3, 1))), 2), flush=True)


meta = json.loads((kit / "release/train/metadata.json").read_text())
frame = json.loads(frame_json.read_text())["train"]
acc = {d: [0.0, 0.0, 0] for d in DIST}
for r in meta:
    if r["method"] != method or frame[r["image_name"]]["acc"] < 0.8:
        continue
    mask = np.array(Image.open(kit / "release/train/masks" / (r["image_name"] + ".png")).convert("L")) > 127
    img = np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB"))
    accumulate(acc, img, mask, [int(c) for c in r["message"]])
report("official full-frame holes", acc)
clean = np.load(clean_path, mmap_mode="r")
acc = {d: [0.0, 0.0, 0] for d in DIST}
for i in rng.choice(len(clean), 30, replace=False):
    S = np.asarray(clean[int(i)])
    bits = rng.integers(0, 2, wm.nbits).tolist()
    mask = ~make_mask(rng, sample_area(rng, 0.003, 0.5, 0.8))
    accumulate(acc, np.where(mask[..., None], wm.embed(S, bits), S), mask, bits)
report("self-made hard composite", acc)
print("PROFILE_DONE", flush=True)
