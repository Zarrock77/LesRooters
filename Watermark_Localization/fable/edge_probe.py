"""Does the kit residual appear as a JUMP across the true mask edge of official pictures?

The starter kit says full-frame pictures are "the whole image watermarked and a part of it restored to the clean
original". If that is a hard composite, a pixel just inside the watermark and its neighbour just outside differ
by the watermark residual at that place (plus the picture's own gradient, which averages out). The jump across
the edge is compared with the kit residual embed(x, m) - x at the watermarked pixel:
  slope near 1 and clear correlation -> the kept pixels are the kit embedding, hard edge;
  jump present but unrelated to the kit residual -> another residual;
  no jump -> blended or re-rendered edge.
Controls: the same statistic on self-made composites, and across a shifted copy of the edge (no jump expected).
Labels are used to locate edges only; nothing is fitted.
usage: edge_probe.py <kit> <clean.npy> <frame json> <method> [n]
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
n = int(sys.argv[5]) if len(sys.argv) > 5 else 60
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(17)


def residual(img, bits):
    """Kit residual for this picture and message, in grey levels (float where the embedder allows it)."""
    if method == "stegastamp":
        return wm.embed(img, bits).astype(np.float32) - img.astype(np.float32)
    from wm_template import embed
    x = torch.from_numpy(img).permute(2, 0, 1)[None].cuda().float() / 255
    b = torch.tensor([bits]).cuda()
    return ((embed(wm, x, b) - x)[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.float32)


def pairs(mask):
    """Horizontal and vertical neighbour pairs straddling the mask edge: (watermarked pixel, clean pixel)."""
    out = []
    a, b = mask[:, :-1], mask[:, 1:]
    ys, xs = np.where(a & ~b); out.append((ys, xs, ys, xs + 1))
    ys, xs = np.where(~a & b); out.append((ys, xs + 1, ys, xs))
    a, b = mask[:-1], mask[1:]
    ys, xs = np.where(a & ~b); out.append((ys, xs, ys + 1, xs))
    ys, xs = np.where(~a & b); out.append((ys + 1, xs, ys, xs))
    return [np.concatenate([o[k] for o in out]) for k in range(4)]


def stats(img, mask, bits):
    r = residual(img, bits)
    x = img.astype(np.float32)
    yw, xw, yc, xc = pairs(mask)
    jump = x[yw, xw] - x[yc, xc]                          # [P,3]
    ref = r[yw, xw]
    # control: the same pairs moved 12 px into the watermarked side along a random direction (both pixels watermarked)
    out = {"n": len(yw), "jr": float((jump * ref).sum()), "rr": float((ref * ref).sum()), "jj": float((jump * jump).sum())}
    dy, dx = [(0, 12), (12, 0), (0, -12), (-12, 0)][int(rng.integers(4))]
    y1, x1, y2, x2 = yw + dy, xw + dx, yc + dy, xc + dx
    ok = (y1 >= 0) & (y1 < 512) & (x1 >= 0) & (x1 < 512) & (y2 >= 0) & (y2 < 512) & (x2 >= 0) & (x2 < 512)
    ok[ok] &= mask[y1[ok], x1[ok]] & mask[y2[ok], x2[ok]]
    j2 = x[y1[ok], x1[ok]] - x[y2[ok], x2[ok]]
    out.update({"c_jr": float((j2 * r[y1[ok], x1[ok]]).sum()), "c_rr": float((r[y1[ok], x1[ok]] ** 2).sum())})
    return out


def summary(tag, rows_):
    jr, rr, jj = (sum(r[k] for r in rows_) for k in ("jr", "rr", "jj"))
    c_jr, c_rr = sum(r["c_jr"] for r in rows_), sum(r["c_rr"] for r in rows_)
    per = np.array([r["jr"] / max(r["rr"], 1e-9) for r in rows_ if r["n"] > 50])
    print("EDGE", method, f"{tag:30s}", "pictures", len(rows_), "edge pairs", sum(r["n"] for r in rows_),
          "| slope of jump on kit residual", round(jr / rr, 3), "| correlation", round(jr / np.sqrt(rr * jj), 3),
          "| per-picture slope median", round(float(np.median(per)), 3), "share > 0.3:", round(float((per > 0.3).mean()), 3),
          "| control slope (no edge)", round(c_jr / max(c_rr, 1e-9), 3), flush=True)


meta = json.loads((kit / "release/train/metadata.json").read_text())
frame = json.loads(frame_json.read_text())["train"]
groups = {"official full-frame holes": [], "official rectangles": [], "official transplants": []}
for r in meta:
    if r["method"] != method:
        continue
    name = r["image_name"]
    mask = np.array(Image.open(kit / "release/train/masks" / (name + ".png")).convert("L")) > 127
    ys, xs = np.where(mask)
    if frame[name]["acc"] >= 0.8:
        g = "official full-frame holes"
    elif mask.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()):
        g = "official rectangles"       # NB: rectangles are embedded in their own frame, the full-frame residual is not expected to match
    else:
        g = "official transplants"
    if len(groups[g]) >= n:
        continue
    img = np.array(Image.open(kit / "release/train/images" / (name + ".png")).convert("RGB"))
    groups[g].append(stats(img, mask, [int(c) for c in r["message"]]))
clean = np.load(clean_path, mmap_mode="r")
own = []
for i in rng.choice(len(clean), 24, replace=False):
    S = np.asarray(clean[int(i)])
    bits = rng.integers(0, 2, wm.nbits).tolist()
    W = wm.embed(S, bits)
    mask = ~make_mask(rng, sample_area(rng, 0.003, 0.5, 0.8))
    comp = np.where(mask[..., None], W, S)
    own.append(stats(comp, mask, bits))
summary("self-made hard composite", own)
for g, v in groups.items():
    if v:
        summary(g, v)
print("EDGE_DONE", flush=True)
