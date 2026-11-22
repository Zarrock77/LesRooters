"""Are the outpainting seams of pure rectangles detectable blindly? (local CPU, read-only)

For every pure-rectangle train image, score each axis-aligned segment of the true edge length by the mean isolated
step across it and report the rank of the 4 true edges among all same-length segments in the image.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

root, out = Path(sys.argv[1]), Path(sys.argv[2])
stats = [s for s in json.load(open(out / "mask_stats.json")) if s["kind"] == "pure_rectangle"]


def step_map(x):
    """Isolated vertical step between row y-1 and y (shape H+1 x W, borders zero)."""
    d = np.abs(np.diff(x, axis=0)).mean(2)            # d[y] = |x[y+1]-x[y]|, y in 0..H-2
    iso = np.zeros((x.shape[0] + 1, x.shape[1]), np.float32)
    core = d[1:-1] - 0.5 * (d[:-2] + d[2:])          # step at boundary between rows (y, y+1) for y=1..H-3
    iso[2:-2] = core
    return iso


def seg_rank(iso, row, a, b):
    """Rank (0 = best) of the mean over columns a..b at boundary `row` among all same-length segments."""
    c = np.concatenate([np.zeros((iso.shape[0], 1), np.float32), np.cumsum(iso, 1)], 1)
    length = b - a
    means = (c[:, length:] - c[:, :-length]) / length
    value = means[row, a]
    return int((means > value).sum()), float(value), float(np.quantile(means, .999)), means.size


rows = []
for s in stats:
    x = np.array(Image.open(root / "images" / (s["name"] + ".png")).convert("RGB")).astype(np.float32)
    x0, y0, w, h = s["bbox"]
    iso_y = step_map(x)                                # horizontal seams (constant row)
    iso_x = step_map(x.transpose(1, 0, 2))             # vertical seams (constant column)
    r = {"name": s["name"], "method": s["method"], "w": w, "h": h}
    res = {}
    if y0 > 1:
        res["top"] = seg_rank(iso_y, y0, x0, x0 + w)
    if y0 + h < 511:
        res["bottom"] = seg_rank(iso_y, y0 + h, x0, x0 + w)
    if x0 > 1:
        res["left"] = seg_rank(iso_x, x0, y0, y0 + h)
    if x0 + w < 511:
        res["right"] = seg_rank(iso_x, x0 + w, y0, y0 + h)
    r["edges"] = res
    rows.append(r)

for method in ("pixelseal", "trustmark", "stegastamp"):
    sub = [r for r in rows if r["method"] == method]
    ranks = np.array([e[0] for r in sub for e in r["edges"].values()])
    frac = np.array([e[0] / e[3] for r in sub for e in r["edges"].values()])
    print(method, "edges", len(ranks), "rank==0:", round(float((ranks == 0).mean()), 3), "rank<5:", round(float((ranks < 5).mean()), 3),
          "rank<50:", round(float((ranks < 50).mean()), 3), "median rank:", float(np.median(ranks)),
          "median percentile:", round(float(np.median(frac)), 5))
    for lo, hi in ((0, 40), (40, 100), (100, 600)):
        rk = np.array([e[0] for r in sub for k, e in r["edges"].items()
                       if lo <= (r["w"] if k in ("top", "bottom") else r["h"]) < hi])
        if len(rk):
            print("   edge length", lo, hi, "n", len(rk), "rank<5:", round(float((rk < 5).mean()), 3), "rank<50:", round(float((rk < 50).mean()), 3))
json.dump(rows, open(out / "seam_probe.json", "w"))
