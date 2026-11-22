"""Describe train mask geometry (CPU only, read-only)."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = json.loads((root / "metadata.json").read_text())
rng = np.random.default_rng(0)

stats = []
for r in rows:
    m = np.array(Image.open(root / "masks" / (r["image_name"] + ".png")).convert("L")) > 127
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    bbox_area = (y1 - y0) * (x1 - x0)
    fill = m.sum() / bbox_area
    lab, ncomp = ndimage.label(m)
    sizes = ndimage.sum(m, lab, range(1, ncomp + 1))
    inv_lab, ninv = ndimage.label(~m)
    # holes = background components not touching the image border
    border = set(np.unique(np.concatenate([inv_lab[0], inv_lab[-1], inv_lab[:, 0], inv_lab[:, -1]])))
    holes = [i for i in range(1, ninv + 1) if i not in border]
    # straight-edge test: fraction of bbox border rows/cols that are fully set
    w, h = x1 - x0, y1 - y0
    stats.append(dict(name=r["image_name"], method=r["method"], full=bool(r["full_watermark"]),
                      area=float(m.mean()), bbox=(int(x0), int(y0), int(w), int(h)),
                      bbox_fill=float(fill), ncomp=int(ncomp), big=float(sizes.max() / m.sum()),
                      nholes=len(holes), touches=(int(x0 == 0) + int(y0 == 0) + int(x1 == 512) + int(y1 == 512)),
                      bbox_is_image=bool(w == 512 and h == 512), aspect=float(max(w / h, h / w))))


def kind(s):
    if s["bbox_fill"] > 0.999:
        return "pure_rectangle"
    if s["bbox_is_image"]:
        return "fullframe_with_holes"
    if s["bbox_fill"] > 0.6:
        return "rectangle_with_holes"
    return "irregular_object"


for s in stats:
    s["kind"] = kind(s)

print("TOTAL", len(stats))
for full in (True, False):
    sub = [s for s in stats if s["full"] == full]
    print(f"\nfull_watermark={full}: n={len(sub)}")
    kinds = {}
    for s in sub:
        kinds.setdefault(s["kind"], []).append(s)
    for k, v in sorted(kinds.items()):
        a = np.array([s["area"] for s in v])
        print(f"  {k:22s} n={len(v):3d} area median={np.median(a):.3f} min={a.min():.3f} max={a.max():.3f} "
              f"ncomp med={np.median([s['ncomp'] for s in v]):.0f} holes med={np.median([s['nholes'] for s in v]):.0f}")
print("\nby method x kind")
for meth in ("pixelseal", "trustmark", "stegastamp"):
    sub = [s for s in stats if s["method"] == meth]
    c = {}
    for s in sub:
        c[s["kind"]] = c.get(s["kind"], 0) + 1
    print(" ", meth, c, "max bbox aspect", round(max(s["aspect"] for s in sub), 2))
a = np.array([s["area"] for s in stats])
print("\narea quantiles", np.round(np.quantile(a, [0, .05, .1, .25, .5, .75, .9, .95, 1]), 4))
print("area<2%:", int((a < .02).sum()), " area<5%:", int((a < .05).sum()), " area>95%:", int((a > .95).sum()), " area>98%:", int((a > .98).sum()))
pr = [s for s in stats if s["kind"] == "pure_rectangle"]
print("pure rectangles: sides touching border histogram", np.bincount([s["touches"] for s in pr], minlength=5).tolist())
print("pure rectangle sizes (w,h) sample", [s["bbox"] for s in pr[:12]])
print("bbox x0,y0,w,h multiples of 4?", np.mean([all(v % 4 == 0 for v in s["bbox"]) for s in pr]),
      " multiples of 8?", np.mean([all(v % 8 == 0 for v in s["bbox"]) for s in pr]))
print("image sizes", {Image.open(root / 'images' / (stats[i]['name'] + '.png')).size for i in range(0, 510, 37)})

# contact sheets: image | mask overlay, 6 per kind/full combination
cells = []
for full in (True, False):
    for k in ("pure_rectangle", "fullframe_with_holes", "rectangle_with_holes", "irregular_object"):
        sub = [s for s in stats if s["full"] == full and s["kind"] == k]
        idx = rng.permutation(len(sub))[:6]
        cells.append((f"full={full} {k} (n={len(sub)})", [sub[i] for i in idx]))
T = 192
sheet = Image.new("RGB", (6 * 2 * T, len(cells) * (T + 18)), "white")
d = ImageDraw.Draw(sheet)
for row, (title, items) in enumerate(cells):
    y = row * (T + 18)
    d.text((4, y + 2), title, fill="black")
    for col, s in enumerate(items):
        img = Image.open(root / "images" / (s["name"] + ".png")).convert("RGB")
        m = np.array(Image.open(root / "masks" / (s["name"] + ".png")).convert("L")) > 127
        ov = np.array(img).astype(np.float32)
        ov[m] = ov[m] * 0.45 + np.array([0, 255, 0]) * 0.55
        ov[~m] = ov[~m] * 0.45 + np.array([255, 0, 0]) * 0.25
        sheet.paste(img.resize((T, T)), (col * 2 * T, y + 18))
        sheet.paste(Image.fromarray(ov.astype(np.uint8)).resize((T, T)), (col * 2 * T + T, y + 18))
        d.text((col * 2 * T + 3, y + 20), s["method"][:5], fill="yellow")
sheet.save(out / "train_kinds.jpg", quality=88)
json.dump(stats, open(out / "mask_stats.json", "w"))
print("saved", out / "train_kinds.jpg", sheet.size)
