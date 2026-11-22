"""Local contact sheets of out-of-fold predictions (image | truth | predicted probability) to study failures."""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

run, train_root, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
with np.load(run / "oof_probs.npz") as z:
    probs = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
with np.load(train_root / "masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
rows = json.loads((train_root / "metadata.json").read_text())


def miou(pred, gt):
    s = 0.0
    for c in (0, 1):
        p, g = pred == c, gt == c
        u = (p | g).sum()
        s += 1.0 if u == 0 else (p & g).sum() / u
    return s / 2


def group(r):
    m = masks[r["image_name"]]
    if r["full_watermark"]:
        return "full"
    ys, xs = np.where(m)
    return "rect" if m.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "transplant"


table = {}
for r in rows:
    n = r["image_name"]
    table.setdefault((r["method"], group(r)), []).append((miou(probs[n] > 0.5, masks[n]), n))
T = 160
rng = np.random.default_rng(0)
summary = {}
for (method, g), items in sorted(table.items()):
    items.sort()
    summary[f"{method}/{g}"] = [len(items), round(float(np.mean([s for s, _ in items])), 4)]
    pick = [items[i] for i in np.linspace(0, len(items) - 1, 12).astype(int)]     # worst ... best, evenly spaced
    sheet = Image.new("RGB", (6 * 3 * T, 2 * (T + 14)), "white")
    d = ImageDraw.Draw(sheet)
    for k, (s, n) in enumerate(pick):
        x0, y0 = (k % 6) * 3 * T, (k // 6) * (T + 14)
        img = Image.open(train_root / "images" / (n + ".png")).convert("RGB").resize((T, T))
        gt = Image.fromarray((masks[n] * 255).astype(np.uint8)).resize((T, T), Image.Resampling.NEAREST).convert("RGB")
        pr = Image.fromarray((probs[n] * 255).astype(np.uint8)).resize((T, T), Image.Resampling.NEAREST).convert("RGB")
        for j, im in enumerate((img, gt, pr)):
            sheet.paste(im, (x0 + j * T, y0 + 14))
        d.text((x0 + 2, y0 + 1), f"{n} mIoU {s:.2f}", fill="black")
    out.mkdir(parents=True, exist_ok=True)
    sheet.save(out / f"oof_{method}_{g}.jpg", quality=85)
print(json.dumps(summary, indent=0))
