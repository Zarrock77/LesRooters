"""Per-image share of the kit template found in official full-frame images, relative to our own embedding on
the same picture. A two-peaked distribution would mean two editing pipelines (one that pastes untouched pixels
back, one that re-renders the whole picture). Diagnostic only.
usage: ratio_probe.py <kit> <frame json> <method>
"""
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
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def beta(x, t, keep):
    a, b = (box(x, 1) - box(x, 5)) * keep, (box(t, 1) - box(t, 5)) * keep
    return ((a * b).flatten(1).sum(1) / (b * b).flatten(1).sum(1)).cpu().numpy() * 255


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
out = []
for s in range(0, len(sel), 12):
    part = sel[s:s + 12]
    x = images[part].permute(0, 3, 1, 2).cuda().float() / 255
    bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in part]).cuda()
    keep = torch.from_numpy(np.stack([np.kron(masks[rows[i]["image_name"]], np.ones((4, 4), bool)) for i in part]))[:, None].cuda().float()
    again = (embed(wm, x, 1 - bits) * 255).round() / 255
    bo, bu = beta(x, template(wm, x, bits), keep), beta(again, template(wm, again, 1 - bits), keep)
    lg = wm.logits(x)
    margin = ((lg * (bits.float() * 2 - 1)).mean(1)).cpu().numpy()
    for j, i in enumerate(part):
        m = masks[rows[i]["image_name"]]
        out.append((rows[i]["image_name"], float(bo[j]), float(bu[j]), float(1 - m.mean()), float(margin[j])))
ratio = np.array([o[1] / o[2] if abs(o[2]) > 1e-6 else np.nan for o in out])
print("RATIO", method, "n", len(out), "official beta median", round(float(np.median([o[1] for o in out])), 3), "ours", round(float(np.median([o[2] for o in out])), 3), flush=True)
print("RATIO", method, "quantiles of official/ours (5,25,50,75,95):", np.round(np.nanquantile(ratio, [0.05, 0.25, 0.5, 0.75, 0.95]), 2).tolist(), flush=True)
hist, edges = np.histogram(np.clip(ratio, -0.5, 1.5), bins=np.arange(-0.5, 1.51, 0.125))
print("RATIO", method, "histogram from -0.5 to 1.5 step 0.125:", hist.tolist(), flush=True)
order = np.argsort(-ratio)
for k in list(order[:6]) + list(order[-4:]):
    print("RATIO", method, out[k][0], "ratio", round(float(ratio[k]), 2), "hole share", round(out[k][3], 3), "margin", round(out[k][4], 2), flush=True)
hole = np.array([o[3] for o in out]); marg = np.array([o[4] for o in out])
ok = np.isfinite(ratio)
print("RATIO", method, "corr(ratio, hole share)", round(float(np.corrcoef(ratio[ok], hole[ok])[0, 1]), 3),
      "corr(ratio, margin)", round(float(np.corrcoef(ratio[ok], marg[ok])[0, 1]), 3), flush=True)
(Path(__file__).resolve().parent.parent / "runs" / f"ratio_{method}.json").write_text(json.dumps(out))
print("RATIO_DONE", flush=True)
