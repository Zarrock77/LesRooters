"""Message-template presence maps for StegaStamp pictures (full-frame template).

For StegaStamp the official pictures follow the kit recipe closely (the kit residual appears as a jump across the
true mask edges), and its residual is strong and depends little on the picture. So the residual the kit embedder
writes for the KNOWN message on the released picture, T = embed(x, m) - x, is a good picture of what the
watermark looks like wherever a full-frame watermark is present (full-frame pictures with holes, transplants).
Local normalised correlation between the high-passed picture and the high-passed template, at three window sizes,
plus the local regression amplitude: four maps per picture on the 128 grid, in the format of the cross-fitted
fusion (fusion_cv.py --occ). No mask label is read; with --score the train maps are scored against the masks
(cell AUC per regime, information only).
usage: ss_template_maps.py <kit> <frame json> <out dir> [--score]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, frame_json, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
score = "--score" in sys.argv
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark("stegastamp", models_dir=kit / "models", device="cuda")
out_dir.mkdir(parents=True, exist_ok=True)
WINDOWS = (5, 9, 17)


def box(v, k):
    return F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def maps_for(img, bits):
    t = torch.from_numpy(wm.embed(img, bits).astype(np.float32) - img.astype(np.float32)).permute(2, 0, 1)[None].cuda()
    x = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)[None].cuda()
    out = []
    for w in WINDOWS:
        hp = 2 * w + 1                                         # high-pass a little wider than the correlation window
        a, b = x - box(x, hp), t - box(t, hp)
        ab, aa, bb = (box((u * v).mean(1, keepdim=True), w) for u, v in ((a, b), (a, a), (b, b)))
        out.append(F.avg_pool2d(ab / (aa * bb + 1e-6).sqrt(), 4)[0, 0])
        if w == 9:
            slope = F.avg_pool2d((ab / (bb + 1e-3)).clamp(-2, 3), 4)[0, 0]
    return torch.stack(out + [slope]).cpu().numpy().astype(np.float16)


def auc(s, label):
    s, label = np.asarray(s).ravel(), np.asarray(label, bool).ravel()
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


frame = json.loads(frame_json.read_text())
for split in ("train", "test"):
    rows, images = load_split(kit, split)
    names, maps = [], []
    for i, r in enumerate(rows):
        if r["method"] != "stegastamp":
            continue
        names.append(r["image_name"]); maps.append(maps_for(images[i].numpy(), [int(c) for c in r["message"]]))
        if len(names) % 60 == 0:
            print("SSMAPS", split, len(names), flush=True)
    np.savez_compressed(out_dir / f"{split}_stegastamp.npz", names=np.array(names), maps=np.stack(maps))
    print("SSMAPS_DONE", split, len(names), flush=True)
    if split == "train" and score:
        with np.load(kit / "release/train/masks_128.npz") as z:
            masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
        groups = {}
        for n, m in zip(names, maps):
            gt = masks[n]
            ys, xs = np.where(gt)
            g = "full" if frame["train"][n]["acc"] >= 0.8 else ("rect" if gt.sum() == (ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()) else "transplant")
            groups.setdefault(g, []).append([auc(m[k].astype(np.float32), gt) for k in range(4)])
        for g, v in groups.items():
            v = np.array(v)
            print("SSMAPS_SCORE", g, "pictures", len(v), "| mean per-picture cell AUC: windows 5 / 9 / 17 px:",
                  [round(float(np.nanmean(v[:, k])), 3) for k in range(3)], "| amplitude map:", round(float(np.nanmean(v[:, 3])), 3), flush=True)
