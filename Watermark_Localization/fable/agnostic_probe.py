"""Does the official full-frame watermark still carry the picture-INDEPENDENT pattern of its message?

The template computed on the released picture matches official full-frame watermarks only weakly. But part of a
learned watermark depends on the message alone. That part is estimated here as the average, over K unrelated
clean pictures, of embed(x, m) - embed(x, 1 - m); it is then correlated with (a) pictures we embed ourselves and
(b) official full-frame pictures, globally and locally inside / outside the true holes (labels for measuring only).
If (b) is as strong as (a), the message-only pattern is a usable local evidence for official pictures.
usage: agnostic_probe.py <kit> <clean.npy> <frame json> <method> [K]
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

kit, clean_path, frame_json, method = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4]
K = int(sys.argv[5]) if len(sys.argv) > 5 else 12
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(31)
clean = np.load(clean_path, mmap_mode="r")
covers = torch.from_numpy(np.stack([clean[int(i)] for i in rng.choice(len(clean), K, replace=False)])).permute(0, 3, 1, 2).cuda().float() / 255


def agnostic(bits):
    """[B,nbits] -> [B,3,512,512] message-only pattern: mean template over K unrelated covers."""
    out = []
    for b in bits:
        out.append(template(wm, covers, b[None].repeat(K, 1)).mean(0))
    return torch.stack(out)


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


def zscore(x, t, lo, hi):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    return ((c[:, 0, 0] - flat.mean(1)) / flat.std(1)).cpu().numpy()


def local_ncc(x, t, lo=1, hi=5, win=33):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    ab, aa, bb = (box((u * v).mean(1, keepdim=True), win) for u, v in ((a, b), (a, a), (b, b)))
    return (ab / (aa * bb + 1e-14).sqrt())[:, 0]


frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
sel = [i for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
stats = {"official own-picture template": [], "official message-only": [], "official message-only, wrong message": [],
         "fresh own-picture template": [], "fresh message-only": []}
inside, outside = {"own": [], "agnostic": []}, {"own": [], "agnostic": []}
for s in range(0, len(sel), 8):
    part = sel[s:s + 8]
    x = images[part].permute(0, 3, 1, 2).cuda().float() / 255
    bits = torch.tensor([[int(c) for c in rows[i]["message"]] for i in part]).cuda()
    own, agn = template(wm, x, bits), agnostic(bits)
    for lo, hi in ((1, 5), (5, 17)):
        stats["official own-picture template"].append((lo, zscore(x, own, lo, hi)))
        stats["official message-only"].append((lo, zscore(x, agn, lo, hi)))
        stats["official message-only, wrong message"].append((lo, zscore(x, agnostic(torch.randint_like(bits, 2)), lo, hi)))
    for name, t in (("own", own), ("agnostic", agn)):
        m = local_ncc(x, t).cpu().numpy()
        for j, i in enumerate(part):
            keep = np.kron(masks[rows[i]["image_name"]], np.ones((4, 4), bool))
            if (~keep).sum() > 500:
                inside[name].append(float(m[j][~keep].mean())); outside[name].append(float(m[j][keep].mean()))
    if s < 32:                                             # control: fresh embedding of other clean pictures with the same messages
        idx = rng.choice(len(clean), len(part), replace=False)
        xc = torch.from_numpy(np.stack([clean[int(i)] for i in idx])).permute(0, 3, 1, 2).cuda().float() / 255
        xw = (embed(wm, xc, bits) * 255).round() / 255
        for lo, hi in ((1, 5), (5, 17)):
            stats["fresh own-picture template"].append((lo, zscore(xw, template(wm, xw, bits), lo, hi)))
            stats["fresh message-only"].append((lo, zscore(xw, agn, lo, hi)))
for name, v in stats.items():
    print("AGNOSTIC", method, f"{name:40s}", {f"z band {lo}": round(float(np.median(np.concatenate([a for l, a in v if l == lo]))), 2) for lo in (1, 5)}, flush=True)
for name in ("own", "agnostic"):
    a, b = np.array(outside[name]), np.array(inside[name])
    print("AGNOSTIC", method, "local correlation with the", name, "template: kept region", round(float(np.median(a)), 4), "| hole", round(float(np.median(b)), 4),
          "| pictures with kept > hole:", round(float((a > b).mean()), 3), "of", len(a), flush=True)
print("AGNOSTIC_DONE", flush=True)
