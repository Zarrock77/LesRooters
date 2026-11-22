"""Estimate the OFFICIAL full-frame watermark pattern from the official pictures themselves (no mask is read).

Earlier probes showed that, in official full-frame pictures, the PixelSeal / TrustMark residual does not match what
the kit embedder writes for the released picture, although the message decodes. Here the pattern is not taken from
the embedder at all. If the residual is roughly a sum of one pattern per message bit (and depends little on the
picture), the per-bit patterns can be estimated by regressing the high-passed official pictures on their known
message bits: every full-frame picture of the method (train and test, about 255 pictures) is one equation system.
The template of a picture is predicted WITHOUT that picture (exact leave-one-out of the ridge regression), so a
picture's own content never enters its template. Local correlation between the high-passed picture and its
template then says where the watermark is present: a message-aware presence map, label-free.

Measured here (train masks for scoring only): frame-level correlation against a control (template of another
picture, i.e. a wrong message), and per-picture cell AUC of the local maps, kept region against hole.
usage: official_template.py <kit> <frame json> <method> <out dir> [ridge]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, frame_json, method, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
ridges = [float(sys.argv[5])] if len(sys.argv) > 5 else [1.0, 10.0, 50.0, 200.0]
out_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(8)
frame = json.loads(frame_json.read_text())
WINDOWS = (9, 17, 33)                     # correlation windows at 256 px


def gauss(x, sigma):
    r = int(3 * sigma + 0.5)
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma * sigma))
    k = (k / k.sum()).to(x.device)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def box(v, k):
    return F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


names, split_of, feats, signs = [], [], [], []
for split in ("train", "test"):
    rows, images = load_split(kit, split)
    for i, r in enumerate(rows):
        if r["method"] != method or frame[split][r["image_name"]]["acc"] < 0.8:
            continue
        x = F.avg_pool2d(images[i:i + 1].permute(0, 3, 1, 2).float(), 2)          # 256 px: the embedders work at about this size
        hp = x - gauss(x, 2.0)
        y = hp / (gauss(hp * hp, 4.0).mean(1, keepdim=True).sqrt() + 1.5)         # local contrast normalisation
        names.append(r["image_name"]); split_of.append(split); feats.append(y[0])
        signs.append([1.0 if c == "1" else -1.0 for c in r["message"]])
Y = torch.stack(feats)                                                             # [N,3,256,256]
S = torch.tensor(signs)
N, B = S.shape
print("OFFTPL", method, "full-frame pictures", N, "bits", B, flush=True)
Yf = Y.reshape(N, -1)
Yf = Yf - Yf.mean(0, keepdim=True)                                                 # what all pictures share is not message-dependent
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def auc(s, label):
    s, label = np.asarray(s).ravel(), np.asarray(label, bool).ravel()
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 < 8 or n0 < 8:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


best = None
for lam in ridges:
    H = S @ torch.linalg.solve(S.T @ S + lam * torch.eye(B), S.T)                  # hat matrix of the ridge regression
    h = torch.diagonal(H).clone()
    T = ((H - torch.diag(h)) @ Yf) / (1 - h)[:, None]                              # leave-one-out template of every picture
    cos = (Yf * T).sum(1) / (Yf.norm(dim=1) * T.norm(dim=1) + 1e-9)
    perm = torch.roll(torch.arange(N), 7)
    ctl = (Yf * T[perm]).sum(1) / (Yf.norm(dim=1) * T[perm].norm(dim=1) + 1e-9)
    z = float((cos.mean() - ctl.mean()) / (ctl.std() / N ** 0.5 + 1e-12))
    print("OFFTPL ridge", lam, "| leverage mean", round(float(h.mean()), 3), "| frame correlation own template", round(float(cos.mean()), 5),
          "wrong template", round(float(ctl.mean()), 5), "+-", round(float(ctl.std()), 5), "| z of the mean", round(z, 1),
          "| pictures above every control", round(float((cos > ctl.max()).float().mean()), 3), flush=True)
    if best is None or z > best[0]:
        best = (z, lam, T.clone())
z, lam, T = best
T = T.reshape(N, 3, 256, 256)
Yc = Yf.reshape(N, 3, 256, 256)
maps = []
for s in range(0, N, 16):
    a, b = Yc[s:s + 16], T[s:s + 16]
    out = []
    for w in WINDOWS:
        ab, aa, bb = (box((u * v).mean(1, keepdim=True), w) for u, v in ((a, b), (a, a), (b, b)))
        out.append(F.avg_pool2d(ab / (aa * bb + 1e-8).sqrt(), 2)[:, 0])
    maps.append(torch.stack(out, 1))
maps = torch.cat(maps).numpy().astype(np.float16)                                  # [N,3,128,128]
aucs = [[auc(maps[i, k].astype(np.float32), masks[n]) for k in range(len(WINDOWS))] for i, n in enumerate(names) if split_of[i] == "train"]
aucs = np.array(aucs)
print("OFFTPL best ridge", lam, "| train pictures scored", int(np.isfinite(aucs[:, 0]).sum()), "| mean per-picture cell AUC kept vs hole, windows 9 / 17 / 33 px at 256:",
      [round(float(np.nanmean(aucs[:, k])), 3) for k in range(len(WINDOWS))], flush=True)
for split in ("train", "test"):
    sel = [i for i in range(N) if split_of[i] == split]
    np.savez_compressed(out_dir / f"{split}_{method}.npz", names=np.array([names[i] for i in sel]), maps=maps[sel])
print("OFFTPL_DONE", method, flush=True)
