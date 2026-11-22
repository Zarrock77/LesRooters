"""Held-out evaluation of checkpoints (no training): split-half stability and a score proxy checked against public scores.

- forget set split in two fixed halves A/B (seed 0, by source image); every metric is reported per half.
- variations: only the report subset (every 4th of sorted [vw]*_[0-3].npz = `vsel` in unlearn.py) is used for measuring;
  training never saw these images (but did see other variations of the same source images).
- gold proxy for image i = orig on its held-out unseen-like variation (strength 0.5-0.6), at T=2.
- proxy score = (1 - (|mean-2.49|+|std-3.91|)/6.40) * (1 - KL(g||sub)/KL(g||orig)) * 1/(1 + drift KL on held-out s>=0.6 variations)

Usage: python heldout.py name=path [name=path ...]
"""
import sys
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F

import task_template as tt

G = "/p/project1/training2666/delegue1/t3gen/"
orig = tt.load_model("model.pt")
imgs = tt.load_forget_images("forget_images.npz")
N = len(imgs)
lo = tt.predict(orig, imgs)
y = lo.argmax(1)
half = torch.randperm(N, generator=torch.Generator().manual_seed(0)) < N // 2  # True = half A
sh = [np.load(f) for f in sorted(glob(G + "[vw]*_[0-3].npz"))]
vimg = torch.from_numpy(np.concatenate([d["images"] for d in sh]))
vsrc, vstr = (torch.from_numpy(np.concatenate([d[k] for d in sh])) for k in ("src", "strength"))
vsel = torch.arange(0, len(vimg), 4)
vimg, vsrc, vstr = vimg[vsel], vsrc[vsel], vstr[vsel]
vo = tt.predict(orig, vimg)
g = (vstr > 0.45) & (vstr < 0.65)                                  # unseen-like: gold proxy
first = torch.full((N,), -1, dtype=torch.long)
first[vsrc[g].flip(0)] = g.nonzero().flatten().flip(0)            # one held-out unseen-like variation per image
hasg = first >= 0
far = vstr > 0.55                                                  # drift monitor


def kl(p, q):  # KL(softmax(p/2) || softmax(q/2)) per row
    return (F.softmax(p / 2, 1) * (F.log_softmax(p / 2, 1) - F.log_softmax(q / 2, 1))).sum(1)


print(f"held-out variations {len(vimg)}, images with a held-out unseen-like variation {int(hasg.sum())}, half A {int(half.sum())}")
for spec in sys.argv[1:]:
    name, path = spec.split("=", 1)
    m = tt.build_model()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m = m.to(tt.DEVICE).eval()
    ls, vs = tt.predict(m, imgs), tt.predict(m, vimg)
    L = F.cross_entropy(ls, y, reduction="none")
    out = []
    for hn, hm in (("A", half), ("B", ~half)):
        ms = 1 - (abs(L[hm].mean() - 2.49) + abs(L[hm].std() - 3.91)) / 6.40
        k = hm & hasg
        gp = vo[first[k]]
        pred = 1 - kl(gp, ls[k]).mean() / kl(gp, lo[k]).mean()
        dm = far & hm[vsrc]
        drift = kl(vo[dm], vs[dm]).mean()
        sc = max(ms, 0) * max(pred, 0) / (1 + drift)
        out.append(f"{hn}: loss {L[hm].mean():.2f}+-{L[hm].std():.2f} ms {ms:.3f} pred {pred:.3f} drift {drift:.3f} proxy {sc:.4f}")
    print(f"{name:14s} " + " | ".join(out), flush=True)
