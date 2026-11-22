"""Server probe without training: orig + negative head bias on some classes, so only their images change.

The score is forget * retain, and forget = 0 as soon as the edit moves the forget images AWAY from gold on average.
So "kill class c" scores > 0 only if gold does not predict c on the forget images of class c (c is gone from gold's
training set); the bias size then scans gold's loss std (few images with a huge loss move std much more than mean).

Usage: python probe.py 29 14 output/probe.pt     (classes comma-separated, target mean CE on their forget images)
"""
import sys

import torch
import torch.nn.functional as F

cls, ell, out = [int(c) for c in sys.argv[1].split(",")], float(sys.argv[2]), sys.argv[3]
lg = torch.load("orig_logits.pt")["clean"]
y = lg.argmax(1)
m = torch.isin(y, torch.tensor(cls))


def shifted(b):
    z = lg.clone()
    z[:, cls] -= b
    return z


lo, hi = 0.0, 100.0
for _ in range(40):  # bisection: mean CE of the selected classes' forget images = ell
    b = (lo + hi) / 2
    lo, hi = (b, hi) if F.cross_entropy(shifted(b)[m], y[m]) < ell else (lo, b)
L = F.cross_entropy(shifted(b), y, reduction="none")
print(f"classes {cls}: {int(m.sum())} images, bias -{b:.2f} -> their loss {L[m].mean():.2f}+-{L[m].std():.2f};"
      f" forget set loss mean {L.mean():.3f} std {L.std():.3f} sum {L.mean() + L.std():.2f}")
sd = torch.load("model.pt", map_location="cpu", weights_only=True)
sd["head.bias"][cls] -= b
torch.save(sd, out)
print("saved", out)
