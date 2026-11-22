"""T3 exploration: original-model behaviour on the forget set.

Usage (from repo root):  .venv/bin/python Unlearning/explore.py [--data data/unlearning]
Caches logits to <data>/orig_logits.pt (clean + a few augmentations) and writes a class-grid image.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/unlearning")
args = ap.parse_args()
DATA = Path(args.data)

sys.path.insert(0, str(DATA))
import task_template as tt  # noqa: E402  (official builder / preprocessing)

torch.set_num_threads(max(1, torch.get_num_threads()))

model = tt.load_model(DATA / "model.pt")
imgs = tt.load_forget_images(DATA / "forget_images.npz")
print("images", tuple(imgs.shape), imgs.dtype, "mean px", imgs.float().mean().item())


def augs(x):
    """A few 'altered copy' guesses: hflip, random-resized-crop-ish, blur-ish downscale, colour jitter."""
    yield "clean", x
    yield "hflip", x.flip(2)
    c = x[:, 16:-16, 16:-16, :].permute(0, 3, 1, 2).float()
    yield "crop", F.interpolate(c, size=224, mode="bilinear", align_corners=False).clamp(0, 255).round().byte().permute(0, 2, 3, 1)
    d = F.interpolate(x.permute(0, 3, 1, 2).float(), size=112, mode="bilinear", align_corners=False)
    return  # shortcut: down2x/bright skipped on CPU, re-enable on GPU
    yield "down2x", F.interpolate(d, size=224, mode="bilinear", align_corners=False).clamp(0, 255).round().byte().permute(0, 2, 3, 1)
    yield "bright", (x.float() * 1.2).clamp(0, 255).byte()


cache = DATA / "orig_logits.pt"
if cache.exists():
    logits = torch.load(cache)
else:
    logits = {}
    for name, xa in augs(imgs):
        out = []
        for i in range(0, len(xa), 64):
            out.append(tt.predict(model, xa[i:i + 64], batch_size=64))
            print(f"\r{name} {i + 64}/{len(xa)}", end="", flush=True)
        logits[name] = torch.cat(out)
        print()
    torch.save(logits, cache)

clean = logits["clean"]
pred = clean.argmax(1)
p1 = clean.softmax(1)
loss = F.cross_entropy(clean, pred, reduction="none")  # label = original argmax (true labels unknown)
ent = -(p1 * p1.clamp_min(1e-12).log()).sum(1)
p2 = (clean / 2).softmax(1)

print("\n=== clean forget images (label = orig argmax) ===")
print(f"max prob    mean {p1.max(1).values.mean():.4f}  median {p1.max(1).values.median():.4f}  min {p1.max(1).values.min():.4f}")
print(f"loss        mean {loss.mean():.4f}  std {loss.std():.4f}")
print(f"entropy     mean {ent.mean():.4f}")
print(f"T=2 maxprob mean {p2.max(1).values.mean():.4f}")
print(f"frac maxprob>0.99 {(p1.max(1).values > 0.99).float().mean():.3f}  <0.5 {(p1.max(1).values < 0.5).float().mean():.3f}")

counts = torch.bincount(pred, minlength=100)
print(f"\npredicted classes: {int((counts > 0).sum())}/100 used; per-class count min {counts.min()} max {counts.max()} "
      f"mean {counts.float().mean():.1f}")
print("top classes:", counts.topk(10).indices.tolist(), counts.topk(10).values.tolist())

print("\n=== augmentations: agreement with clean argmax / loss vs clean argmax ===")
for name, lg in logits.items():
    l = F.cross_entropy(lg, pred, reduction="none")
    print(f"{name:7s} agree {(lg.argmax(1) == pred).float().mean():.3f}   loss mean {l.mean():.3f} std {l.std():.3f}"
          f"   maxprob {lg.softmax(1).max(1).values.mean():.3f}")

# class grid: 4 examples for the 25 most frequent predicted classes -> look at it to identify the dataset
try:
    from PIL import Image

    top = counts.topk(25).indices.tolist()
    tiles = []
    for c in top:
        idx = (pred == c).nonzero().flatten()[:4]
        row = [imgs[i].numpy() for i in idx] + [np.zeros((224, 224, 3), np.uint8)] * (4 - len(idx))
        tiles.append(np.concatenate([np.asarray(Image.fromarray(r).resize((96, 96))) for r in row], 1))
    grid = np.concatenate([np.concatenate(tiles[i:i + 5], 1) for i in range(0, 25, 5)], 0)
    out = DATA / "class_grid.png"
    Image.fromarray(grid).save(out)
    print(f"\nclass grid (rows of 5 classes x 4 imgs, classes {top}) -> {out}")
except Exception as e:  # noqa: BLE001
    print("grid failed:", e)
