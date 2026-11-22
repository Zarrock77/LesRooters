"""Cross-fitted fusion of the detector's probability map with decoder-occlusion evidence.

A tiny convolutional head (about 12k weights) corrects the detector's out-of-fold probability map using the
occlusion maps, the full-frame decode strength and the method. For outer fold k it is fitted ONLY on images of the
other four folds (with their own out-of-fold detector outputs) and then applied to fold k, so the fused OOF stays
out-of-sample. The head for the test set is fitted on all 510 OOF images.
Writes a folds-like directory (fold_k/outer_probs.npz, fold_k/test_probs.npz) usable by rect_refine/assemble.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

METHODS = ("pixelseal", "trustmark", "stegastamp")


def load_maps(folder, split):
    """Evidence maps per image for the methods that have them (missing methods get zeros later)."""
    out = {}
    if folder is None:
        return out
    for m in METHODS:
        f = folder / f"{split}_{m}.npz"
        if f.exists():
            with np.load(f) as z:
                for n, a in zip(z["names"].tolist(), z["maps"]):
                    out[n] = a.astype(np.float32)
    return out


CHANNELS = 16


def features(names, probs, occ, ig, frame, method_of):
    x = np.zeros((len(names), CHANNELS, 128, 128), np.float32)
    for i, n in enumerate(names):
        p = np.clip(probs[i].astype(np.float32), 1e-4, 1 - 1e-4)
        x[i, 0] = np.log(p / (1 - p)) / 4
        if n in occ:
            x[i, 1:5] = np.clip(occ[n], -2, 2)
        if n in ig:
            x[i, 5:7] = np.clip(ig[n], -4, 4) / 2
        x[i, 7] = float(frame[n]["acc"] >= 0.8)
        x[i, 8] = np.clip(frame[n]["margin"], -1, 12) / 8
        x[i, 9 + METHODS.index(method_of[n])] = 1
    x = torch.from_numpy(x)
    # the gradient evidence is noisy cell by cell: also give it averaged over 9- and 21-cell neighbourhoods
    x[:, 12:14] = F.avg_pool2d(x[:, 5:7], 9, 1, 4, count_include_pad=False)
    x[:, 14:16] = F.avg_pool2d(x[:, 5:7], 21, 1, 10, count_include_pad=False)
    return x


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(CHANNELS, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.GELU(),
                                 nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.GELU(), nn.Conv2d(32, 1, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x[:, 0] * 4 + self.net(x)[:, 0]          # starts as the identity on the detector's logit


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    i1 = ((p * target).sum((1, 2)) + 1) / ((p + target - p * target).sum((1, 2)) + 1)
    q, t0 = 1 - p, 1 - target
    i0 = ((q * t0).sum((1, 2)) + 1) / ((q + t0 - q * t0).sum((1, 2)) + 1)
    return bce + (1 - (i0 + i1) / 2).mean()


def dihedral(x, y, k):
    if k >= 4:
        x, y = x.flip(-1), y.flip(-1)
    return torch.rot90(x, k % 4, (-2, -1)), torch.rot90(y, k % 4, (-2, -1))


def fit(x, y, steps, seed):
    torch.manual_seed(seed)
    head = Head().cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    x, y = x.cuda(), y.cuda()
    g = torch.Generator(device="cuda").manual_seed(seed)
    for _ in range(steps):
        idx = torch.randint(len(x), (48,), device="cuda", generator=g)
        xb, yb = dihedral(x[idx], y[idx], int(torch.randint(8, (1,), generator=torch.Generator().manual_seed(int(idx.sum()))).item()))
        loss = loss_fn(head(xb), yb)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    return head.eval()


@torch.no_grad()
def apply(head, x):
    return torch.cat([head(x[i:i + 128].cuda()).sigmoid().cpu() for i in range(0, len(x), 128)]).numpy()


def miou(pred, gt):
    out = np.zeros(len(pred))
    for c in (0, 1):
        p, g = pred == c, gt == c
        u = (p | g).sum((1, 2)); i = (p & g).sum((1, 2))
        out += np.where(u == 0, 1.0, i / np.maximum(u, 1))
    return out / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", required=True, type=Path)
    ap.add_argument("--occ", type=Path)
    ap.add_argument("--ig", type=Path)
    ap.add_argument("--frame-json", required=True, type=Path)
    ap.add_argument("--train-root", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=500)
    args = ap.parse_args()
    frame = json.loads(args.frame_json.read_text())
    rows = json.loads((args.train_root / "metadata.json").read_text())
    method_of = {r["image_name"]: r["method"] for r in rows}
    full_of = {r["image_name"]: bool(r["full_watermark"]) for r in rows}
    test_rows = json.loads((args.train_root.parent / "test" / "metadata.json").read_text())
    test_method = {r["image_name"]: r["method"] for r in test_rows}
    with np.load(args.train_root / "masks_128.npz") as z:
        masks = dict(zip(z["names"].tolist(), z["masks"].astype(np.float32)))
    folds = json.loads((args.reference / "protocol.json").read_text())["folds"]
    with np.load(args.folds / "oof_probs.npz") as z:
        oof = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
    with np.load(args.folds / "test_mean_probs.npz") as z:
        test_names, test_probs = z["names"].tolist(), z["probs"].astype(np.float32)
    occ_train, occ_test = load_maps(args.occ, "train"), load_maps(args.occ, "test")
    ig_train, ig_test = load_maps(args.ig, "train"), load_maps(args.ig, "test")
    print("MAPS occ", len(occ_train), len(occ_test), "ig", len(ig_train), len(ig_test), flush=True)

    names = [n for f in folds for n in f]
    X = features(names, np.stack([oof[n] for n in names]), occ_train, ig_train, frame["train"], method_of)
    Y = torch.from_numpy(np.stack([masks[n] for n in names]))
    fold_id = np.array([k for k, f in enumerate(folds) for _ in f])
    fused = np.zeros((len(names), 128, 128), np.float32)
    for k in range(5):
        tr, te = np.where(fold_id != k)[0], np.where(fold_id == k)[0]
        fused[te] = apply(fit(X[tr], Y[tr], args.steps, 20261121 + k), X[te])
    gt = Y.numpy() > 0.5
    before, after = miou(X[:, 0].numpy() > 0, gt), miou(fused > 0.5, gt)
    report = {"oof_before": float(before.mean()), "oof_fused": float(after.mean()),
              "folds_before": [float(before[fold_id == k].mean()) for k in range(5)],
              "folds_fused": [float(after[fold_id == k].mean()) for k in range(5)]}
    for m in METHODS:
        for f in (True, False):
            sel = np.array([method_of[n] == m and full_of[n] == f for n in names])
            report[f"{m}/full={f}"] = [float(before[sel].mean()), float(after[sel].mean())]
    print(json.dumps(report, indent=1), flush=True)
    head = fit(X, Y, args.steps, 20261199)
    fused_test = apply(head, features(test_names, test_probs, occ_test, ig_test, frame["test"], test_method))
    for k in range(5):
        d = args.out / f"fold_{k}"
        d.mkdir(parents=True, exist_ok=True)
        sel = np.where(fold_id == k)[0]
        np.savez_compressed(d / "outer_probs.npz", names=np.array([names[i] for i in sel]), probs=fused[sel].astype(np.float16), scores=after[sel])
        np.savez_compressed(d / "test_probs.npz", names=np.array(test_names), probs=fused_test.astype(np.float16))
    (args.out / "fusion_report.json").write_text(json.dumps(report, indent=1))
    print("FUSION_DONE", flush=True)


if __name__ == "__main__":
    main()
