"""Does the KNOWN message help to find holes locally, through the embedder instead of the decoder?

The decoders cannot read the message from a small window (tile_probe.py), but the embedder tells us what the
watermark for this message looks like on this very picture: template = embed(image, m) - embed(image, 1 - m).
On pictures we watermark ourselves the template correlates at 0.65 with the true residual and at 0 for another
message. A small network learns the matched filter: image features x template features, pooled over windows.

Training: clean COCO pictures, full-frame watermark with the kit embedder, random holes. No official image.
Scoring only: official full-frame train images of the method (holes = edited regions), true message vs a random
message (the gap is what the message contributes).
usage: tpl_probe.py <kit> <clean.npy> <frame json> <method> <out dir> [iters]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402
from train_det import blend, down128, make_mask, miou_batch, sample_area  # noqa: E402

kit, clean_path, frame_json, method, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
iters = int(sys.argv[6]) if len(sys.argv) > 6 else 1500
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

out_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(3)
rng = np.random.default_rng(3)
wm = load_watermark(method, models_dir=kit / "models", device="cuda")
clean = np.load(clean_path, mmap_mode="r")
split = len(clean) - 400
B = 16


@torch.no_grad()
def embed(x, bits):
    """x [B,3,H,W] in [0,1] on cuda, bits [B,nbits] in {0,1} -> watermarked float image (kit embedder, batched)."""
    if method == "pixelseal":
        return wm.model.embed(x, msgs=bits.float(), is_video=False)["imgs_w"].clamp(0, 1)
    tm = wm.tm                                               # same steps as trustmark.encode, batched
    res = tm.model_resolution_enc
    cover = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    stego, _ = tm.encoder(cover, bits.float())
    residual = stego.clamp(-1, 1) - cover
    residual = residual - residual.mean(dim=(2, 3), keepdim=True)
    residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear")
    return ((x * 2 - 1 + residual).clamp(-1, 1) + 1) / 2


@torch.no_grad()
def template(x, bits):
    """Message-specific part of the watermark the embedder would put on this picture, in grey levels."""
    return (embed(x, bits) - embed(x, 1 - bits)).mul(127.5)


class Corr(nn.Module):
    """Learned matched filter: features of the picture times features of the template, pooled over windows."""
    def __init__(self, k=16, tc=3):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(3, 32, 5, padding=2), nn.GELU(), nn.Conv2d(32, 32, 5, padding=2), nn.GELU(), nn.Conv2d(32, k, 3, padding=1))
        self.g = nn.Sequential(nn.Conv2d(tc, 32, 5, padding=2), nn.GELU(), nn.Conv2d(32, 32, 5, padding=2), nn.GELU(), nn.Conv2d(32, k, 3, padding=1))
        self.agg = nn.Sequential(nn.Conv2d(12 * k, 96, 3, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
                                 nn.Conv2d(96, 96, 3, padding=2, dilation=2), nn.GroupNorm(8, 96), nn.GELU(),
                                 nn.Conv2d(96, 96, 3, padding=4, dilation=4), nn.GroupNorm(8, 96), nn.GELU(),
                                 nn.Conv2d(96, 1, 1))

    def features(self, x, t):
        a, b = self.f((x - 0.5) * 4), self.g(t)
        base = [F.avg_pool2d(v, 4) for v in (a * b, a * a, b * b)]
        return torch.cat([F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False) if k > 1 else v for v in base for k in (1, 3, 7, 15)], 1)

    def forward(self, x, t):
        return self.agg(self.features(x, t))[:, 0]


def batch(train=True, n=B):
    lo, hi = (0, split) if train else (split, len(clean))
    idx = rng.integers(lo, hi, n)
    x = torch.from_numpy(np.stack([clean[int(i)] for i in idx])).cuda().permute(0, 3, 1, 2).float() / 255
    bits = torch.from_numpy(rng.integers(0, 2, (n, wm.nbits))).cuda()
    xw = (embed(x, bits) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()      # stored pictures are 8 bit
    xc = (x * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    outs, labels = [], []
    for j in range(n):
        hole = make_mask(rng, sample_area(rng, 0.003, 0.5, 0.8))
        filler = xc[j] if rng.random() < 0.5 else xc[(j + 1) % n]
        outs.append(blend(xw[j], filler, hole, rng))
        labels.append(down128(~hole))
    image = torch.from_numpy(np.stack(outs)).cuda().permute(0, 3, 1, 2).float() / 255
    return image, bits, torch.from_numpy(np.stack(labels)).cuda()


def auc(s, label):
    s, label = np.asarray(s).ravel(), np.asarray(label, bool).ravel()
    n1, n0 = int(label.sum()), int((~label).sum())
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# official full-frame train images of this method (scoring only)
frame = json.loads(frame_json.read_text())["train"]
rows, images = load_split(kit, "train")
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
real = [(i, r) for i, r in enumerate(rows) if r["method"] == method and frame[r["image_name"]]["acc"] >= 0.8]
real_x = torch.stack([images[i] for i, _ in real]).permute(0, 3, 1, 2)
real_bits = torch.tensor([[int(c) for c in r["message"]] for _, r in real])
real_gt = np.stack([masks[r["image_name"]] for _, r in real])
print("REAL full-frame images", len(real), "hole share", round(float(1 - real_gt.mean()), 4), flush=True)

net = Corr(tc=3).cuda()
opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=iters, pct_start=0.1)


@torch.no_grad()
def predict_real(wrong):
    out = []
    for s in range(0, len(real_x), B):
        x = real_x[s:s + B].cuda().float() / 255
        bits = real_bits[s:s + B].cuda()
        if wrong:
            bits = torch.from_numpy(np.random.default_rng(11 + s).integers(0, 2, tuple(bits.shape))).cuda()
        out.append(net(x, template(x, bits)).sigmoid().cpu().numpy())
    return np.concatenate(out)


def evaluate(it):
    net.eval()
    state = rng.bit_generator.state
    probs, gts, probs_wrong = [], [], []
    with torch.no_grad():
        for _ in range(4):
            image, bits, label = batch(train=False)
            probs.append(net(image, template(image, bits)).sigmoid().cpu().numpy())
            probs_wrong.append(net(image, template(image, torch.randint_like(bits, 2))).sigmoid().cpu().numpy())
            gts.append(label.cpu().numpy())
    rng.bit_generator.state = state
    probs, probs_wrong, gts = np.concatenate(probs), np.concatenate(probs_wrong), np.concatenate(gts)
    r, rw = predict_real(False), predict_real(True)
    report = {"it": it, "syn_auc": round(auc(probs, gts), 4), "syn_auc_wrong_msg": round(auc(probs_wrong, gts), 4),
              "syn_miou": round(float(miou_batch(probs > 0.5, gts).mean()), 4),
              "real_auc": round(auc(r, real_gt), 4), "real_auc_wrong_msg": round(auc(rw, real_gt), 4),
              "real_miou": round(float(miou_batch(r > 0.5, real_gt).mean()), 4)}
    print("EVAL", json.dumps(report), flush=True)
    np.savez_compressed(out_dir / "real_probs.npz", names=np.array([r_["image_name"] for _, r_ in real]), probs=r.astype(np.float16))
    net.train()


t0 = time.time()
for it in range(1, iters + 1):
    image, bits, label = batch()
    logits = net(image, template(image, bits))
    loss = F.binary_cross_entropy_with_logits(logits, label.float())
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step(); sched.step()
    if it % 50 == 0:
        print("IT", it, "loss", round(float(loss), 4), "it/s", round(it / (time.time() - t0), 2), flush=True)
    if it % 250 == 0 or it == iters:
        evaluate(it)
torch.save(net.state_dict(), out_dir / "corr.pt")
print("TPL_DONE", flush=True)
