"""Does message-template matching work at all on images we watermark ourselves?

Earlier, on official images, the local correlation between the image and the re-embedded message template
(known message minus complemented message) separated watermarked from edited cells at AUC 0.50-0.54. This runs
the SAME statistic on clean COCO pictures we embed with the kit: watermarked picture vs the same picture clean.
High AUC here and chance on official images would mean the official pictures went through a step our synthetic
data lacks; chance here means the statistic is useless by construction.
usage: template_selftest.py <kit> <clean.npy> <method> [n]
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

kit, clean_path, method = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else 24
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=kit / "models", device="cuda")
clean = np.load(clean_path, mmap_mode="r")
rng = np.random.default_rng(5)


def template(x, bits):
    good = wm.embed(x, bits).astype(np.float32)
    bad = wm.embed(x, [1 - v for v in bits]).astype(np.float32)
    return (good - bad) / 2 / 255


def corr_maps(x, T):
    xf = torch.from_numpy(x.copy()).permute(2, 0, 1)[None].float() / 255
    Tf = torch.from_numpy(T.copy()).permute(2, 0, 1)[None].float()
    res = {}
    for hp in (3, 7):
        hx = xf - F.avg_pool2d(xf, hp, 1, hp // 2, count_include_pad=False)
        hT = Tf - F.avg_pool2d(Tf, hp, 1, hp // 2, count_include_pad=False)
        xT, xx, TT = (hx * hT).mean(1, keepdim=True), (hx * hx).mean(1, keepdim=True), (hT * hT).mean(1, keepdim=True)
        for win in (17, 33):
            sm = lambda v: F.avg_pool2d(v, win, 1, win // 2, count_include_pad=False)  # noqa: E731
            cov, vx, vT = sm(xT), sm(xx), sm(TT)
            res[f"ncc_hp{hp}_w{win}"] = (cov / (vx * vT + 1e-14).sqrt())[0, 0].numpy()
    return res


def down(a):
    return a.reshape(128, 4, 128, 4).mean((1, 3))


def auc(pos, neg):
    s = np.concatenate([pos, neg]); label = np.concatenate([np.ones(len(pos), bool), np.zeros(len(neg), bool)])
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


pos, neg, corr_true, corr_other, amp = {}, {}, [], [], []
for i in rng.choice(len(clean), n, replace=False):
    x = np.asarray(clean[int(i)])
    bits = rng.integers(0, 2, wm.nbits).tolist()
    other = rng.integers(0, 2, wm.nbits).tolist()
    xw = wm.embed(x, bits)
    true_res = (xw.astype(np.float32) - x.astype(np.float32)) / 255
    amp.append(float(np.sqrt((true_res ** 2).mean()) * 255))
    T_w, T_c = template(xw, bits), template(x, bits)
    corr_true.append(float(np.corrcoef(true_res.ravel(), T_w.ravel())[0, 1]))
    corr_other.append(float(np.corrcoef(true_res.ravel(), template(xw, other).ravel())[0, 1]))
    for key, value in corr_maps(xw, T_w).items():
        pos.setdefault(key, []).append(down(value).ravel())
    for key, value in corr_maps(x, T_c).items():
        neg.setdefault(key, []).append(down(value).ravel())
print("SELFTEST", method, "residual RMS (grey levels)", round(float(np.mean(amp)), 3),
      "corr(true residual, template true msg)", round(float(np.mean(corr_true)), 3),
      "corr(true residual, template other msg)", round(float(np.mean(corr_other)), 3), flush=True)
for key in pos:
    print("SELFTEST", method, key, "cell AUC watermarked vs clean", round(auc(np.concatenate(pos[key]), np.concatenate(neg[key])), 4), flush=True)
print("SELFTEST_DONE", flush=True)
