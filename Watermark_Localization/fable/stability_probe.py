"""GPU probe: is a freshly embedded watermark pixel-exact reproducible by re-embedding, and how strong does the kit
decoder read it, compared with what we measure on the official images?

For clean COCO sources S: W = embed(S, m). Then re-embed on W and compare residuals; also decode margins.
If fresh watermarks are far 'cleaner' than the official images, the official images were post-processed after
watermarking and the synthetic training data must imitate that. Nothing is trained.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

kit, bank = Path(sys.argv[1]), Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
out = Path(sys.argv[4]) if len(sys.argv) > 4 else None
sys.path.insert(0, str(kit))
import onnxruntime as ort  # noqa: E402
ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402

clean = np.load(bank / "clean.npy", mmap_mode="r")
rng = np.random.default_rng(5)
N = 24


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().to(device)


def box(t, k):
    return F.avg_pool2d(t, k, 1, k // 2, count_include_pad=False)


def beta(x, T, hp=7, win=33):
    hx, hT = x - box(x, hp), T - box(T, hp)
    cov = box((hx * hT).mean(1, keepdim=True), win); vT = box((hT * hT).mean(1, keepdim=True), win)
    return float((cov / (vT + 1e-10)).clamp(-4, 4).mean())


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / ((a * a).sum().sqrt() * (b * b).sum().sqrt() + 1e-12))


saved = {}
for method in ("pixelseal", "trustmark", "stegastamp"):
    wm = load_watermark(method, models_dir=kit / "models", device=device)
    rec = {k: [] for k in ("corr_true_re", "corr_true_antisym", "beta_fresh", "margin_fresh", "acc_fresh", "rms_true", "margin_clean")}
    keep = []
    for i in range(N):
        S = np.array(clean[len(clean) - 1 - i])
        bits = rng.integers(0, 2, wm.nbits)
        W = np.asarray(wm.embed(S, bits.tolist()), dtype=np.uint8)
        good = np.asarray(wm.embed(W, bits.tolist()), dtype=np.float32)
        bad = np.asarray(wm.embed(W, (1 - bits).tolist()), dtype=np.float32)
        T_true = to_t(W.astype(np.float32) - S) / 255
        T_re = to_t(good - W) / 255
        T_anti = to_t(good - bad) / 510
        sign = torch.tensor(bits, dtype=torch.float32, device=device) * 2 - 1
        lg = wm.logits(to_t(W) / 255).to(device)[0]
        lg0 = wm.logits(to_t(S) / 255).to(device)[0]
        rec["corr_true_re"].append(corr(T_true, T_re)); rec["corr_true_antisym"].append(corr(T_true, T_anti))
        rec["beta_fresh"].append(beta(to_t(W) / 255, T_anti))
        rec["margin_fresh"].append(float((lg * sign).mean())); rec["acc_fresh"].append(float(((lg > 0) == (sign > 0)).float().mean()))
        rec["margin_clean"].append(float((lg0 * sign).mean()))
        rec["rms_true"].append(float(T_true.pow(2).mean().sqrt() * 255))
        keep.append((S, W, bits))
    print("STABILITY", method, json.dumps({k: round(float(np.mean(v)), 3) for k, v in rec.items()}), flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"fresh_{method}.npz", S=np.stack([k[0] for k in keep]), W=np.stack([k[1] for k in keep]),
                            bits=np.stack([k[2] for k in keep]))
    del wm
    torch.cuda.empty_cache()
print("STABILITY_DONE", flush=True)
