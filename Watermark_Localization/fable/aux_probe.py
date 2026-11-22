"""GPU probe on official train images: which extra signals say WHERE the watermark is?

 1. TrustMark's own removal network: residual = image - removed(image).
 2. Message template (re-embedding the known message) correlated with the image in several frequency bands.
 3. Occlusion of the kit decoder in the full frame (destroy one cell / keep one cell) with the known message.
Each signal is scored by per-image AUC against the true mask (128 grid, or cell grid for occlusion).
Nothing is trained. Read-only on the kit.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

kit, out = Path(sys.argv[1]), Path(sys.argv[2])
device = sys.argv[3] if len(sys.argv) > 3 else "cuda:0"
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 40
sys.path.insert(0, str(kit))
import onnxruntime as ort  # noqa: E402
ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402

METHODS = ("trustmark", "pixelseal", "stegastamp")
rows = json.loads((kit / "release/train/metadata.json").read_text())


def auc(s, label):
    label = np.asarray(label).ravel().astype(bool)
    s = np.asarray(s, dtype=np.float64).ravel()
    n1 = int(label.sum()); n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def gauss(t, sigma):
    if sigma <= 0:
        return t
    k = int(sigma * 3) * 2 + 1
    ax = torch.arange(k, dtype=torch.float32, device=t.device) - k // 2
    g = torch.exp(-ax ** 2 / (2 * sigma ** 2)); g = g / g.sum()
    c = t.shape[1]
    t = F.pad(t, (k // 2,) * 4, mode="reflect")
    t = F.conv2d(t, g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(t, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)


def to_t(a):
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float().to(device)


def down128(t):
    return F.avg_pool2d(t, 4)[0, 0].cpu().numpy()


results, t0 = [], time.time()
for method in METHODS:
    wm = load_watermark(method, models_dir=kit / "models", device=device)
    chosen, counts = [], {"full": 0, "transplant": 0}
    for r in rows:
        if r["method"] != method:
            continue
        m = np.array(Image.open(kit / "release/train/masks" / (r["image_name"] + ".png")).convert("L")) > 127
        ys, xs = np.where(m)
        fill = m.sum() / ((ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()))
        group = "full" if r["full_watermark"] else ("rect" if fill > 0.999 else "transplant")
        if group in counts and counts[group] < limit:
            counts[group] += 1
            chosen.append((r, m, group))
    for r, m, group in chosen:
        x = np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB"))
        bits = [int(b) for b in r["message"]]
        sign = torch.tensor(bits, dtype=torch.float32, device=device) * 2 - 1
        m128 = m.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
        rec = {"name": r["image_name"], "method": method, "group": group, "area": float(m128.mean())}
        xt = to_t(x) / 255
        # 1. removal network (TrustMark only)
        if method == "trustmark":
            try:
                removed = np.asarray(wm.tm.remove_watermark(Image.fromarray(x)).convert("RGB"), dtype=np.float32)
                d = to_t(x.astype(np.float32) - removed) / 255
                for name, sig in (("rm_abs", d.abs().mean(1, keepdim=True)), ("rm_sq", (d ** 2).mean(1, keepdim=True))):
                    for win in (9, 33):
                        v = down128(F.avg_pool2d(sig, win, 1, win // 2, count_include_pad=False))
                        rec[f"{name}_w{win}"] = auc(v, m128)
                rec["rm_rms"] = float(d.pow(2).mean().sqrt() * 255)
            except Exception as error:  # the probe must not die on an API difference
                rec["rm_error"] = repr(error)[:200]
        # 2. message template in frequency bands
        good = np.asarray(wm.embed(x, bits), dtype=np.float32)
        bad = np.asarray(wm.embed(x, [1 - b for b in bits]), dtype=np.float32)
        T = to_t(good - bad) / 510
        rec["template_rms"] = float(T.pow(2).mean().sqrt() * 255)
        for lo, hi in ((0, 1.5), (1.5, 4), (4, 10), (0, 4)):
            bx, bT = gauss(xt, lo) - gauss(xt, hi), gauss(T, lo) - gauss(T, hi)
            for win in (17, 49):
                sm = lambda v: F.avg_pool2d(v, win, 1, win // 2, count_include_pad=False)  # noqa: E731
                cov = sm((bx * bT).mean(1, keepdim=True)); vx = sm((bx * bx).mean(1, keepdim=True)); vT = sm((bT * bT).mean(1, keepdim=True))
                rec[f"tpl_ncc_b{lo}-{hi}_w{win}"] = auc(down128(cov / (vx * vT + 1e-16).sqrt()), m128)
                rec[f"tpl_beta_b{lo}-{hi}_w{win}"] = auc(down128(cov / (vT + 1e-12)), m128)
        # 3. occlusion of the decoder in the full frame
        mean_img = xt.mean((2, 3), keepdim=True).expand_as(xt)
        blur_img = gauss(xt, 12.0)
        base = float((wm.logits(xt).to(device) * sign).mean())
        rec["M_full"] = base
        for G in (8, 16):
            c = 512 // G
            cov_cells = m.reshape(G, c, G, c).mean((1, 3)) >= 0.5
            for dname, dimg in (("mean", mean_img), ("blur", blur_img)):
                m0 = float((wm.logits(dimg).to(device) * sign).mean())
                for mode in ("remove", "keep"):
                    if G == 16 and mode == "keep":
                        continue
                    batch = []
                    for gy in range(G):
                        for gx in range(G):
                            b = (xt if mode == "remove" else dimg).clone()
                            src = dimg if mode == "remove" else xt
                            b[:, :, gy * c:(gy + 1) * c, gx * c:(gx + 1) * c] = src[:, :, gy * c:(gy + 1) * c, gx * c:(gx + 1) * c]
                            batch.append(b)
                    lg = torch.cat([wm.logits(torch.cat(batch[i:i + 64])).to(device) for i in range(0, len(batch), 64)])
                    mg = (lg * sign).mean(1).reshape(G, G).cpu().numpy()
                    a = base - mg if mode == "remove" else mg - m0
                    rec[f"occ_G{G}_{dname}_{mode}"] = auc(a, cov_cells)
        results.append(rec)
    print("AUX", method, counts, round(time.time() - t0, 1), "s", flush=True)
    del wm
    torch.cuda.empty_cache()
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results))
for method in METHODS:
    for group in ("full", "transplant"):
        sub = [r for r in results if r["method"] == method and r["group"] == group]
        if not sub:
            continue
        print(f"\n== {method} {group} n={len(sub)} template rms={np.mean([r['template_rms'] for r in sub]):.2f}/255 M_full={np.mean([r['M_full'] for r in sub]):.2f}"
              + (f" remover rms={np.mean([r['rm_rms'] for r in sub if 'rm_rms' in r]):.2f}" if any('rm_rms' in r for r in sub) else "")
              + (" RM_ERROR " + sub[0]["rm_error"] if "rm_error" in sub[0] else ""))
        keys = sorted(k for k in sub[0] if k.startswith(("rm_abs", "rm_sq", "tpl_", "occ_")))
        scored = []
        for k in keys:
            v = [r[k] for r in sub if r.get(k) is not None]
            if v:
                scored.append((float(np.mean(v)), k, float(np.median(v))))
        for mean_auc, k, med in sorted(scored, key=lambda s: -abs(s[0] - 0.5))[:12]:
            print(f"   {k:28s} AUC mean={mean_auc:.3f} median={med:.3f}")
print("AUX_DONE", round(time.time() - t0, 1), "s", flush=True)
