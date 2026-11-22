"""GPU probe (TrustMark): occlusion with TrustMark's own removal network as the 'destroyer'.

Instead of blanking a window (which also removes image content the decoder relies on), the window is replaced by
the same pixels with the watermark removed by Adobe's released removal model. If the window carried the
watermark, the decoder's agreement with the known message drops; if it was edited, nothing changes.
Reports per-image AUC and the mIoU reachable with the best per-image threshold (an upper bound, for diagnosis only).
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
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10 ** 9
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark("trustmark", models_dir=kit / "models", device=device)
rows = [r for r in json.loads((kit / "release/train/metadata.json").read_text()) if r["method"] == "trustmark"][:limit]
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))


def auc(s, label):
    label = np.asarray(label).ravel().astype(bool); s = np.asarray(s, np.float64).ravel()
    n1 = int(label.sum()); n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def miou(p, g):
    s = 0.0
    for c in (0, 1):
        a, b = p == c, g == c
        u = (a | b).sum()
        s += 1.0 if u == 0 else (a & b).sum() / u
    return s / 2


@torch.no_grad()
def margins(batch, sign):
    out = []
    for i in range(0, len(batch), 128):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = wm.net(wm._prep(batch[i:i + 128])).float()
        out.append((lg * sign).mean(1))
    return torch.cat(out)


records, names, maps, t0 = [], [], [], time.time()
for n, r in enumerate(rows):
    name = r["image_name"]
    x_np = np.array(Image.open(kit / "release/train/images" / (name + ".png")).convert("RGB"))
    removed = np.asarray(wm.tm.remove_watermark(Image.fromarray(x_np)).convert("RGB"), dtype=np.uint8)
    x = torch.from_numpy(x_np).permute(2, 0, 1)[None].float().div(255).to(device)
    d = torch.from_numpy(removed).permute(2, 0, 1)[None].float().div(255).to(device)
    sign = torch.tensor([int(b) for b in r["message"]], dtype=torch.float32, device=device) * 2 - 1
    m_full, m_removed = float(margins(x, sign)[0]), float(margins(d, sign)[0])
    channels = []
    for size, stride in ((64, 32), (32, 16)):
        pos = [(y, xx) for y in range(0, 512 - size + 1, stride) for xx in range(0, 512 - size + 1, stride)]
        drops = []
        for s in range(0, len(pos), 256):
            chunk = pos[s:s + 256]
            batch = x.repeat(len(chunk), 1, 1, 1)
            for k, (y, xx) in enumerate(chunk):
                batch[k, :, y:y + size, xx:xx + size] = d[0, :, y:y + size, xx:xx + size]
            drops.append(m_full - margins(batch, sign))
        drop = torch.cat(drops)
        acc = torch.zeros(128, 128, device=device); cnt = torch.zeros(128, 128, device=device)
        for k, (y, xx) in enumerate(pos):
            acc[y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += drop[k]
            cnt[y // 4:(y + size) // 4, xx // 4:(xx + size) // 4] += 1
        channels.append((acc / cnt).cpu().numpy())
    gt = masks[name]
    rec = {"name": name, "full": bool(r["full_watermark"]), "M_full": m_full, "M_removed": m_removed,
           "residual_rms": float((x - d).pow(2).mean().sqrt() * 255)}
    for key, v in (("w64", channels[0]), ("w32", channels[1]), ("sum", channels[0] + 4 * channels[1])):
        rec["auc_" + key] = auc(v, gt)
        rec["oracle_" + key] = max(miou(v > q, gt) for q in np.quantile(v, np.linspace(0.02, 0.98, 49)))
    rec["const"] = max(gt.mean(), 1 - gt.mean()) / 2
    records.append(rec); names.append(name); maps.append(np.stack(channels).astype(np.float16))
    if n % 20 == 0:
        print("RMOCC", n, "/", len(rows), round(time.time() - t0, 1), "s", flush=True)
out.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out / "train_trustmark_rm.npz", names=np.array(names), maps=np.stack(maps))
(out / "records.json").write_text(json.dumps(records))
for full in (True, False):
    sub = [r for r in records if r["full"] == full]
    line = {k: round(float(np.mean([r[k] for r in sub if r[k] is not None])), 3) for k in sub[0] if k not in ("name", "full")}
    print("RMOCC_SUMMARY", "full" if full else "sparse", len(sub), json.dumps(line), flush=True)
print("RMOCC_DONE", round(time.time() - t0, 1), "s", flush=True)
