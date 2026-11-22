"""TrustMark ships a watermark REMOVAL network. What does it take out of official pictures, and where?

For official TrustMark pictures: (1) does removal erase the official watermark (decoder accuracy after removal)?
(2) does the removed signal match our encoder's residual for the known message, or is it another pattern?
(3) is the removed signal's local energy, or its local agreement with the decoder-relevant direction, different in
kept regions and in holes (labels for measuring only)? Same numbers for pictures we embed ourselves.
An earlier test used the remover only as an occluder for the decoder; this one looks at what it removes.
usage: remover_probe.py <kit> <clean.npy> <frame json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_det import make_mask, sample_area  # noqa: E402
from wm_template import embed  # noqa: E402

kit, clean_path, frame_json = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark("trustmark", models_dir=kit / "models", device="cuda")
tm = wm.tm
rng = np.random.default_rng(29)


@torch.no_grad()
def removed(x):
    """x [B,3,512,512] in [0,1] -> what the removal network subtracts, in grey levels (same steps as remove_watermark)."""
    res = tm.model_resolution_remove
    s = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    d = tm.removal(s).clamp(-1, 1) - s
    return F.interpolate(d, size=x.shape[-2:], mode="bilinear") * 127.5


def auc(s, label):
    s, label = np.asarray(s).ravel(), np.asarray(label, bool).ravel()
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def down(a):
    return a.reshape(128, 4, 128, 4).mean((1, 3))


def study(tag, images, masks128, bits):
    aucs_e, aucs_c, corr, acc_before, acc_after, rms_in, rms_out = [], [], [], [], [], [], []
    for s in range(0, len(images), 8):
        x = torch.from_numpy(images[s:s + 8]).permute(0, 3, 1, 2).cuda().float() / 255
        b = torch.from_numpy(bits[s:s + 8]).cuda()
        d = removed(x)                                              # what the remover takes out (negative of the watermark estimate)
        ours = (embed(wm, x, b) - x) * 255                          # what our encoder would add for the known message
        sign = b.float() * 2 - 1
        acc_before.append(((wm.logits(x) > 0) == (sign > 0)).float().mean(1).cpu().numpy())
        acc_after.append(((wm.logits((x + d / 255).clamp(0, 1)) > 0) == (sign > 0)).float().mean(1).cpu().numpy())
        energy = F.avg_pool2d((d ** 2).mean(1, keepdim=True), 4)[:, 0].sqrt().cpu().numpy()
        agree = F.avg_pool2d((-d * ours).mean(1, keepdim=True), 9, 1, 4)                # local agreement of the estimates
        agree = F.avg_pool2d(agree, 4)[:, 0].cpu().numpy()
        for j in range(len(x)):
            m = masks128[s + j]
            if 0.005 < (~m).mean() < 0.995:
                aucs_e.append(auc(energy[j], m)); aucs_c.append(auc(agree[j], m))
                rms_in.append(energy[j][m].mean()); rms_out.append(energy[j][~m].mean())
            dd, oo = -d[j].flatten(), ours[j].flatten()
            corr.append(float((dd * oo).sum() / (dd.norm() * oo.norm() + 1e-9)))
    print("REMOVER", f"{tag:26s}", "n", len(images), "| bit acc before", round(float(np.concatenate(acc_before).mean()), 3), "after removal", round(float(np.concatenate(acc_after).mean()), 3),
          "| correlation(removed signal, our encoder's residual)", round(float(np.mean(corr)), 3),
          "| removed energy in kept / clean cells", round(float(np.mean(rms_in)), 2), "/", round(float(np.mean(rms_out)), 2),
          "| cell AUC kept vs clean: energy", round(float(np.nanmean(aucs_e)), 3), "agreement with our residual", round(float(np.nanmean(aucs_c)), 3), flush=True)


meta = json.loads((kit / "release/train/metadata.json").read_text())
frame = json.loads(frame_json.read_text())["train"]
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
for regime in ("full", "other"):
    sel = [r for r in meta if r["method"] == "trustmark" and (frame[r["image_name"]]["acc"] >= 0.8) == (regime == "full")][:64]
    imgs = np.stack([np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB")) for r in sel])
    study(f"official {regime}", imgs, np.stack([masks[r["image_name"]] for r in sel]), np.array([[int(c) for c in r["message"]] for r in sel]))
clean = np.load(clean_path, mmap_mode="r")
idx = rng.choice(len(clean), 32, replace=False)
S = np.stack([clean[int(i)] for i in idx])
bits = rng.integers(0, 2, (32, 100))
W = np.stack([wm.embed(S[j], bits[j].tolist()) for j in range(32)])
keep = np.stack([~make_mask(rng, sample_area(rng, 0.003, 0.5, 0.8)) for _ in range(32)])
comp = np.where(keep[..., None], W, S)
study("self-made, holes", comp, np.stack([k.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5 for k in keep]), bits)
print("REMOVER_DONE", flush=True)
