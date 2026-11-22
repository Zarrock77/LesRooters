"""Which TrustMark encoder wrote the official TrustMark watermarks?

On official pictures the kit decoder (variant Q) reads the message with a higher margin than on our own Q
embeddings, yet the pixels show no trace of the Q encoder's residual (template 0.08x, no jump across mask edges),
while PixelSeal and StegaStamp do match the kit. Test: embed with every released TrustMark variant (Q, B, C, P),
decode with the kit's Q decoder, and correlate each variant's template with official full-frame pictures and with
the jump across official mask edges. A variant that decodes with Q and matches the official pixels is the one used.
usage: tm_variant_probe.py <kit> <clean.npy> <frame json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

kit, clean_path, frame_json = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, str(kit))
from trustmark import TrustMark as TM  # noqa: E402
from wmloc.watermarks import load_watermark  # noqa: E402

wmq = load_watermark("trustmark", models_dir=kit / "models", device="cuda")
rng = np.random.default_rng(23)
clean = np.load(clean_path, mmap_mode="r")
meta = json.loads((kit / "release/train/metadata.json").read_text())
frame = json.loads(frame_json.read_text())["train"]


def box(v, k):
    return v if k == 1 else F.avg_pool2d(v, k, 1, k // 2, count_include_pad=False)


@torch.no_grad()
def embed(tm, x, bits):
    """Batched float embedding with the encoder of `tm` (same steps as trustmark.encode)."""
    res = tm.model_resolution_enc
    cover = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
    stego, _ = tm.encoder(cover, bits.float())
    residual = stego.clamp(-1, 1) - cover
    residual = residual - residual.mean(dim=(2, 3), keepdim=True)
    residual = F.interpolate(residual, size=x.shape[-2:], mode="bilinear")
    strength = 1.25 if tm.model_type == "P" else 1.0
    return ((x * 2 - 1 + strength * residual).clamp(-1, 1) + 1) / 2


def z(x, t, lo=1, hi=5):
    a, b = box(x, lo) - box(x, hi), box(t, lo) - box(t, hi)
    a = a - a.mean((2, 3), keepdim=True); b = b - b.mean((2, 3), keepdim=True)
    c = torch.fft.irfft2(torch.fft.rfft2(a) * torch.fft.rfft2(b).conj(), s=a.shape[-2:]).sum(1)
    flat = c.flatten(1)
    return round(float(((c[:, 0, 0] - flat.mean(1)) / flat.std(1)).median()), 2)


# official full-frame pictures and official transplants (edges)
sel = [r for r in meta if r["method"] == "trustmark" and frame[r["image_name"]]["acc"] >= 0.8][:32]
x_off = torch.from_numpy(np.stack([np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB")) for r in sel])).permute(0, 3, 1, 2).cuda().float() / 255
b_off = torch.tensor([[int(c) for c in r["message"]] for r in sel]).cuda()
idx = rng.choice(len(clean), 16, replace=False)
xc = torch.from_numpy(np.stack([clean[int(i)] for i in idx])).permute(0, 3, 1, 2).cuda().float() / 255
bits = torch.from_numpy(rng.integers(0, 2, (16, 100))).cuda()


def edge_slope(tm, rows_):
    jr = rr = 0.0
    for r in rows_:
        img = np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB"))
        mask = np.array(Image.open(kit / "release/train/masks" / (r["image_name"] + ".png")).convert("L")) > 127
        x = torch.from_numpy(img).permute(2, 0, 1)[None].cuda().float() / 255
        res = ((embed(tm, x, torch.tensor([[int(c) for c in r["message"]]]).cuda()) - x)[0].permute(1, 2, 0).cpu().numpy() * 255)
        xf = img.astype(np.float32)
        for a, b_, dy, dx in ((mask[:, :-1], mask[:, 1:], 0, 1), (mask[:-1], mask[1:], 1, 0)):
            ys, xs = np.where(a & ~b_)
            jump, ref = xf[ys, xs] - xf[ys + dy, xs + dx], res[ys, xs]
            jr += float((jump * ref).sum()); rr += float((ref * ref).sum())
            ys, xs = np.where(~a & b_)
            jump, ref = xf[ys + dy, xs + dx] - xf[ys, xs], res[ys + dy, xs + dx]
            jr += float((jump * ref).sum()); rr += float((ref * ref).sum())
    return round(jr / max(rr, 1e-9), 3)


transplants = [r for r in meta if r["method"] == "trustmark" and frame[r["image_name"]]["acc"] < 0.8][:40]
for variant in ("Q", "B", "C", "P"):
    try:
        tm = TM(use_ECC=False, verbose=False, secret_len=100, model_type=variant, device="cuda")
    except Exception as error:  # noqa: BLE001
        print("VARIANT", variant, "could not be loaded:", repr(error)[:160], flush=True)
        continue
    xw = (embed(tm, xc, bits) * 255).round() / 255
    lg = wmq.logits(xw).float().cpu()
    sign = bits.float().cpu() * 2 - 1
    acc, margin = float(((lg > 0) == (sign > 0)).float().mean()), float((lg * sign).mean())
    rms = float(((xw - xc) * 255).pow(2).mean().sqrt())
    t_off = (embed(tm, x_off, b_off) - embed(tm, x_off, 1 - b_off)) * 127.5
    print("VARIANT", variant, "| own embedding read by the kit Q decoder: bit acc", round(acc, 4), "margin", round(margin, 2), "residual RMS", round(rms, 2),
          "| template vs OFFICIAL full-frame: z fine", z(x_off, t_off), "z mid", z(x_off, t_off, 5, 17),
          "| jump across official edges, slope on this variant's residual: full-frame", edge_slope(tm, sel), "non-full", edge_slope(tm, transplants), flush=True)
print("VARIANT_DONE", flush=True)
