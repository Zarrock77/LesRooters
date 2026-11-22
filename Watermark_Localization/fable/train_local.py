"""Local, message-aware watermark evidence: per-position bit logits read with a SMALL receptive field.

The blind detector asks "is there some watermark here?". Here the known message is used coherently: the shallow
stages of the kit extractor (local receptive field) are fine-tuned to output the message bits at every position;
the evidence map is the agreement between those local bits and the image's known message. Edited or clean areas
cannot agree with a random 256/100-bit message beyond chance, wherever they are.
Training uses only clean external sources + the kit embedder (banks from gen_layers.py). Official train images are
only scored (no fitting): per-image AUC and mIoU of the evidence map against the true masks.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_det import blend, make_mask, sample_area  # noqa: E402

cv2.setNumThreads(1)
NBITS = {"pixelseal": 256, "trustmark": 100}


class FullFrame(Dataset):
    """Full-frame watermarked sources with their embedded bits, optionally with edited holes."""
    def __init__(self, roots, method, train, length, seed=None):
        self.roots, self.method, self.train, self.length, self.seed = [r for r in roots.split(",") if r], method, train, length, seed
        self.ready = False

    def open(self):
        self.banks = []
        for root in self.roots:
            root = Path(root)
            clean = np.load(root / "clean.npy", mmap_mode="r")
            full = np.load(root / self.method / "full.npy", mmap_mode="r")
            bits = np.zeros((len(clean), NBITS[self.method]), np.uint8)
            old = root / self.method / "messages.npz"
            if old.exists():
                bits[:] = np.load(old)["full"]
            else:
                for f in sorted((root / self.method).glob("messages_*.npz")):
                    z = np.load(f)
                    bits[z["sources"]] = z["full"]
            n = len(clean)
            split = n - max(2, int(n * 0.04))
            self.banks.append((clean, full, bits, (0, split) if self.train else (split, n)))
        self.ready = True

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if not self.ready:
            self.open()
        rng = np.random.default_rng(None if self.seed is None else self.seed + index)
        clean, full, bits, (lo, hi) = self.banks[int(rng.integers(len(self.banks)))]
        i = int(rng.integers(lo, hi))
        S, W = np.asarray(clean[i]), np.asarray(full[i])
        u = rng.random()
        if u < 0.5:                                        # watermark with edited holes
            hole = make_mask(rng, sample_area(rng, 0.003, 0.5, 0.8))
            j = int(rng.integers(lo, hi))
            filler = S if rng.random() < 0.6 else np.asarray(clean[j])
            out, label = blend(W, filler, hole, rng), ~hole
        elif u < 0.8:                                      # watermark only on an object (same frame)
            obj = make_mask(rng, sample_area(rng, 0.004, 0.45, 0.75))
            j = int(rng.integers(lo, hi))
            bg = S if rng.random() < 0.5 else np.asarray(clean[j])
            out, label = blend(bg, W, obj, rng), obj
        else:
            out, label = np.array(W), np.ones((512, 512), bool)
        label32 = label.reshape(32, 16, 32, 16).mean((1, 3)) >= 0.5
        label128 = label.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5
        return (torch.from_numpy(np.ascontiguousarray(out)), torch.from_numpy(bits[i].astype(np.float32)),
                torch.from_numpy(label32), torch.from_numpy(label128))


class LocalDecoder(nn.Module):
    def __init__(self, kit, method, dim=256, pretrained=True):
        super().__init__()
        self.method, self.nbits = method, NBITS[method]
        sys.path.insert(0, str(kit))
        from wmloc.watermarks import load_watermark
        wm = load_watermark(method, models_dir=Path(kit) / "models", device="cpu")
        if method == "pixelseal":
            cn = wm.det.convnext
            self.stem = nn.ModuleList([nn.Sequential(cn.downsample_layers[0], cn.stages[0]),      # 96 @ 64
                                       nn.Sequential(cn.downsample_layers[1], cn.stages[1])])     # 192 @ 32
            chans, self.res = (96, 192), 256
        else:
            n = wm.net
            self.stem = nn.ModuleList([nn.Sequential(n.conv1, n.bn1, n.relu, n.maxpool, n.layer1),  # 256 @ 62
                                       n.layer2])                                                    # 512 @ 31
            chans, self.res = (256, 512), int(wm.resolution)
        for p in self.parameters():
            p.requires_grad_(True)
        self.lat0 = nn.Conv2d(chans[0], dim, 1)
        self.lat1 = nn.Conv2d(chans[1], dim, 1)
        self.mix = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(),
                                 nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.bits = nn.Conv2d(dim, self.nbits, 1)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def train(self, mode=True):
        super().train(mode)
        if self.method == "trustmark":                     # released batch-norm statistics stay fixed
            for m in self.stem.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, image_uint8):
        x = image_uint8.permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
        f0 = self.stem[0](x)
        f1 = self.stem[1](f0)
        z = self.lat1(f1) + F.adaptive_avg_pool2d(self.lat0(f0), f1.shape[-2:])
        logits = self.bits(self.mix(z))
        return F.interpolate(logits, size=(32, 32), mode="bilinear", align_corners=False) if logits.shape[-1] != 32 else logits

    def evidence(self, logits, bits):
        """Agreement of the local bits with the known message: about 0 where there is no watermark."""
        sign = (bits * 2 - 1)[:, :, None, None]
        return torch.tanh(logits * sign).mean(1)

    def presence_logit(self, logits, bits):
        return self.scale * self.evidence(logits, bits) + self.bias


class FullDecoder(nn.Module):
    """The whole kit extractor, fine-tuned so that its per-position bit logits (8x8 grid, 64 px cells) agree with
    the message only where the watermark is locally present. Starts from the released weights, which already
    decode the message."""
    def __init__(self, kit, method):
        super().__init__()
        self.method, self.nbits = method, NBITS[method]
        sys.path.insert(0, str(kit))
        from wmloc.watermarks import load_watermark
        wm = load_watermark(method, models_dir=Path(kit) / "models", device="cpu")
        if method == "pixelseal":
            self.det, self.res = wm.det, 256
        else:
            self.net, self.res = wm.net, int(wm.resolution)
        for p in self.parameters():
            p.requires_grad_(True)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def train(self, mode=True):
        super().train(mode)
        for m in self.modules():                            # released normalisation statistics stay fixed
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, image_uint8):
        x = image_uint8.permute(0, 3, 1, 2).float() / 255
        x = F.interpolate(x, size=(self.res, self.res), mode="bilinear", align_corners=False, antialias=True)
        if self.method == "pixelseal":
            emb = self.det.pixel_decoder.output_upscaling(self.det.convnext(self.det.preprocess(x)))
            lin = self.det.pixel_decoder.linear
            return (torch.einsum("bchw,oc->bohw", emb, lin.weight) + lin.bias[None, :, None, None])[:, 1:]
        n = self.net
        z = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
        z = n.layer4(n.layer3(n.layer2(n.layer1(z))))
        return torch.einsum("bchw,oc->bohw", z, n.fc.weight) + n.fc.bias[None, :, None, None]

    evidence = LocalDecoder.evidence
    presence_logit = LocalDecoder.presence_logit


def auc(s, label):
    label = np.asarray(label).ravel().astype(bool); s = np.asarray(s, np.float64).ravel()
    n1 = int(label.sum()); n0 = label.size - n1
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def miou_batch(pred, gt):
    out = np.zeros(len(pred))
    for c in (0, 1):
        p, g = pred == c, gt == c
        union = (p | g).sum((1, 2)); inter = (p & g).sum((1, 2))
        out += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


@torch.no_grad()
def presence_128(model, images, bits, batch=64):
    model.eval()
    out = []
    for s in range(0, len(images), batch):
        lg = model(images[s:s + batch].cuda())
        p = torch.sigmoid(model.presence_logit(lg, bits[s:s + batch].cuda()))
        out.append(F.interpolate(p[:, None], size=(128, 128), mode="bilinear", align_corners=False)[:, 0].cpu())
    return torch.cat(out).numpy()


def load_real(kit, method, split="train"):
    rows = [r for r in json.loads((kit / "release" / split / "metadata.json").read_text()) if r["method"] == method]
    images = torch.from_numpy(np.stack([np.array(Image.open(kit / "release" / split / "images" / (r["image_name"] + ".png")).convert("RGB")) for r in rows]))
    bits = torch.tensor([[int(b) for b in r["message"]] for r in rows], dtype=torch.float32)
    return rows, images, bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--iters", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--arch", default="local", choices=["local", "full"])
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261122)
    model = (LocalDecoder if args.arch == "local" else FullDecoder)(args.kit, args.method).cuda()
    train = FullFrame(args.layers, args.method, True, args.iters * args.batch)
    val = FullFrame(args.layers, args.method, False, 256, seed=99)
    loader = DataLoader(train, batch_size=args.batch, num_workers=args.workers, drop_last=True, pin_memory=True,
                        persistent_workers=True, worker_init_fn=lambda w: cv2.setNumThreads(1))
    items = [val[k] for k in range(len(val))]
    v_img, v_bits = torch.stack([b[0] for b in items]), torch.stack([b[1] for b in items])
    v_m32, v_m128 = torch.stack([b[2] for b in items]).numpy(), torch.stack([b[3] for b in items]).numpy()
    rows, r_img, r_bits = load_real(args.kit, args.method)
    with np.load(args.kit / "release/train/masks_128.npz") as z:
        allmasks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
    r_mask = np.stack([allmasks[r["image_name"]] for r in rows])
    frame = json.loads(Path(__file__).with_name("train_frame_decode.json").read_text())
    r_flag = np.array([frame[r["image_name"]]["acc"] >= 0.8 for r in rows])
    trunk = ("stem.", "det.", "net.")
    stem = [p for n, p in model.named_parameters() if n.startswith(trunk)]
    rest = [p for n, p in model.named_parameters() if not n.startswith(trunk)]
    opt = torch.optim.AdamW([{"params": stem, "lr": args.lr * 0.5}, {"params": rest, "lr": args.lr}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / 300) * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, it / args.iters)))))
    t0, run_bit, run_pres, history = time.time(), 0.0, 0.0, []
    for it, (image, bits, m32, _) in enumerate(loader, 1):
        model.train()
        image, bits, m32 = image.cuda(non_blocking=True), bits.cuda(non_blocking=True), m32.cuda(non_blocking=True).float()
        logits = model(image)
        if logits.shape[-1] != m32.shape[-1]:               # coarser grid: a cell is watermarked if most of it is
            m32 = (F.adaptive_avg_pool2d(m32[:, None], logits.shape[-1])[:, 0] >= 0.5).float()
        target = bits[:, :, None, None].expand_as(logits)
        bit_loss = (F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean(1) * m32).sum() / m32.sum().clamp_min(1)
        pres_loss = F.binary_cross_entropy_with_logits(model.presence_logit(logits, bits), m32)
        loss = bit_loss + pres_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step(); sched.step()
        run_bit = bit_loss.item() if it == 1 else 0.98 * run_bit + 0.02 * bit_loss.item()
        run_pres = pres_loss.item() if it == 1 else 0.98 * run_pres + 0.02 * pres_loss.item()
        if it % 100 == 0:
            print("IT", it, "bit", round(run_bit, 4), "presence", round(run_pres, 4), "it/s", round(it / (time.time() - t0), 2), flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                accs, aucs = [], []
                for s in range(0, len(v_img), 64):
                    lg = model(v_img[s:s + 64].cuda())
                    b = v_bits[s:s + 64].cuda()
                    hit = ((lg > 0).float() == b[:, :, None, None]).float().mean(1).cpu().numpy()
                    ev = model.evidence(lg, b).cpu().numpy()
                    g = lg.shape[-1]
                    for k in range(len(hit)):
                        m = v_m32[s + k].reshape(g, 32 // g, g, 32 // g).mean((1, 3)) >= 0.5
                        if m.any():
                            accs.append(hit[k][m].mean())
                        a = auc(ev[k], m)
                        if a is not None:
                            aucs.append(a)
            vp = presence_128(model, v_img, v_bits)
            vs = miou_batch(vp > 0.5, v_m128)
            rp = presence_128(model, r_img, r_bits)
            rs = miou_batch(rp > 0.5, r_mask)
            r_auc = np.array([auc(rp[k], r_mask[k]) for k in range(len(rp))])
            rec = {"it": it, "syn_local_bit_acc": float(np.mean(accs)), "syn_auc": float(np.mean(aucs)), "syn_miou": float(vs.mean()),
                   "real_miou": float(rs.mean()), "real_miou_fullframe": float(rs[r_flag].mean()), "real_miou_other": float(rs[~r_flag].mean()),
                   "real_auc_fullframe": float(r_auc[r_flag].mean()), "real_auc_other": float(r_auc[~r_flag].mean())}
            history.append(rec)
            print("EVAL", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in rec.items()}), flush=True)
            (args.out / "history.json").write_text(json.dumps(history, indent=1))
            torch.save({"model": model.state_dict(), "it": it, "method": args.method}, args.out / "ckpt_last.pt")
            np.savez_compressed(args.out / "real_presence_last.npz", names=np.array([r["image_name"] for r in rows]), probs=rp.astype(np.float16))
        if it >= args.iters:
            break
    print("LOCAL_DONE", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
