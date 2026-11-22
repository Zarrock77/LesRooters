"""Train a dense watermark-presence detector on synthetic compositions that mimic how the dataset was built.

Training uses ONLY clean external sources + the kit's embedders (layers from gen_layers.py). The 510 official train
images are never used for fitting here: they are scored periodically as a pure held-out check.
Fable's own folder; reads the kit read-only.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

cv2.setNumThreads(1)
METHODS = ("pixelseal", "trustmark", "stegastamp")
YY, XX = np.mgrid[0:512, 0:512].astype(np.float32)


# ----------------------------------------------------------------------------- masks
def sample_area(rng, lo, hi, power):
    return lo * (hi / lo) ** (rng.random() ** power)


def noise_blob(rng, area_px):
    g = int(rng.choice([8, 12, 16, 24, 32, 48]))
    field = cv2.resize(rng.standard_normal((g, g)).astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
    r = math.sqrt(area_px / math.pi) * rng.uniform(1.1, 2.6)
    cx, cy = rng.uniform(0, 512, 2)
    ax = rng.uniform(0.5, 2.0)
    env = ((XX - cx) ** 2 * ax + (YY - cy) ** 2 / ax) / (r * r)
    score = field * rng.uniform(0.4, 2.5) - env
    k = int(min(max(area_px, 16), 512 * 512 - 1))
    thr = np.partition(score.ravel(), -k)[-k]
    return score >= thr


def shape_blob(rng, area_px):
    m = np.zeros((512, 512), np.uint8)
    kind = rng.choice(["ellipse", "poly", "stroke", "rect"], p=[0.3, 0.35, 0.15, 0.2])
    cx, cy = rng.uniform(20, 492, 2)
    if kind == "ellipse":
        asp = rng.uniform(1, 4)
        a, b = math.sqrt(area_px * asp / math.pi), math.sqrt(area_px / asp / math.pi)
        cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(a)), max(1, int(b))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
    elif kind == "poly":
        n = int(rng.integers(5, 16))
        base = math.sqrt(area_px / math.pi) * 1.15
        ang = np.sort(rng.uniform(0, 2 * math.pi, n))
        rad = base * rng.uniform(0.45, 1.45, n)
        pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
        cv2.fillPoly(m, [pts], 1)
    elif kind == "stroke":
        t = int(rng.integers(5, 28))
        length, n = area_px / t, int(rng.integers(3, 9))
        pts, ang = [(cx, cy)], rng.uniform(0, 2 * math.pi)
        for _ in range(n):
            ang += rng.normal(0, 0.8)
            pts.append((pts[-1][0] + length / n * math.cos(ang), pts[-1][1] + length / n * math.sin(ang)))
        cv2.polylines(m, [np.array(pts, np.int32)], False, 1, thickness=t)
    else:
        asp = math.exp(rng.uniform(-1.6, 1.6))
        w, h = math.sqrt(area_px * asp), math.sqrt(area_px / asp)
        cv2.rectangle(m, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), 1, -1)
    return m.astype(bool)


def field_mask(rng, frac):
    """Blob field: a smooth random field over the whole frame, thresholded so that a share `frac` is covered.

    Many official hole masks of full-frame pictures look like this at 512 px (release/train/masks): several scattered
    smooth blobs, from a few dots to a camouflage pattern, not one compact region."""
    g = int(rng.choice([4, 5, 6, 7, 8, 10, 12]))           # features of 130 to 45 px, as in the official masks (finer looks like speckle)
    field = cv2.resize(rng.standard_normal((g, g)).astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
    if g <= 6 and rng.random() < 0.3:                      # a second, finer octave: blobs of mixed sizes
        field += 0.6 * cv2.resize(rng.standard_normal((2 * g, 2 * g)).astype(np.float32), (512, 512), interpolation=cv2.INTER_CUBIC)
    k = int(min(max(frac * 512 * 512, 16), 512 * 512 - 1))
    return field >= np.partition(field.ravel(), -k)[-k]


def make_mask(rng, frac):
    n = int(rng.choice([1, 2, 3, 4, 6, 10], p=[0.5, 0.18, 0.12, 0.1, 0.06, 0.04]))
    parts = rng.dirichlet(np.ones(n) * 1.5) * frac * 512 * 512
    m = np.zeros((512, 512), bool)
    for a in parts:
        if a < 12:
            continue
        m |= noise_blob(rng, a) if rng.random() < 0.5 else shape_blob(rng, a)
    return m


def object_mask(rng, inst_map, lo=0.004, hi=0.7):
    """A real object silhouette (one to three annotated instances of the picture), or None when none is suitable."""
    counts = np.bincount(inst_map.ravel(), minlength=256)[1:]
    ids = np.nonzero(counts >= lo * 512 * 512)[0] + 1
    if len(ids) == 0:
        return None
    n = min(len(ids), int(rng.choice([1, 2, 3], p=[0.7, 0.2, 0.1])))
    m = np.isin(inst_map, rng.choice(ids, n, replace=False))
    if rng.random() < 0.1:                                 # some edits use the box of the object
        ys, xs = np.where(m)
        m = np.zeros_like(m)
        m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    else:                                                  # edit masks are rarely pixel-tight
        r = int(rng.choice([0, 0, 1, 2, 4, 7]))
        if r:
            m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))).astype(bool)
    return m if m.mean() <= hi else None


def blend(base, top, mask, rng, sigmas=(0.0, 0.0, 0.7, 1.5, 2.5)):
    """`top` where mask, `base` elsewhere, optionally with a soft seam like generative compositing."""
    sigma = float(rng.choice(sigmas))
    if sigma == 0:
        out = base.copy()
        out[mask] = top[mask]
        return out
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)[..., None]
    return np.clip(np.rint(base.astype(np.float32) * (1 - alpha) + top.astype(np.float32) * alpha), 0, 255).astype(np.uint8)


def down128(mask):
    return mask.reshape(128, 4, 128, 4).mean((1, 3)) >= 0.5


# ----------------------------------------------------------------------------- data
class Bank:
    """One clean source bank with its pre-embedded layers (possibly written by several shards)."""
    def __init__(self, root, methods, hold_fraction=0.04):
        self.root = Path(root)
        self.clean = np.load(self.root / "clean.npy", mmap_mode="r")
        n = len(self.clean)
        self.split = n - max(2, int(n * hold_fraction))
        self.full = {m: np.load(self.root / m / "full.npy", mmap_mode="r") for m in methods}
        # Optional diffusion re-synthesis of every source (same picture, regenerated): the look of edited regions.
        self.regen = np.load(self.root / "regen.npy", mmap_mode="r") if (self.root / "REGEN_DONE").exists() else None
        # Optional instance masks of the source pictures (prepare_instances.py): object-shaped edits.
        self.inst = np.load(self.root / "inst.npy", mmap_mode="r") if (self.root / "inst.npy").exists() else None
        self.pdat, self.pidx = {}, {}
        for m in methods:
            files = sorted((self.root / m).glob("patch_index*.npy"))
            assert files, f"no patches for {m} in {root}"
            parts, self.pdat[m] = [], []
            for k, f in enumerate(files):
                idx = np.load(f)
                self.pdat[m].append(np.memmap(f.with_name(f.name.replace("patch_index", "patches").replace(".npy", ".u8")), dtype=np.uint8, mode="r"))
                parts.append(np.concatenate([idx, np.full((len(idx), 1), k, np.int64)], 1))
            self.pidx[m] = np.concatenate(parts)

    def subset(self, m, train):
        idx = self.pidx[m]
        return idx[idx[:, 0] < self.split] if train else idx[idx[:, 0] >= self.split]


class Synthetic(Dataset):
    """Compositions that mimic how the dataset was built, drawn from one or several source banks."""
    def __init__(self, roots, train, length, seed=None, methods=(0, 1, 2), objects=0.0, pure=0.0, shapes=None, shape_share=0.0, fields=0.0):
        self.roots = [r for r in str(roots).split(",") if r]
        self.fields = float(fields)         # share of generated holes that are blob fields (field_mask) instead of compact blobs
        # Official mask shapes the caller is allowed to train on (for a fold fine-tune: its fit pictures only), at
        # 512 px: {"hole": [N,512,512] bool, True where a full-frame picture was restored to clean;
        # "object": [M,512,512] bool, True on an object-shaped watermarked region}. With probability shape_share a
        # composition takes its shape there instead of a COCO instance or a random blob.
        self.shapes, self.shape_share = shapes, float(shape_share)
        self.train, self.length, self.seed = train, length, seed
        self.objects = float(objects)       # share of transplants shaped like a real object (0 = random blobs only)
        # Share of samples built exactly as the starter kit describes the dataset ("the whole image was watermarked and a
        # part of it restored to the clean original"): the unwatermarked part is ALWAYS the same clean picture, mostly
        # with a hard edge, and nothing foreign is pasted. The other samples keep the earlier, more varied fillers.
        self.pure = float(pure)
        self.methods = tuple(methods)
        self.ready = False

    def open(self):
        used = [METHODS[k] for k in self.methods]
        self.banks = [Bank(r, used) for r in self.roots]
        self.rects = [{m: b.subset(m, self.train) for m in used} for b in self.banks]
        self.ready = True

    def __len__(self):
        return self.length

    def shape(self, rng, kind):
        if self.shapes is None or len(self.shapes[kind]) == 0 or rng.random() >= self.shape_share:
            return None
        m = self.shapes[kind][int(rng.integers(len(self.shapes[kind])))]
        return np.ascontiguousarray(m[:, ::-1]) if rng.random() < 0.5 else m.copy()

    def pick(self, rng, bank):
        return int(rng.integers(0, bank.split)) if self.train else int(rng.integers(bank.split, len(bank.clean)))

    def other(self, rng, bank, i):
        j = self.pick(rng, bank)
        return np.asarray(bank.clean[j if j != i else (bank.split - 1 if self.train else len(bank.clean) - 1 - (i == len(bank.clean) - 1))])

    def __getitem__(self, index):
        if not self.ready:
            self.open()
        rng = np.random.default_rng(None if self.seed is None else self.seed + index)
        mi = int(self.methods[rng.integers(len(self.methods))])
        method = METHODS[mi]
        bi = int(rng.integers(len(self.banks)))
        bank = self.banks[bi]
        pure = self.pure > 0 and rng.random() < self.pure
        edges = (0.0, 0.0, 0.0, 0.0, 0.7, 1.5) if pure else (0.0, 0.0, 0.7, 1.5, 2.5)
        u = rng.random()
        if u < 0.25:                                       # rectangle watermarked in its own frame (outpainting)
            rects = self.rects[bi][method]
            i, x0, y0, w, h, off, shard = (int(v) for v in rects[rng.integers(len(rects))])
            # Outpainting: the surroundings of the kept rectangle are regenerated content when available.
            out = np.array(bank.regen[i] if bank.regen is not None and not pure and rng.random() < 0.6 else bank.clean[i])
            out[y0:y0 + h, x0:x0 + w] = bank.pdat[method][shard][off:off + h * w * 3].reshape(h, w, 3)
            label = np.zeros((512, 512), bool)
            label[y0:y0 + h, x0:x0 + w] = True
            mode = 1
        else:
            i = self.pick(rng, bank)
            S, W = np.asarray(bank.clean[i]), np.asarray(bank.full[method][i])
            use_objects = self.objects > 0 and bank.inst is not None
            if u < 0.71:                                   # full-frame watermark with edited holes
                hole, filler = self.shape(rng, "hole"), None
                if hole is None and use_objects:
                    q = rng.random()
                    if q < 0.15:                           # object addition: a foreign object appears in the picture
                        j = self.pick(rng, bank)
                        hole = object_mask(rng, np.asarray(bank.inst[j]), hi=0.6)
                        if hole is not None and not pure:
                            filler = np.asarray(bank.clean[j])
                    elif q < 0.30:                         # an object of the picture is edited in place
                        hole = object_mask(rng, np.asarray(bank.inst[i]), hi=0.6)
                if hole is None:
                    area = sample_area(rng, 0.003, 0.5, 0.8)
                    hole = field_mask(rng, min(area, 0.45)) if rng.random() < self.fields else make_mask(rng, area)
                v = rng.random()
                if pure:
                    filler = S
                elif filler is not None:
                    pass
                elif bank.regen is not None and v < 0.5:
                    filler = np.asarray(bank.regen[i])      # edited region = same picture, re-synthesised
                elif v < 0.7:
                    filler = S
                elif v < 0.9:
                    filler = self.other(rng, bank, i)
                else:
                    filler = cv2.GaussianBlur(S, (0, 0), float(rng.uniform(2, 7)))
                out, label, mode = blend(W, filler, hole, rng, edges), ~hole, 0
            elif u < 0.96:                                 # watermark only on a transplanted object
                obj = self.shape(rng, "object")
                if obj is None and use_objects and rng.random() < self.objects:   # the kept object is a real one, its scene is replaced
                    obj = object_mask(rng, np.asarray(bank.inst[i]), hi=0.75)
                if obj is None:
                    obj = make_mask(rng, sample_area(rng, 0.004, 0.45, 0.75))
                v = rng.random()
                if pure:
                    bg = S
                elif bank.regen is not None and v < 0.4:
                    bg = np.asarray(bank.regen[i])
                elif v < 0.65:
                    bg = S
                else:
                    bg = self.other(rng, bank, i)
                out, label, mode = blend(bg, W, obj, rng, edges), obj, 2
            elif u < 0.98:
                out, label, mode = np.array(W), np.ones((512, 512), bool), 3
            else:
                out, label, mode = np.array(S), np.zeros((512, 512), bool), 4
        if not pure and rng.random() < 0.3:                # unwatermarked paste: an edit without watermark
            d = make_mask(rng, sample_area(rng, 0.003, 0.2, 1.0))
            out = blend(out, self.other(rng, bank, i), d, rng)
            label = label & ~d
        # Regime flag = "the known message decodes in the full frame". At test time it comes from the kit decoder;
        # here it is derived from how the sample was built, with a little noise so the model cannot rely on it blindly.
        flag = int(mode in (0, 3) or (mode == 2 and label.mean() >= 0.35))
        if rng.random() < 0.04:
            flag = 1 - flag
        return torch.from_numpy(np.ascontiguousarray(out)), torch.from_numpy(down128(label)), mi, mode, flag


class Real:
    """Official train images, used for scoring only."""
    def __init__(self, kit, frame_json):
        kit = Path(kit)
        rows = json.loads((kit / "release/train/metadata.json").read_text())
        frame = json.loads(Path(frame_json).read_text())
        self.flag = torch.tensor([int(frame[r["image_name"]]["acc"] >= 0.8) for r in rows])
        self.names = [r["image_name"] for r in rows]
        self.method = torch.tensor([METHODS.index(r["method"]) for r in rows])
        images, masks, groups = [], [], []
        for r in rows:
            images.append(np.array(Image.open(kit / "release/train/images" / (r["image_name"] + ".png")).convert("RGB")))
            m = np.array(Image.open(kit / "release/train/masks" / (r["image_name"] + ".png")).convert("L")) > 127
            ys, xs = np.where(m)
            fill = m.sum() / ((ys.max() + 1 - ys.min()) * (xs.max() + 1 - xs.min()))
            groups.append("full" if r["full_watermark"] else ("rect" if fill > 0.999 else "transplant"))
            masks.append(down128(m))
        self.images = torch.from_numpy(np.stack(images))
        self.masks = np.stack(masks)
        self.groups = groups


# ----------------------------------------------------------------------------- model
class Detector(nn.Module):
    """ConvNeXt-T/FPN detector at 128x128. Optional branches reuse the kit extractors (PixelSeal ConvNeXt,
    TrustMark ResNet-50) at their native full-frame resolution: they already encode the watermark they decode."""
    def __init__(self, backbone="convnext_tiny.in12k_ft_in1k", pretrained=True, dim=160, use_flag=False, kit=None, hr=False):
        super().__init__()
        self.use_flag = use_flag
        # Optional full-resolution branch (steganalysis practice): the backbone's stride-4 patch embedding and the
        # extractors' 256 px input average away the top octave, where a residual that was computed at 256 px and
        # enlarged differs from picture detail. Unpooled 3x3 convolutions at 512 px, reduced to the 128 grid at the end.
        self.use_hr = hr
        if hr:
            c = 32
            self.hr_in = nn.Conv2d(3, c, 3, padding=1)
            self.hr_blocks = nn.ModuleList(nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1)) for _ in range(4))
            self.hr_down = nn.Sequential(nn.GELU(), nn.Conv2d(c, 64, 3, stride=2, padding=1), nn.GELU(), nn.Conv2d(64, dim, 3, stride=2, padding=1), nn.GELU())
            self.hr_mix = nn.Conv2d(dim, dim, 1)
            nn.init.zeros_(self.hr_mix.weight)
            nn.init.zeros_(self.hr_mix.bias)              # starts exactly as the model without the branch
        self.flag = nn.Embedding(2, dim) if use_flag else None
        self.backbone = timm.create_model(backbone, pretrained=pretrained, features_only=True, drop_path_rate=0.1)
        chs = self.backbone.feature_info.channels()
        self.lat = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in chs)
        self.smooth = nn.ModuleList(nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
                                    for _ in chs)
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 48, 3, stride=2, padding=1),
                                  nn.GELU(), nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GELU())
        self.method = nn.Embedding(3, dim)
        self.head = nn.Sequential(nn.Conv2d(dim + 64, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.GELU(), nn.Conv2d(128, 3, 1))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.use_wm = kit is not None
        if self.use_wm:
            sys.path.insert(0, str(kit))
            from wmloc.watermarks import load_watermark
            ps = load_watermark("pixelseal", models_dir=Path(kit) / "models", device="cpu")
            self.ps = ps.det.convnext                       # stages: 96@64, 192@32, 384@16, 768@8 for a 256 input
            tm = load_watermark("trustmark", models_dir=Path(kit) / "models", device="cpu")
            self.tm = tm.net                                # ResNet-50: 256@62, 512@31, 1024@16, 2048@8 for 245
            self.tm_res = int(tm.resolution)
            for q in list(self.ps.parameters()) + list(self.tm.parameters()):
                q.requires_grad_(True)
            self.ps_proj = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in (96, 192, 384, 768))
            self.tm_proj = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in (256, 512, 1024, 2048))
            self.wm_mix = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU(), nn.Conv2d(dim, dim, 1))
            nn.init.zeros_(self.wm_mix[-1].weight)
            nn.init.zeros_(self.wm_mix[-1].bias)            # starts exactly as the plain detector

    def train(self, mode=True):
        super().train(mode)
        if self.use_wm:                                    # released normalisation statistics stay fixed
            self.ps.eval()
            self.tm.eval()
        return self

    def wm_features(self, x, method):
        extra = x.new_zeros(x.shape[0], self.method.embedding_dim, 128, 128, dtype=torch.float32)
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            idx = (method == 0).nonzero(as_tuple=True)[0]
            if len(idx):
                z = F.interpolate(x[idx], size=(256, 256), mode="bilinear", align_corners=False, antialias=True) * 2 - 1
                acc = 0
                for k, (down, stage) in enumerate(zip(self.ps.downsample_layers, self.ps.stages)):
                    z = stage(down(z))
                    acc = acc + F.interpolate(self.ps_proj[k](z), size=(128, 128), mode="bilinear", align_corners=False)
                extra[idx] = self.wm_mix(acc)
            idx = (method == 1).nonzero(as_tuple=True)[0]
            if len(idx):
                z = F.interpolate(x[idx], size=(self.tm_res,) * 2, mode="bilinear", align_corners=False, antialias=True) * 2 - 1
                n = self.tm
                z = n.maxpool(n.relu(n.bn1(n.conv1(z))))
                acc = 0
                for k, name in enumerate(("layer1", "layer2", "layer3", "layer4")):
                    z = getattr(n, name)(z)
                    acc = acc + F.interpolate(self.tm_proj[k](z), size=(128, 128), mode="bilinear", align_corners=False)
                extra[idx] = self.wm_mix(acc)
        return extra

    def forward(self, image_uint8, method, flag=None):
        x = (image_uint8.permute(0, 3, 1, 2).float() / 255).contiguous(memory_format=torch.channels_last)
        feats = self.backbone((x - self.mean) / self.std)
        p = self.smooth[3](self.lat[3](feats[3]))
        for level in (2, 1, 0):
            p = self.lat[level](feats[level]) + F.interpolate(p, size=feats[level].shape[-2:], mode="bilinear", align_corners=False)
            p = self.smooth[level](p)
        p = p + self.method(method)[:, :, None, None]
        if self.use_flag:
            p = p + self.flag(flag)[:, :, None, None]
        if self.use_wm:
            p = p + self.wm_features(x, method).to(p.dtype)
        if self.use_hr:
            h = self.hr_in(x * 4 - 2)
            for block in self.hr_blocks:
                h = h + block(F.gelu(h))
            p = p + self.hr_mix(self.hr_down(h))
        logits = self.head(torch.cat([p, self.stem(x * 2 - 1)], 1))
        return logits.gather(1, method.view(-1, 1, 1, 1).expand(-1, 1, *logits.shape[-2:]))[:, 0]


def build_from_checkpoint(state, kit=None):
    """Rebuild the architecture a checkpoint was trained with and load its weights."""
    cfg = state.get("args", {})
    model = Detector(cfg.get("backbone", "convnext_tiny.in12k_ft_in1k"), pretrained=False, use_flag=bool(cfg.get("flag", False)),
                     kit=kit if cfg.get("wm_branch") else None, hr=bool(cfg.get("hr_branch", False)))
    model.load_state_dict(state["ema"])
    return model


def loss_fn(logits, target):
    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = logits.sigmoid()
    eps = 1.0
    iou1 = ((p * target).sum((1, 2)) + eps) / ((p + target - p * target).sum((1, 2)) + eps)
    q, t0 = 1 - p, 1 - target
    iou0 = ((q * t0).sum((1, 2)) + eps) / ((q + t0 - q * t0).sum((1, 2)) + eps)
    return bce + (1 - (iou0 + iou1) / 2).mean()


def miou_batch(pred, gt):
    """Official per-image mIoU for boolean arrays [N,128,128]."""
    out = np.zeros(len(pred))
    for c in (0, 1):
        p, g = pred == c, gt == c
        union = (p | g).sum((1, 2))
        inter = (p & g).sum((1, 2))
        out += np.where(union == 0, 1.0, inter / np.maximum(union, 1))
    return out / 2


@torch.no_grad()
def predict(model, images, methods, flags, batch=32):
    model.eval()
    probs = []
    for s in range(0, len(images), batch):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(images[s:s + batch].cuda(non_blocking=True), methods[s:s + batch].cuda(), flags[s:s + batch].cuda())
        probs.append(lg.float().sigmoid().cpu())
    return torch.cat(probs).numpy()


def report_real(real, probs, tag):
    res = {}
    for thr in (0.3, 0.5, 0.7):
        scores = miou_batch(probs > thr, real.masks.astype(bool))
        res[f"all@{thr}"] = float(scores.mean())
    scores = miou_batch(probs > 0.5, real.masks.astype(bool))
    meth = real.method.numpy()
    for mi, m in enumerate(METHODS):
        res[m] = float(scores[meth == mi].mean())
        for g in ("full", "rect", "transplant"):
            sel = np.array([(meth[k] == mi) and (real.groups[k] == g) for k in range(len(scores))])
            res[f"{m}/{g}"] = float(scores[sel].mean())
    for g in ("full", "rect", "transplant"):
        sel = np.array([x == g for x in real.groups])
        res[g] = float(scores[sel].mean())
    print("REAL", tag, json.dumps({k: round(v, 4) for k, v in res.items()}), flush=True)
    return res, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=16000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--sources", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--init", default="")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--objects", type=float, default=0.0, help="share of synthetic transplants shaped like real objects")
    ap.add_argument("--pure", type=float, default=0.0, help="share of samples whose unwatermarked part is the same clean picture (kit recipe)")
    ap.add_argument("--fields", type=float, default=0.0, help="share of generated holes that are blob fields (several scattered smooth blobs)")
    ap.add_argument("--flag", action="store_true")
    ap.add_argument("--wm-branch", action="store_true", help="add the kit PixelSeal/TrustMark extractors as trainable branches")
    ap.add_argument("--hr-branch", action="store_true", help="add an unpooled full-resolution convolutional branch")
    ap.add_argument("--methods", default="pixelseal,trustmark,stegastamp")
    ap.add_argument("--backbone", default="convnext_tiny.in12k_ft_in1k")
    ap.add_argument("--frame-json", default=str(Path(__file__).with_name("train_frame_decode.json")))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    # per-method sub-batches change size every step: autotuning each new shape would stall the decoder branches
    torch.backends.cudnn.benchmark = not args.wm_branch
    used = [METHODS.index(m) for m in args.methods.split(",")]
    train = Synthetic(args.layers, True, args.iters * args.batch, methods=used, objects=args.objects, pure=args.pure, fields=args.fields)
    val = Synthetic(args.layers, False, 256, seed=777, methods=used)
    loader = DataLoader(train, batch_size=args.batch, num_workers=args.workers, drop_last=True, pin_memory=True,
                        persistent_workers=True, prefetch_factor=4,
                        worker_init_fn=lambda w: cv2.setNumThreads(1))
    val_batches = [val[k] for k in range(len(val))]
    val_img = torch.stack([b[0] for b in val_batches]); val_mask = torch.stack([b[1] for b in val_batches]).numpy()
    val_meth = torch.tensor([b[2] for b in val_batches]); val_mode = np.array([b[3] for b in val_batches])
    val_flag = torch.tensor([b[4] for b in val_batches])
    real = Real(args.kit, args.frame_json)
    print("DATA", len(train), "real", len(real.names), {g: real.groups.count(g) for g in set(real.groups)}, flush=True)

    kit_arg = args.kit if args.wm_branch else None
    model = Detector(args.backbone, use_flag=args.flag, kit=kit_arg, hr=args.hr_branch).cuda().to(memory_format=torch.channels_last)
    if args.init:
        missing, unexpected = model.load_state_dict(torch.load(args.init, map_location="cuda")["ema"], strict=False)
        assert not unexpected and all(k.startswith(("ps.", "tm.", "ps_proj.", "tm_proj.", "wm_mix.", "hr_")) for k in missing), (missing[:5], unexpected[:5])
        print("INIT", args.init, "new tensors:", len(missing), flush=True)
    ema = Detector(args.backbone, pretrained=False, use_flag=args.flag, kit=kit_arg, hr=args.hr_branch).cuda().to(memory_format=torch.channels_last)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    back = [p for n, p in model.named_parameters() if n.startswith("backbone.")]
    wmp = [p for n, p in model.named_parameters() if n.startswith(("ps.", "tm."))]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("backbone.", "ps.", "tm."))]
    groups = [{"params": back, "lr": args.lr * 0.5}, {"params": rest, "lr": args.lr}]
    if wmp:
        groups.append({"params": wmp, "lr": args.lr * 0.2})
    opt = torch.optim.AdamW(groups, weight_decay=0.02)
    warm = 400
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: min(1.0, (it + 1) / warm) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * min(1.0, it / args.iters)))))
    scaler = torch.amp.GradScaler("cuda")
    t0, run, history = time.time(), 0.0, []
    for it, (image, mask, method, _, flag) in enumerate(loader, 1):
        model.train()
        image, mask, method, flag = (v.cuda(non_blocking=True) for v in (image, mask, method, flag))
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(image, method, flag)
        loss = loss_fn(logits.float(), mask)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(opt); scaler.update(); sched.step()
        decay = min(0.999, (1 + it) / (10 + it))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
        run = loss.item() if it == 1 else 0.98 * run + 0.02 * loss.item()
        if it % 100 == 0:
            print("IT", it, "loss", round(run, 4), "it/s", round(it / (time.time() - t0), 2), flush=True)
        if it % args.eval_every == 0 or it == args.iters:
            vp = predict(ema, val_img, val_meth, val_flag)
            vs = miou_batch(vp > 0.5, val_mask)
            print("SYNVAL", it, round(float(vs.mean()), 4), {int(m): round(float(vs[val_mode == m].mean()), 4) for m in np.unique(val_mode)},
                  {METHODS[k]: round(float(vs[val_meth.numpy() == k].mean()), 4) for k in range(3)}, flush=True)
            probs = predict(ema, real.images, real.method, real.flag)
            res, _ = report_real(real, probs, f"it{it}")
            history.append({"it": it, "loss": run, "synval": float(vs.mean()), **res})
            (out / "history.json").write_text(json.dumps(history, indent=1))
            torch.save({"ema": ema.state_dict(), "it": it, "args": vars(args)}, out / "ckpt_last.pt")
            np.savez_compressed(out / "real_probs_last.npz", names=np.array(real.names), probs=probs.astype(np.float16))
        if it >= args.iters:
            break
    print("TRAIN_DONE", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
