"""Teach the PixelSeal extractor to read the known message from ONE small tile.

The released extractor needs a large part of the frame (tile_probe.py: chance below 190 px), so it cannot say
where the watermark is missing. Here it is fine-tuned, with a curriculum on the tile size, to decode the message
from a single tile that is cut out and resized to its working resolution: nothing outside the tile reaches the
network, so a tile inside a hole has nothing to read. Sliding the tile gives a message-aware presence map.

Training data, no mask label anywhere:
  - fresh kit embeddings of clean COCO pictures with random messages;
  - official pictures in which the message decodes in the full frame (train and test), with their known message
    (tiles that fall into an edited hole are a few per cent of label noise).
Scoring only: official train full-frame pictures, tiles entirely inside the kept region vs entirely inside a hole.
usage: tile_decoder.py <kit> <clean.npy> <frame json> <out dir> [iters] [official share] [stop at step]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, clean_path, frame_json, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
iters = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
official_share = float(sys.argv[6]) if len(sys.argv) > 6 else 0.5
# Stop early: a first full run showed the extractor starting to memorise the official pictures after about 1000
# steps (tiles inside holes of train pictures began to "decode": 0.50 at step 1000, 0.53 at step 1500).
stop = int(sys.argv[7]) if len(sys.argv) > 7 else iters
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

out_dir.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
torch.manual_seed(5)
rng = np.random.default_rng(5)
wm = load_watermark("pixelseal", models_dir=kit / "models", device="cuda")
det = wm.det
for p in det.parameters():
    p.requires_grad_(True)
RES, B = 256, 48
# Tile side in the 512 px picture. A first run that went down to 96 px showed where the information stops on official
# pictures: 192 px tiles reach bit accuracy 0.68 (kept) against 0.51 (hole), 128 px tiles only 0.53, 96 px nothing,
# and training on the small sizes made the large ones worse. So: warm up on large tiles, then stay on 224-128.
WARMUP = (256, 224, 192)
SIZES, WEIGHTS = (224, 192, 160, 128), (0.2, 0.4, 0.3, 0.1)
EVAL_SIZES = (192, 160, 128)


def tiles_from(x, boxes):
    """x [N,3,512,512] in [0,1]; boxes [(index, x0, y0, side)] -> tiles resized to the extractor resolution."""
    out = [F.interpolate(x[i:i + 1, :, y0:y0 + s, x0:x0 + s], size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
           for i, x0, y0, s in boxes]
    return torch.cat(out)


def logits(t):
    return det(t)[:, 1:].float()


clean = np.load(clean_path, mmap_mode="r")
frame = json.loads(frame_json.read_text())
pool_x, pool_bits = [], []
for split in ("train", "test"):
    rows, images = load_split(kit, split)
    keep = [i for i, r in enumerate(rows) if r["method"] == "pixelseal" and frame[split][r["image_name"]]["acc"] >= 0.8]
    pool_x.append(images[keep]); pool_bits += [[int(c) for c in rows[i]["message"]] for i in keep]
    if split == "train":
        train_rows, train_keep = rows, keep
pool_x = torch.cat(pool_x)                               # uint8 [M,512,512,3]
pool_bits = torch.tensor(pool_bits)
print("TILEDEC official full-frame pictures (train + test, messages only):", len(pool_x), flush=True)
with np.load(kit / "release/train/masks_128.npz") as z:
    masks = dict(zip(z["names"].tolist(), z["masks"].astype(bool)))
eval_x = pool_x[:len(train_keep)]
eval_bits = pool_bits[:len(train_keep)]
eval_masks = np.stack([np.kron(masks[train_rows[i]["image_name"]], np.ones((4, 4), bool)) for i in train_keep])


def batch(step):
    warm = iters * 0.3
    side = WARMUP[min(len(WARMUP) - 1, int(step / warm * len(WARMUP)))] if step < warm else int(rng.choice(SIZES, p=WEIGHTS))
    n_off = int(round(B * official_share))
    idx = rng.integers(0, len(clean), B - n_off)
    x = torch.from_numpy(np.stack([clean[int(i)] for i in idx])).cuda().permute(0, 3, 1, 2).float() / 255
    bits = torch.from_numpy(rng.integers(0, 2, (B - n_off, 256))).cuda()
    with torch.no_grad():
        x = (wm.model.embed(x, msgs=bits.float(), is_video=False)["imgs_w"].clamp(0, 1) * 255).round() / 255
    if n_off:
        j = rng.integers(0, len(pool_x), n_off)
        x = torch.cat([x, pool_x[j].cuda().permute(0, 3, 1, 2).float() / 255])
        bits = torch.cat([bits, pool_bits[j].cuda()])
    boxes = [(i, int(rng.integers(0, 513 - side)), int(rng.integers(0, 513 - side)), side) for i in range(B)]
    return tiles_from(x, boxes), bits.float(), side


def auc(s, label):
    s, label = np.asarray(s), np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


@torch.no_grad()
def evaluate(step):
    det.eval()
    report = {"it": step}
    for side in EVAL_SIZES:
        margins, accs, frac = [], [], []
        stride = side // 2
        for i in range(len(eval_x)):
            x = eval_x[i:i + 1].cuda().permute(0, 3, 1, 2).float() / 255
            boxes = [(0, x0, y0, side) for y0 in range(0, 513 - side, stride) for x0 in range(0, 513 - side, stride)]
            lg = torch.cat([logits(tiles_from(x, boxes[k:k + 64])) for k in range(0, len(boxes), 64)])
            sign = eval_bits[i].cuda().float() * 2 - 1
            margins.append((lg * sign).mean(1).cpu().numpy()); accs.append(((lg > 0) == (sign > 0)).float().mean(1).cpu().numpy())
            frac.append(np.array([eval_masks[i][y0:y0 + side, x0:x0 + side].mean() for _, x0, y0, _ in boxes]))
        margins, accs, frac = np.concatenate(margins), np.concatenate(accs), np.concatenate(frac)
        kept, hole = frac >= 0.95, frac <= 0.15
        sel = kept | hole
        report[f"tile{side}"] = {"acc_kept": round(float(accs[kept].mean()), 4), "acc_hole": round(float(accs[hole].mean()), 4) if hole.any() else None,
                                 "n_hole": int(hole.sum()), "auc": round(auc(margins[sel], kept[sel]), 4)}
    print("EVAL", json.dumps(report), flush=True)
    det.train()


opt = torch.optim.AdamW(det.parameters(), lr=5e-5, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=5e-5, total_steps=iters, pct_start=0.05)
evaluate(0)
det.train()
t0 = time.time()
for step in range(1, iters + 1):
    t, bits, side = batch(step)
    lg = logits(t)
    loss = F.binary_cross_entropy_with_logits(lg, bits)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(det.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 50 == 0:
        acc = float(((lg > 0) == (bits > 0.5)).float().mean())
        print("IT", step, "tile", side, "loss", round(float(loss), 4), "bit acc", round(acc, 4), "it/s", round(step / (time.time() - t0), 2), flush=True)
    if step % 500 == 0 or step == stop:
        evaluate(step)
        torch.save(det.state_dict(), out_dir / "tile_decoder.pt")
    if step == stop:
        break
print("TILEDEC_DONE", flush=True)
