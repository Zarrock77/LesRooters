"""Predict watermark probability maps [N,128,128] for the official train and test images with one checkpoint."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_det import METHODS, build_from_checkpoint  # noqa: E402


def load_split(kit, split):
    rows = json.loads((kit / "release" / split / "metadata.json").read_text())
    images = np.stack([np.array(Image.open(kit / "release" / split / "images" / (r["image_name"] + ".png")).convert("RGB"))
                       for r in rows])
    return rows, torch.from_numpy(images)


@torch.no_grad()
def run(model, images, methods, flags, batch=32):
    out = []
    for s in range(0, len(images), batch):
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(images[s:s + batch].cuda(), methods[s:s + batch].cuda(), flags[s:s + batch].cuda())
        out.append(lg.float().sigmoid().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--frame-json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", default="train,test")
    args = ap.parse_args()
    state = torch.load(args.ckpt, map_location="cuda")
    model = build_from_checkpoint(state, args.kit).cuda().eval().to(memory_format=torch.channels_last)
    frame = json.loads(args.frame_json.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for split in args.splits.split(","):
        rows, images = load_split(args.kit, split)
        methods = torch.tensor([METHODS.index(r["method"]) for r in rows])
        flags = torch.tensor([int(frame[split][r["image_name"]]["acc"] >= 0.8) for r in rows])
        probs = run(model, images, methods, flags)
        np.savez_compressed(args.out / f"{split}_probs.npz", names=np.array([r["image_name"] for r in rows]), probs=probs)
        print("PREDICTED", split, probs.shape, flush=True)


if __name__ == "__main__":
    main()
