"""Pre-embed watermark layers from a clean source bank with the kit's exact embedders (sharded for speed).

For every clean 512x512 source and one method:
  full.npy               the whole image watermarked in the full frame with a random message
  patches_<k>.u8         R rectangles of the same source, each watermarked in its own frame (like the dataset's)
  patch_index_<k>.npy    (source, x0, y0, w, h, byte offset) per rectangle
  messages_<k>.npz       the embedded bits
Shard k handles sources i with i % shards == k; all shards write disjoint rows of the same full.npy.
No official image, mask or message is read.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(3)
ap = argparse.ArgumentParser()
ap.add_argument("--kit", required=True, type=Path)
ap.add_argument("--data", required=True, type=Path)
ap.add_argument("--method", required=True)
ap.add_argument("--rects", type=int, default=4)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--shards", type=int, default=1)
ap.add_argument("--init", action="store_true", help="only create the shared full.npy and exit")
args = ap.parse_args()
method = args.method
out = args.data / method
out.mkdir(parents=True, exist_ok=True)
clean = np.load(args.data / "clean.npy", mmap_mode="r")
N = len(clean)
if args.init:
    np.lib.format.open_memmap(out / "full.npy", mode="w+", dtype=np.uint8, shape=(N, 512, 512, 3)).flush()
    print("INIT", method, N, flush=True)
    raise SystemExit(0)

sys.path.insert(0, str(args.kit))
rng = np.random.default_rng([20261121, {"pixelseal": 11, "trustmark": 22, "stegastamp": 33}[method], args.shard, N])
if method == "stegastamp":
    import onnxruntime as ort
    ort.preload_dlls()
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark(method, models_dir=args.kit / "models", device=args.device)
if method == "stegastamp":
    assert "CUDAExecutionProvider" in wm.sess.get_providers()
nbits = wm.nbits
max_aspect = 2.0 if method == "trustmark" else 16.0


def sample_rect():
    """Area and aspect distribution fitted to the train rectangles (median area about 2 % of the image)."""
    for _ in range(1000):
        area = 0.003 * (0.45 / 0.003) ** (rng.random() ** 1.3) * 512 * 512
        aspect = np.exp(rng.uniform(-np.log(max_aspect), np.log(max_aspect)))
        w, h = int(round(np.sqrt(area * aspect))), int(round(np.sqrt(area / aspect)))
        if 8 <= w <= 512 and 8 <= h <= 512 and max(w / h, h / w) <= max_aspect:
            return int(rng.integers(0, 512 - w + 1)), int(rng.integers(0, 512 - h + 1)), w, h
    raise RuntimeError("no rectangle")


full = np.load(out / "full.npy", mmap_mode="r+")
mine = list(range(args.shard, N, args.shards))
full_bits, index, patch_bits, offset = [], [], [], 0
t0 = time.time()
with open(out / f"patches_{args.shard}.u8", "wb") as stream:
    for count, i in enumerate(mine):
        S = np.array(clean[i])
        bits = rng.integers(0, 2, nbits)
        full_bits.append(bits.astype(np.uint8))
        full[i] = wm.embed(S, bits.tolist())
        for _ in range(args.rects):
            x0, y0, w, h = sample_rect()
            crop = np.ascontiguousarray(S[y0:y0 + h, x0:x0 + w])
            bits = rng.integers(0, 2, nbits)
            p = np.asarray(wm.embed(crop, bits.tolist()), dtype=np.uint8)
            assert p.shape == crop.shape, (p.shape, crop.shape)
            stream.write(p.tobytes())
            index.append((i, x0, y0, w, h, offset))
            patch_bits.append(bits.astype(np.uint8))
            offset += p.size
        if count % 250 == 0:
            print("GEN", method, args.shard, count, "/", len(mine), round(time.time() - t0, 1), "s", flush=True)
full.flush()
np.save(out / f"patch_index_{args.shard}.npy", np.array(index, np.int64))
np.savez_compressed(out / f"messages_{args.shard}.npz", sources=np.array(mine), full=np.stack(full_bits), patches=np.stack(patch_bits))
(out / f"DONE_{args.shard}").write_text(json.dumps({"sources": len(mine), "patches": len(index), "seconds": time.time() - t0}))
print("GEN_DONE", method, args.shard, len(mine), len(index), round(time.time() - t0, 1), "s", flush=True)
