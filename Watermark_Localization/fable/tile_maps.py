"""Message-aware presence maps from the tile decoder (tile_decoder.py), for every PixelSeal picture.

Every tile is cut out, resized to the extractor's resolution and decoded on its own; its agreement with the known
message is spread over the cells it covers and averaged. Three tile sizes. No mask label is read. Output in the
format the cross-fitted fusion expects: <out>/<split>_pixelseal.npz with names and maps [N,4,128,128].
usage: tile_maps.py <kit> <tile_decoder.pt> <out dir> [stride]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import load_split  # noqa: E402

kit, weights, out_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
stride = int(sys.argv[4]) if len(sys.argv) > 4 else 32
sys.path.insert(0, str(kit))
from wmloc.watermarks import load_watermark  # noqa: E402

wm = load_watermark("pixelseal", models_dir=kit / "models", device="cuda")
det = wm.det
det.load_state_dict(torch.load(weights, map_location="cuda"))
det.eval()
RES, SIZES = 256, (224, 192, 160)
out_dir.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def maps_for(x, sign):
    """x [1,3,512,512] in [0,1]; sign [256] of +-1 -> [4,128,128]: mean signed logit per tile size, and the
    z-score of the bit agreement of the 192 px tiles."""
    out = []
    zmap = None
    for side in SIZES:
        pos = list(range(0, 513 - side, stride))
        boxes = [(x0, y0) for y0 in pos for x0 in pos]
        tiles = torch.cat([F.interpolate(x[:, :, y0:y0 + side, x0:x0 + side], size=(RES, RES), mode="bilinear", align_corners=False, antialias=True)
                           for x0, y0 in boxes])
        lg = torch.cat([det(tiles[k:k + 128])[:, 1:].float() for k in range(0, len(tiles), 128)])
        margin = (lg * sign).mean(1)
        agree = (((lg > 0) == (sign > 0)).float().mean(1) - 0.5) / 0.03125          # z-score of 256 fair coins
        acc, cnt, accz = (torch.zeros(128, 128, device=x.device) for _ in range(3))
        c = side // 4
        for (x0, y0), m, a in zip(boxes, margin, agree):
            acc[y0 // 4:y0 // 4 + c, x0 // 4:x0 // 4 + c] += m
            accz[y0 // 4:y0 // 4 + c, x0 // 4:x0 // 4 + c] += a
            cnt[y0 // 4:y0 // 4 + c, x0 // 4:x0 // 4 + c] += 1
        out.append(acc / cnt.clamp(min=1))
        if side == 192:
            zmap = accz / cnt.clamp(min=1) / 4
    return torch.stack(out + [zmap]).cpu().numpy().astype(np.float16)


for split in ("train", "test"):
    rows, images = load_split(kit, split)
    names, maps = [], []
    for i, r in enumerate(rows):
        if r["method"] != "pixelseal":
            continue
        x = images[i:i + 1].cuda().permute(0, 3, 1, 2).float() / 255
        sign = torch.tensor([1.0 if c == "1" else -1.0 for c in r["message"]], device="cuda")
        names.append(r["image_name"]); maps.append(maps_for(x, sign))
        if len(names) % 50 == 0:
            print("TILEMAPS", split, len(names), flush=True)
    np.savez_compressed(out_dir / f"{split}_pixelseal.npz", names=np.array(names), maps=np.stack(maps))
    print("TILEMAPS_DONE", split, len(names), flush=True)
