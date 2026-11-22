"""Snap a detector blob to an exact rectangle by maximising the kit decoder's agreement with the known message.

For images without a full-frame watermark, the dataset's rectangles were watermarked in their own frame, so the
decoder reads the message only when the crop matches the rectangle. Starting from the detector's blob, a local
coordinate search over (x0, y0, x1, y1) maximises the signed-logit margin. Nothing is trained here.
Output json: {image_name: {"box": [x0,y0,x1,y1], "margin": float, "acc": float, "init": [...], "init_margin": float}}
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision.ops import roi_align

METHODS = ("pixelseal", "trustmark", "stegastamp")
RES = {"pixelseal": 256, "trustmark": 245, "stegastamp": 400}
MAX_ASPECT = {"pixelseal": 16.5, "trustmark": 2.1, "stegastamp": 16.5}


class Scorer:
    def __init__(self, kit, method, device="cuda:0"):
        sys.path.insert(0, str(kit))
        if method == "stegastamp":
            import onnxruntime as ort
            ort.preload_dlls()
        from wmloc.watermarks import load_watermark
        self.method, self.device = method, device
        self.wm = load_watermark(method, models_dir=Path(kit) / "models", device=device)

    @torch.no_grad()
    def __call__(self, image, boxes, sign):
        """image: 1x3x512x512 float on device; boxes: K x 4 float (pixel-edge coords); sign: nbits (+-1)."""
        rois = torch.cat([torch.zeros(len(boxes), 1, device=self.device), boxes], 1)
        crops = roi_align(image, rois, output_size=RES[self.method], spatial_scale=1.0, sampling_ratio=-1, aligned=True)
        if self.method == "trustmark":
            with torch.autocast("cuda", dtype=torch.float16):
                lg = self.wm.net(crops * 2 - 1)
        elif self.method == "pixelseal":
            lg = self.wm.det(crops)[:, 1:]          # float32: half precision overflows in this extractor
        else:
            x = crops.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
            lg = torch.from_numpy(np.concatenate([self.wm._run(x[i:i + 16], [self.wm.LOGITS])[0] for i in range(0, len(x), 16)])).to(self.device)
        lg = lg.float()
        return (lg * sign).mean(1), ((lg > 0) == (sign > 0)).float().mean(1)


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


SCAN_ASPECTS = {"trustmark": (0.5, 0.63, 0.79, 1.0, 1.26, 1.59, 2.0),
                "pixelseal": (1 / 16, 1 / 11.3, 1 / 8, 1 / 5.66, 1 / 4, 1 / 2.83, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0, 5.66, 8.0, 11.3, 16.0)}
SCAN_MIN = {"trustmark": 40.0, "pixelseal": 70.0}       # smaller rectangles are not decodable anyway


@torch.no_grad()
def global_scan(scorer, image, sign, top=6):
    """Coarse search over the whole image: the decoder trunk is run fully convolutionally on the image rescaled so
    that a rectangle of the hypothesised size fills the decoder's field; averaging its per-position message
    agreement over one field gives the agreement of every window at once. Returns the best distinct boxes."""
    method, wm, dev = scorer.method, scorer.wm, scorer.device
    res = RES[method]
    cells = -(-res // 32)                                   # positions covered by one decoder field (8)
    found = []
    size = SCAN_MIN[method]
    while size <= 512.0:
        for aspect in SCAN_ASPECTS[method]:
            w, h = size * aspect ** 0.5, size / aspect ** 0.5
            if w > 512 or h > 512 or w < 8 or h < 8:
                continue
            H, W = int(round(512 * res / h)), int(round(512 * res / w))
            if H * W > 14e6 or min(H, W) < res:
                continue
            x = F.interpolate(image, size=(H, W), mode="bilinear", align_corners=False, antialias=True)
            if method == "trustmark":
                n = wm.net
                with torch.autocast("cuda", dtype=torch.float16):
                    z = n.maxpool(n.relu(n.bn1(n.conv1(x * 2 - 1))))
                    z = n.layer4(n.layer3(n.layer2(n.layer1(z))))
                z = z.float()
                v = (n.fc.weight.float() * sign[:, None]).mean(0)
                mm = (z * v[None, :, None, None]).sum(1, keepdim=True) + (n.fc.bias.float() * sign).mean()
                up = 1
            else:
                det = wm.det
                emb = det.pixel_decoder.output_upscaling(det.convnext(det.preprocess(x))).float()
                lin = det.pixel_decoder.linear
                v = (lin.weight[1:].float() * sign[:, None]).mean(0)
                mm = (emb * v[None, :, None, None]).sum(1, keepdim=True) + (lin.bias[1:].float() * sign).mean()
                up = max(1, round(emb.shape[-1] / (W / 32)))
            if mm.shape[-2] < cells * up or mm.shape[-1] < cells * up:
                continue
            win = F.avg_pool2d(mm, kernel_size=cells * up, stride=up)[0, 0]
            value, index = win.flatten().max(0)
            i, j = divmod(int(index), win.shape[1])
            x0, y0 = j * 32 * w / res, i * 32 * h / res
            found.append((float(value), (x0, y0, min(512.0, x0 + w), min(512.0, y0 + h))))
        size *= 1.25
    found.sort(key=lambda t: -t[0])
    boxes = []
    for value, box in found:
        if all(box_iou(box, b) < 0.4 for b in boxes):
            boxes.append(box)
        if len(boxes) >= top:
            break
    return boxes


def clip_box(b, max_aspect):
    x0, y0, x1, y1 = b
    x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(512.0, x1), min(512.0, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    w, h = x1 - x0, y1 - y0
    if max(w / h, h / w) > max_aspect:
        return None
    return (x0, y0, x1, y1)


def moves(b, s):
    x0, y0, x1, y1 = b
    return [(x0 - s, y0, x1, y1), (x0 + s, y0, x1, y1), (x0, y0 - s, x1, y1), (x0, y0 + s, x1, y1),
            (x0, y0, x1 - s, y1), (x0, y0, x1 + s, y1), (x0, y0, x1, y1 - s), (x0, y0, x1, y1 + s),
            (x0 - s, y0, x1 - s, y1), (x0 + s, y0, x1 + s, y1), (x0, y0 - s, x1, y1 - s), (x0, y0 + s, x1, y1 + s),
            (x0 - s, y0 - s, x1 + s, y1 + s), (x0 + s, y0 + s, x1 - s, y1 - s),
            (x0 - s, y0 - s, x1, y1), (x0, y0, x1 + s, y1 + s), (x0 + s, y0 + s, x1, y1), (x0, y0, x1 - s, y1 - s)]


def refine(scorer, image, sign, box, max_aspect):
    box = clip_box(box, max_aspect)
    if box is None:
        return None
    dev = scorer.device
    m, a = scorer(image, torch.tensor([box], device=dev, dtype=torch.float32), sign)
    best, best_m, best_a, init_m = box, float(m[0]), float(a[0]), float(m[0])
    for frac in (0.25, 0.15, 0.08, 0.04, 0.02, 0.01, 0.0):
        size = min(best[2] - best[0], best[3] - best[1])
        s = max(1.0, round(frac * size))
        for _ in range(3):
            cands = [c for c in (clip_box(c, max_aspect) for c in moves(best, s)) if c is not None]
            if not cands:
                break
            m, a = scorer(image, torch.tensor(cands, device=dev, dtype=torch.float32), sign)
            k = int(m.argmax())
            if float(m[k]) <= best_m + 1e-4:
                break
            best, best_m, best_a = cands[k], float(m[k]), float(a[k])
    return {"box": [float(v) for v in best], "margin": best_m, "acc": best_a, "init_margin": init_m}


def proposals(prob):
    """Candidate boxes (512-px coords) from a 128x128 probability map."""
    out = []
    for thr in (0.5, 0.3, 0.7):
        lab, n = ndimage.label(prob > thr)
        if n == 0:
            continue
        sizes = ndimage.sum(prob > thr, lab, range(1, n + 1))
        for k in np.argsort(-sizes)[:2]:
            ys, xs = np.where(lab == k + 1)
            if len(ys) < 3:
                continue
            out.append((xs.min() * 4.0, ys.min() * 4.0, (xs.max() + 1) * 4.0, (ys.max() + 1) * 4.0))
        ys, xs = np.where(prob > thr)
        out.append((xs.min() * 4.0, ys.min() * 4.0, (xs.max() + 1) * 4.0, (ys.max() + 1) * 4.0))
    seen, uniq = set(), []
    for b in out:
        if b not in seen and b != (0.0, 0.0, 512.0, 512.0):
            seen.add(b); uniq.append(b)
    return uniq[:3]


def map_hash(p):
    return hashlib.sha1(np.ascontiguousarray(p, dtype=np.float32).tobytes()).hexdigest()[:20]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True, type=Path)
    ap.add_argument("--split", required=True)
    ap.add_argument("--probs", required=True, type=Path, help="npz with names + probs [N,128,128]")
    ap.add_argument("--frame-json", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prior", type=Path, nargs="*", default=[], help="earlier results: confident rectangles are reused, not searched again")
    ap.add_argument("--methods", default="pixelseal,trustmark,stegastamp")
    ap.add_argument("--global-scan", action="store_true", help="search the whole image when the detector blob gives no decodable rectangle")
    args = ap.parse_args()
    rows = json.loads((args.kit / "release" / args.split / "metadata.json").read_text())
    frame = json.loads(args.frame_json.read_text())[args.split]
    with np.load(args.probs) as z:
        probs = dict(zip(z["names"].tolist(), z["probs"].astype(np.float32)))
    result, t0 = {}, time.time()
    prior = {}
    # Exact reuse: an earlier search on the byte-identical probability map of a picture gives the same proposals, so
    # its outcome (a rectangle or none) is taken over without searching again. Each result file has a side file
    # <out>.hash.json with the hash of every map it was computed from. Typical case: a method whose maps did not
    # change between two combinations (StegaStamp took five seconds per picture on a busy GPU for nothing).
    hashes = {name: map_hash(p) for name, p in probs.items()}
    same = {}
    for f in args.prior:
        if f.exists():
            earlier = json.loads(f.read_text())
            for name, v in earlier.items():
                if v and (name not in prior or v["margin"] > prior[name]["margin"]):
                    prior[name] = v
            side = f.with_suffix(".hash.json")
            if side.exists():
                for name, h in json.loads(side.read_text()).items():
                    if hashes.get(name) == h and name in earlier and name not in same:
                        same[name] = earlier[name]
    reused = 0
    for method in args.methods.split(","):
        scorer = Scorer(args.kit, method, args.device)
        sub = [r for r in rows if r["method"] == method and frame[r["image_name"]]["acc"] < 0.8]
        for n, r in enumerate(sub):
            name = r["image_name"]
            if name in prior and prior[name]["margin"] >= 3.0:      # the image (not the detector) defines this rectangle
                result[name] = prior[name]
                continue
            if name in same:                                       # same map searched before: same outcome
                best = same[name]
                if name in prior and (best is None or prior[name]["margin"] > best["margin"]):
                    best = prior[name]
                result[name] = best
                reused += 1
                continue
            image = torch.from_numpy(np.array(Image.open(args.kit / "release" / args.split / "images" / (name + ".png")).convert("RGB")))
            image = image.permute(2, 0, 1)[None].float().div(255).to(args.device)
            sign = torch.tensor([int(b) for b in r["message"]], device=args.device, dtype=torch.float32) * 2 - 1
            best = None
            for box in proposals(probs[name]):
                res = refine(scorer, image, sign, box, MAX_ASPECT[method])
                if res is not None:
                    res["init"] = [float(v) for v in box]
                    if best is None or res["margin"] > best["margin"]:
                        best = res
            if args.global_scan and method in SCAN_ASPECTS and (best is None or best["margin"] < 3.0):
                for box in global_scan(scorer, image, sign):
                    res = refine(scorer, image, sign, box, MAX_ASPECT[method])
                    if res is not None:
                        res["init"] = [float(v) for v in box]
                        res["scan"] = True
                        if best is None or res["margin"] > best["margin"]:
                            best = res
            if name in prior and (best is None or prior[name]["margin"] > best["margin"]):
                best = prior[name]
            result[name] = best
            if n % 10 == 0:
                print("RECT", args.split, method, n, "/", len(sub), round(time.time() - t0, 1), "s", flush=True)
        del scorer
        torch.cuda.empty_cache()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result))
    args.out.with_suffix(".hash.json").write_text(json.dumps({name: hashes[name] for name in result}))
    print("RECT reused from identical maps:", reused, flush=True)
    found = [v for v in result.values() if v]
    print("RECT_DONE", args.split, len(result), "with proposal", len(found), "margin>1:", sum(v["margin"] > 1 for v in found), flush=True)


if __name__ == "__main__":
    main()
