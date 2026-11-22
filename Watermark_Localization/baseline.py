"""Inspect the official release and build a leakage-aware first localization baseline.

Run with the starter kit's environment, on a Slurm compute node. No leaderboard
request is made. Full-watermark flags are used for stratification only: they are
not available at test time and must never be prediction inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.fft import dctn
import torch
import torch.nn.functional as F

METHODS = ("pixelseal", "trustmark", "stegastamp")
FEATURE_VERSION = "message-agreement-v1-full-spatial-or-256-window-stride128"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_image(root, name):
    with Image.open(root / "train" / "images" / f"{name}.png") as image:
        return np.array(image.convert("RGB"))


def font(size=16):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def contact_sheet(rows, root, masks, path, predictions=None, score_values=None):
    """Each row shows source, ground-truth overlay, and optionally prediction/map."""
    columns = 4 if predictions is not None else 2
    tile, header, row_header = 256, 40, 44
    sheet = Image.new("RGB", (columns * tile, header + len(rows) * (tile + row_header)), "white")
    draw = ImageDraw.Draw(sheet)
    labels = ["Image", "Verite terrain (vert)", "Prediction (magenta)", "Accord au message"]
    for j, label in enumerate(labels[:columns]):
        draw.text((j * tile + 8, 10), label, font=font(), fill="black")
    for i, row in enumerate(rows):
        name = row["image_name"]
        image = load_image(root, name)
        gt = np.array(Image.fromarray(masks[name] * 255).resize((512, 512), Image.Resampling.NEAREST)) > 0
        overlay = image.copy()
        overlay[gt] = (0.55 * image[gt] + 0.45 * np.array([0, 255, 80])).astype(np.uint8)
        y = header + i * (tile + row_header)
        title = f"{row['method']} | {name} | zone {100*masks[name].mean():.1f}%"
        if predictions is not None:
            from wmloc.utils import miou_image
            title += f" | mIoU {miou_image(predictions[name], masks[name]):.3f}"
        draw.text((8, y + 5), title, font=font(14), fill="black")
        panels = [Image.fromarray(image), Image.fromarray(overlay)]
        if predictions is not None:
            p = np.array(Image.fromarray(predictions[name] * 255).resize((512, 512), Image.Resampling.NEAREST)) > 0
            pred_overlay = image.copy()
            pred_overlay[p] = (0.55 * image[p] + 0.45 * np.array([255, 0, 200])).astype(np.uint8)
            values = score_values[name].clip(0, 1)
            heat = np.stack([values, np.zeros_like(values), 1 - values], -1)
            panels.extend([Image.fromarray(pred_overlay), Image.fromarray((heat * 255).astype(np.uint8))])
        for j, panel in enumerate(panels):
            sheet.paste(panel.resize((tile, tile), Image.Resampling.BILINEAR), (j * tile, y + row_header))
    sheet.save(path)


def inspect(args):
    from wmloc.utils import downsample_mask, miou_image
    root, out = args.data_root, args.output_root
    rows = json.loads((root / "train" / "metadata.json").read_text())
    test_rows = json.loads((root / "test" / "metadata.json").read_text())
    names = [r["image_name"] for r in rows]
    assert len(names) == len(set(names)), "Duplicate train names"
    assert not set(names) & {r["image_name"] for r in test_rows}, "Train/test name overlap"
    with np.load(root / "train" / "masks_128.npz") as z:
        masks = dict(zip(z["names"].tolist(), z["masks"].astype(np.uint8)))
    assert set(names) == set(masks)

    # Cluster identical and perceptually very similar images before assigning a split.
    # The metadata contains no source-image ID, so this cannot prove source independence.
    hashes, phashes, areas = [], [], []
    for row in rows:
        name = row["image_name"]
        image = load_image(root, name)
        assert image.shape == (512, 512, 3), (name, image.shape)
        hashes.append(digest(image.tobytes()))
        gray = np.asarray(Image.fromarray(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=float)
        low = dctn(gray, norm="ortho")[:8, :8].ravel()[1:]
        bits = low > np.median(low)
        phashes.append(sum(int(bit) << j for j, bit in enumerate(bits)))
        with Image.open(root / "train" / "masks" / f"{name}.png") as mask_image:
            original = (np.array(mask_image) > 127).astype(np.uint8)
        assert np.array_equal(downsample_mask(original), masks[name]), f"Mask mismatch: {name}"
        assert masks[name].shape == (128, 128) and np.isin(masks[name], [0, 1]).all()
        assert miou_image(masks[name], masks[name]) == 1.0
        areas.append(float(masks[name].mean()))
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    exact_pairs, near_pairs = 0, 0
    for i in range(len(rows)):
        for j in range(i):
            exact = hashes[i] == hashes[j]
            near = (phashes[i] ^ phashes[j]).bit_count() <= 4
            if exact or near:
                exact_pairs += int(exact)
                near_pairs += int(near and not exact)
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    strata = sorted({(r["method"], bool(r["full_watermark"])) for r in rows})
    stratum_ids = {s: i for i, s in enumerate(strata)}
    row_strata = np.array([stratum_ids[(r["method"], bool(r["full_watermark"]))] for r in rows])
    counts = np.bincount(row_strata, minlength=len(strata))
    targets = np.rint(counts * 0.2).astype(int)
    remaining = targets.copy()
    ordered = list(groups.values())
    random.Random(args.seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)  # Allocate large groups first, fill gaps with singletons.
    validation = set()
    for indices in ordered:
        count = np.bincount(row_strata[indices], minlength=len(strata))
        if np.all(count <= remaining):
            validation.update(indices)
            remaining -= count
    assert len(validation) >= 0.15 * len(rows), "Grouped validation unexpectedly small"
    assert np.all(targets - remaining > 0), "Missing validation stratum"
    train_names = sorted(names[i] for i in range(len(rows)) if i not in validation)
    val_names = sorted(names[i] for i in validation)
    assert set(train_names).isdisjoint(val_names)
    split = {
        "seed": args.seed, "metadata_sha256": digest((root / "train" / "metadata.json").read_bytes()),
        "train": train_names, "validation": val_names,
        "groups": [[names[i] for i in group] for group in groups.values()],
        "grouping": "SHA256 of decoded RGB or 63-bit DCT pHash Hamming distance <= 4; transitive groups",
        "limitation": "No source-image identifiers supplied; similar source images may escape this heuristic.",
    }
    split_path = out / "split.json"
    if split_path.exists():
        assert json.loads(split_path.read_text()) == split, "Existing frozen split differs; use a new output directory"
    else:
        write_json(split_path, split)
    summary = {
        "train_total": len(rows), "test_total": len(test_rows),
        "calibration_images": len(train_names), "validation_images": len(val_names),
        "train_methods": dict(Counter(r["method"] for r in rows)),
        "test_methods": dict(Counter(r["method"] for r in test_rows)),
        "metadata_fields_train": sorted(rows[0]), "metadata_fields_test": sorted(test_rows[0]),
        "exact_duplicate_pairs": exact_pairs, "near_duplicate_pairs": near_pairs,
        "groups": len(groups), "mask_downsampling_matches": len(rows),
        "strata": [{"method": s[0], "full_watermark": s[1], "total": int(counts[i]),
                    "train": int(counts[i] - targets[i] + remaining[i]),
                    "validation": int(targets[i] - remaining[i])} for i, s in enumerate(strata)],
        "watermarked_area_by_method": {m: {"min": float(np.min([areas[i] for i,r in enumerate(rows) if r['method']==m])),
                                              "median": float(np.median([areas[i] for i,r in enumerate(rows) if r['method']==m])),
                                              "max": float(np.max([areas[i] for i,r in enumerate(rows) if r['method']==m]))} for m in METHODS},
        "validation_caveat": split["limitation"],
    }
    write_json(out / "data_summary.json", summary)
    examples = []
    for method in METHODS:
        for full in (False, True):
            selected = [r for r in rows if r["method"] == method and r["full_watermark"] == full and r["image_name"] in train_names]
            selected.sort(key=lambda r: masks[r["image_name"]].mean())
            examples.append(selected[len(selected)//2])
    contact_sheet(examples, root, masks, out / "examples.png")
    print("INSPECTION", json.dumps(summary), flush=True)
    return rows, masks, split


def message_scores(logits, bits):
    signs = bits.to(logits.device).float() * 2 - 1
    while signs.ndim < logits.ndim:
        signs = signs.unsqueeze(-1)
    signed = logits.float() * signs
    return torch.stack([(signed > 0).float().mean(1), signed.sigmoid().mean(1)], dim=1)


def extract_features(args, rows, split):
    from wmloc.watermarks import load_watermark
    out = args.output_root / "features"
    out.mkdir(exist_ok=True)
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        names = [r["image_name"] for r in subset]
        cache = out / f"{method}.npz"
        signature = f"{FEATURE_VERSION}:{split['metadata_sha256']}"
        if cache.exists():
            with np.load(cache) as z:
                assert z["signature"].item() == signature and z["names"].tolist() == names, "Stale feature cache"
            print("CACHE", method, flush=True)
            continue
        print("LOADING", method, flush=True)
        if method == "stegastamp":
            import onnxruntime as ort
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        wm = load_watermark(method, models_dir=args.kit_root / "models", device=args.device)
        if method == "stegastamp" and "cuda" in args.device:
            assert "CUDAExecutionProvider" in wm.sess.get_providers(), "StegaStamp fell back to CPU"
        output = []
        batch_size = 4 if method != "stegastamp" else 1
        with torch.inference_mode():
            for start in range(0, len(subset), batch_size):
                batch = subset[start:start+batch_size]
                images = torch.from_numpy(np.stack([load_image(args.data_root, r["image_name"]) for r in batch])).permute(0, 3, 1, 2).float().div_(255)
                bits = torch.tensor([[int(b) for b in r["message"]] for r in batch])
                if method != "stegastamp":
                    logits = wm.features(images)["per_position_logits"]
                    scores = F.interpolate(message_scores(logits, bits), size=(128, 128), mode="bilinear", align_corners=False)
                else:
                    # Decode nine overlapping crops; average agreement in each covered cell.
                    positions = [(y, x) for y in (0, 128, 256) for x in (0, 128, 256)]
                    crops = torch.cat([images[:, :, y:y+256, x:x+256] for y,x in positions])
                    logits = wm.logits(crops)
                    crop_scores = message_scores(logits, bits.repeat(len(positions), 1)).cpu()
                    scores, coverage = torch.zeros((1, 2, 128, 128)), torch.zeros((1, 1, 128, 128))
                    for k, (y,x) in enumerate(positions):
                        scores[:, :, y//4:(y+256)//4, x//4:(x+256)//4] += crop_scores[k][None, :, None, None]
                        coverage[:, :, y//4:(y+256)//4, x//4:(x+256)//4] += 1
                    scores /= coverage
                output.append(scores.cpu().numpy())
                if start % 20 == 0:
                    print("FEATURES", method, min(start+batch_size, len(subset)), "/", len(subset), flush=True)
        maps = np.concatenate(output)
        assert maps.shape == (len(names), 2, 128, 128) and np.isfinite(maps).all()
        np.savez_compressed(cache, names=np.array(names), maps=maps, signature=np.array(signature))
        del wm, output, maps
        gc.collect()
        torch.cuda.empty_cache()


def image_scores(pred, gt):
    """Vectorized scoring; compared against the official helper before reporting."""
    p, g = pred.astype(bool), gt.astype(bool)
    result = np.zeros(len(g), dtype=np.float64)
    for a, b in ((p, g), (~p, ~g)):
        intersection = (a & b).sum(axis=(1, 2))
        union = (a | b).sum(axis=(1, 2))
        result += np.divide(intersection, union, out=np.ones(len(g), dtype=float), where=union != 0) / 2
    return result


def evaluate(args, rows, masks, split):
    from wmloc.utils import miou_image
    val = set(split["validation"])
    results, chosen_masks, chosen_values, per_image, feature_masks = {}, {}, {}, [], {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        names = [r["image_name"] for r in subset]
        gt = np.stack([masks[n] for n in names])
        is_val = np.array([n in val for n in names])
        tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        with np.load(args.output_root / "features" / f"{method}.npz") as z:
            assert z["names"].tolist() == names
            maps = z["maps"]
        prior = gt[tr].mean(0)
        candidates = {}
        predictions = {}
        for label, pred in (("all_clean", np.zeros_like(gt)), ("all_watermarked", np.ones_like(gt))):
            predictions[label] = pred
            candidates[label] = {"threshold": None, "train_miou": float(image_scores(pred[tr], gt[tr]).mean())}
        for label, values, thresholds in (
            ("train_spatial_prior", np.broadcast_to(prior, gt.shape), np.linspace(0.05, 0.95, 37)),
            ("message_bit_agreement", maps[:, 0], np.linspace(0.30, 0.95, 66)),
            ("message_soft_agreement", maps[:, 1], np.linspace(0.30, 0.95, 66)),
        ):
            scores = [float(image_scores(values[tr] >= t, gt[tr]).mean()) for t in thresholds]
            best = int(np.argmax(scores))
            threshold = float(thresholds[best])
            predictions[label] = (values >= threshold).astype(np.uint8)
            candidates[label] = {"threshold": threshold, "train_miou": scores[best]}
        # Both decisions are frozen using calibration masks only, before looking at validation scores.
        chosen = max(candidates, key=lambda k: candidates[k]["train_miou"])
        chosen_feature = max(("message_bit_agreement", "message_soft_agreement"), key=lambda k: candidates[k]["train_miou"])
        for label, pred in predictions.items():
            measured = image_scores(pred[va], gt[va])
            official = np.array([miou_image(p,g) for p,g in zip(pred[va], gt[va])])
            assert np.allclose(measured, official, atol=1e-12)
            candidates[label]["validation_miou"] = float(official.mean())
        results[method] = {"n_train": len(tr), "n_validation": len(va), "selected_on_train": chosen,
                           "selected_message_feature_on_train": chosen_feature, "candidates": candidates}
        feature_index = 0 if chosen_feature == "message_bit_agreement" else 1
        for i in va:
            name = names[i]
            chosen_masks[name] = predictions[chosen][i]
            feature_masks[name] = predictions[chosen_feature][i]
            chosen_values[name] = maps[i, feature_index]
            per_image.append({"image_name": name, "method": method,
                              "watermarked_fraction": float(gt[i].mean()),
                              "selected_miou": miou_image(predictions[chosen][i], gt[i]),
                              "message_feature_miou": miou_image(predictions[chosen_feature][i], gt[i])})
    total = sum(v["n_validation"] for v in results.values())
    aggregate = {label: sum(v["n_validation"] * v["candidates"][label]["validation_miou"] for v in results.values()) / total
                 for label in ("all_clean", "all_watermarked", "train_spatial_prior", "message_bit_agreement", "message_soft_agreement")}
    aggregate["selected_per_method_on_train"] = float(np.mean([r["selected_miou"] for r in per_image]))
    aggregate["message_feature_selected_on_train"] = float(np.mean([r["message_feature_miou"] for r in per_image]))
    report = {"feature_version": FEATURE_VERSION, "seed": args.seed, "split_sha256": digest((args.output_root/'split.json').read_bytes()),
              "validation_images": total, "methods": results, "validation_miou": aggregate,
              "selection_rule": "All thresholds, feature choices and fallback choices use calibration images only.",
              "caveats": [split["limitation"], "This is one local split, not a leaderboard or hidden-test estimate.",
                          "full_watermark and ground-truth mask area are not inference inputs."]}
    write_json(args.output_root / "metrics.json", report)
    write_json(args.output_root / "per_image_validation.json", per_image)
    np.savez_compressed(args.output_root / "validation_predictions.npz", names=np.array(sorted(chosen_masks)),
                        masks=np.stack([chosen_masks[n] for n in sorted(chosen_masks)]))
    by_name = {r["image_name"]:r for r in rows}
    for method in METHODS:
        ordered = sorted([r for r in per_image if r["method"] == method], key=lambda r: r["message_feature_miou"])
        selected = [by_name[ordered[i]["image_name"]] for i in (0, len(ordered)//2, len(ordered)-1)]
        contact_sheet(selected, args.data_root, masks, args.output_root / f"validation_{method}.png", feature_masks, chosen_values)
    print("RESULTS", json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit-root", required=True, type=Path)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--output-root", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--stage", choices=("inspect", "all"), default="all")
    args = ap.parse_args()
    args.kit_root, args.data_root, args.output_root = (p.resolve() for p in (args.kit_root, args.data_root, args.output_root))
    sys.path.insert(0, str(args.kit_root))
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    write_json(args.output_root / "runtime.json", {
        "script_sha256": digest(Path(__file__).read_bytes()),
        "python": sys.version, "host": platform.node(), "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "torchvision", "numpy", "scipy", "Pillow", "trustmark", "onnxruntime-gpu")},
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })
    rows, masks, split = inspect(args)
    if args.stage == "all":
        assert torch.cuda.is_available() or args.device == "cpu", "GPU job required"
        extract_features(args, rows, split)
        evaluate(args, rows, masks, split)


if __name__ == "__main__":
    main()
