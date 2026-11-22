"""Export the frozen, validated baseline on the official test set (no upload).

Recomputes validation predictions first and requires bit-for-bit equality with
the original baseline. No threshold is retuned and no validation mask is fitted.
"""
import argparse
from collections import Counter
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from baseline import METHODS, FEATURE_VERSION, message_scores, write_json


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_payload(path, expected_names):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert set(payload) == {"image_names", "masks"}
    names, masks = payload["image_names"], payload["masks"]
    assert isinstance(names, list) and all(type(n) is str for n in names)
    assert len(names) == len(set(names)) == 1020
    assert names == expected_names, "Names/order do not match official test metadata"
    assert isinstance(masks, torch.Tensor) and masks.shape == (1020, 128, 128)
    assert masks.dtype in (torch.bool, torch.uint8)
    assert bool(((masks == 0) | (masks == 1)).all())
    assert path.stat().st_size < 20_000_000, "Use the stricter PDF size limit"
    return masks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit-root", type=Path, required=True)
    ap.add_argument("--baseline-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    kit, base, out = args.kit_root.resolve(), args.baseline_root.resolve(), args.output_root.resolve()
    sys.path.insert(0, str(kit))
    from wmloc.watermarks import load_watermark
    data = kit / "release"
    out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "Run inside the GPU Slurm job"
    torch.set_num_threads(4)
    torch.manual_seed(20261121)
    metrics = json.loads((base / "metrics.json").read_text())
    split = json.loads((base / "split.json").read_text())
    runtime = json.loads((base / "runtime.json").read_text())
    assert sha256(Path(__file__).with_name("baseline.py")) == runtime["script_sha256"]
    assert sha256(base / "split.json") == metrics["split_sha256"]
    assert sha256(data / "train" / "metadata.json") == split["metadata_sha256"]
    assert metrics["feature_version"] == FEATURE_VERSION
    train_rows = json.loads((data / "train" / "metadata.json").read_text())
    test_rows = json.loads((data / "test" / "metadata.json").read_text())
    test_names = [r["image_name"] for r in test_rows]
    assert len(test_names) == len(set(test_names)) == 1020
    assert Counter(r["method"] for r in test_rows) == {m: 340 for m in METHODS}
    assert not set(test_names) & {r["image_name"] for r in train_rows}
    for row in test_rows:
        assert set(row) == {"image_name", "method", "message"}
        assert len(row["message"]) == (256 if row["method"] == "pixelseal" else 100)
        assert set(row["message"]) <= {"0", "1"}
        with Image.open(data / "test" / "images" / (row["image_name"] + ".png")) as image:
            assert image.size == (512, 512)
            image.verify()
    with np.load(data / "train" / "masks_128.npz") as z:
        training_masks = dict(zip(z["names"].tolist(), z["masks"]))
    with np.load(base / "validation_predictions.npz") as z:
        reference = dict(zip(z["names"].tolist(), z["masks"]))
    assert set(reference) == set(split["validation"])
    calibration_names = set(split["train"])
    prediction = {}
    selected = {}
    verified = 0
    for method in METHODS:
        result = metrics["methods"][method]
        choice = result["selected_on_train"]
        threshold = result["candidates"][choice]["threshold"]
        selected[method] = {"method": choice, "threshold": threshold}
        fixed = None
        wm = None
        if choice == "train_spatial_prior":
            names = [r["image_name"] for r in train_rows
                     if r["method"] == method and r["image_name"] in calibration_names]
            assert len(names) == result["n_train"]
            prior = np.stack([training_masks[n] for n in names]).mean(0)
            fixed = (prior >= threshold).astype(np.uint8)
        elif choice in ("all_clean", "all_watermarked"):
            fixed = np.full((128, 128), int(choice == "all_watermarked"), dtype=np.uint8)
        else:
            assert choice in ("message_bit_agreement", "message_soft_agreement")
            assert method != "stegastamp", "This exporter expects the v1 StegaStamp fallback"
            wm = load_watermark(method, models_dir=kit / "models", device="cuda")
        for stage, rows in (("train", [r for r in train_rows if r["image_name"] in reference]),
                            ("test", test_rows)):
            subset = [r for r in rows if r["method"] == method]
            for start in range(0, len(subset), 4):
                batch = subset[start:start + 4]
                if fixed is not None:
                    masks = np.repeat(fixed[None], len(batch), axis=0)
                else:
                    images = []
                    for row in batch:
                        with Image.open(data / stage / "images" / (row["image_name"] + ".png")) as image:
                            images.append(np.array(image.convert("RGB")))
                    images = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float().div_(255)
                    bits = torch.tensor([[int(b) for b in r["message"]] for r in batch])
                    with torch.inference_mode():
                        logits = wm.features(images)["per_position_logits"]
                        values = F.interpolate(message_scores(logits, bits), size=(128, 128),
                                               mode="bilinear", align_corners=False)
                    channel = 0 if choice == "message_bit_agreement" else 1
                    # Match the original NumPy thresholding, including its float32 scalar conversion.
                    masks = (values[:, channel].cpu().numpy() >= threshold).astype(np.uint8)
                for row, mask in zip(batch, masks):
                    name = row["image_name"]
                    if stage == "train":
                        assert np.array_equal(mask, reference[name]), f"Validation regression: {name}"
                        verified += 1
                    else:
                        prediction[name] = mask.copy()
                if start % 80 == 0:
                    print("PREDICT", method, stage, min(start + 4, len(subset)), "/", len(subset), flush=True)
        del wm
        gc.collect()
        torch.cuda.empty_cache()
    assert verified == 102
    assert set(prediction) == set(test_names)
    payload = {"image_names": test_names,
               "masks": torch.from_numpy(np.stack([prediction[n] for n in test_names])).bool()}
    target = out / "lesrooters_watermark_baseline_v1.pt"
    temporary = target.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    masks = validate_payload(temporary, test_names)
    temporary.replace(target)
    with target.open("rb") as source, (out / (target.name + ".gz")).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            shutil.copyfileobj(source, compressed)
    manifest = {
        "artifact": target.name, "sha256": sha256(target), "bytes": target.stat().st_size,
        "task_id": "34-watermark-localisation-image", "images": len(test_names),
        "shape": list(masks.shape), "dtype": str(masks.dtype), "binary": True,
        "unique_names_match_test_metadata": True, "validation_predictions_reproduced_exactly": verified,
        "selection": selected, "thresholds_refitted": False,
        "metrics_sha256": sha256(base / "metrics.json"), "split_sha256": sha256(base / "split.json"),
        "test_metadata_sha256": sha256(data / "test" / "metadata.json"),
        "export_script_sha256": sha256(__file__), "baseline_script_sha256": runtime["script_sha256"],
        "slurm_job_id": os.getenv("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(0),
        "prediction_summary": {method: {
            "images": 340,
            "mean_watermarked_fraction": float(np.mean([prediction[r["image_name"]].mean() for r in test_rows if r["method"] == method])),
            "uniform_masks": sum(int(prediction[r["image_name"]].min() == prediction[r["image_name"]].max()) for r in test_rows if r["method"] == method),
        } for method in METHODS},
        "uploaded": False,
    }
    write_json(out / "manifest.json", manifest)
    print("READY", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
