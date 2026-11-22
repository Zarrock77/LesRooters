"""Prepare external COCO photographs for synthetic watermark pretraining (CPU only).

Reads competition RGB files solely to exclude duplicates. It never opens their
metadata, messages, masks, or labels. COCO object annotations are not used.
Source photos are not redistributed by this script or committed to the repo.

Run separate output directories for smoke and full runs: the specification,
including sample count, is immutable once preparation starts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.parse import urlsplit, urlunsplit
import zipfile

import numpy as np
import PIL
from PIL import Image, UnidentifiedImageError
import requests
import scipy
from scipy.fft import dctn


VERSION = "external-coco-clean-v1"
SEED = 20261130
HAMMING_LIMIT = 12
HOST = "images.cocodataset.org"
ARCHIVE_PATH = "/annotations/annotations_trainval2017.zip"
ARCHIVE_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"
ARCHIVE_BYTES = 252907541
DOWNLOAD_SOURCE = "https://github.com/cocodataset/cocodataset.github.io/blob/master/dataset/download.htm"
CHECKSUM_SOURCE = "https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/url_checksums/coco_captions.txt"
LICENSE_NAMES = {
    4: "Attribution License",
    5: "Attribution-ShareAlike License",
    7: "No known copyright restrictions",
    8: "United States Government Work",
}
DISCLAIMER = (
    "Exact RGB and perceptual hashes are conservative exclusion heuristics, not "
    "proof of source independence. COCO photos are assumed to be unmodified base "
    "images, not certified free of pre-existing watermarks. License names and "
    "source links are retained as supplied by COCO; this is not a legal clearance "
    "or a claim that the metadata contains complete author attribution. Do not "
    "redistribute photographs without meeting their individual license terms."
)


def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def frozen_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Frozen specification/provenance changed: {path}")
    else:
        write_json(path, value)


def hash_bits(bits):
    return sum(int(bit) << index for index, bit in enumerate(bits.ravel()))


def fingerprints(image):
    """Match baseline.py's 63-bit pHash; add the standard 64-bit horizontal dHash."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = image.convert("L")
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    low = dctn(small, norm="ortho")[:8, :8].ravel()[1:]
    dh = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    return {
        "rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
        "size": list(image.size),
        "phash63": hash_bits(low > np.median(low)),
        "dhash64": hash_bits(dh[:, 1:] > dh[:, :-1]),
    }


class ExclusionIndex:
    def __init__(self):
        self.exact = {}
        self.perceptual = []

    def add(self, fingerprint, source):
        self.exact.setdefault(fingerprint["rgb_sha256"], source)
        self.perceptual.append((fingerprint["phash63"], fingerprint["dhash64"], source))

    def match(self, fingerprint):
        exact = self.exact.get(fingerprint["rgb_sha256"])
        if exact is not None:
            return {"kind": "exact_rgb", "source": exact, "distance": 0}
        ph, dh = fingerprint["phash63"], fingerprint["dhash64"]
        for other_ph, other_dh, source in self.perceptual:
            p_distance, d_distance = (ph ^ other_ph).bit_count(), (dh ^ other_dh).bit_count()
            if p_distance <= HAMMING_LIMIT or d_distance <= HAMMING_LIMIT:
                return {"kind": "near_duplicate", "source": source,
                        "phash_distance": p_distance, "dhash_distance": d_distance}
        return None


def official_exclusions(data_root):
    index, rows = ExclusionIndex(), []
    # Deliberately enumerate only image folders, without opening metadata JSON.
    for split, expected in (("train", 510), ("test", 1020)):
        paths = sorted(p for p in (data_root / split / "images").iterdir()
                       if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        if len(paths) != expected:
            raise RuntimeError(f"Expected all {expected} official {split} RGB images; found {len(paths)}")
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                if rgb.size != (512, 512):
                    raise ValueError(f"Unexpected official image dimensions: {path}")
                fp = fingerprints(rgb)
            source = f"official/{split}/{path.name}"
            rows.append({"source": source, **fp})
            index.add(fp, source)
    return index, rows


def canonical_download_url(url, scheme, image_id=None):
    parsed = urlsplit(url)
    if (parsed.scheme not in ("http", "https") or parsed.hostname != HOST
            or parsed.username or parsed.password or parsed.port not in (None, 80, 443)
            or parsed.query or parsed.fragment):
        raise ValueError("Source URL is outside the allowed official COCO host")
    expected = ARCHIVE_PATH if image_id is None else f"/train2017/{image_id:012d}.jpg"
    if parsed.path != expected:
        raise ValueError(f"Unexpected COCO source path: {parsed.path}")
    return urlunsplit((scheme, HOST, expected, "", ""))


def download(session, url, path, max_bytes, expected_sha=None):
    """Three total GET attempts; no redirects, credential use, or silent downgrade."""
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = path.with_suffix(path.suffix + ".download.json")
    if path.exists():
        measured = sha256_file(path)
        if path.stat().st_size > max_bytes or (expected_sha and measured != expected_sha):
            raise RuntimeError(f"Cached download failed integrity check: {path}")
        receipt = {"url": url, "bytes": path.stat().st_size, "sha256": measured}
        frozen_json(receipt_path, receipt)
        return measured
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            digest, size = hashlib.sha256(), 0
            with session.get(url, stream=True, timeout=(15, 90), allow_redirects=False) as response:
                response.raise_for_status()
                if response.status_code != 200:
                    raise requests.RequestException(f"Expected HTTP 200, got {response.status_code}")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        size += len(chunk)
                        if size > max_bytes:
                            raise requests.RequestException("Download exceeds configured size bound")
                        digest.update(chunk)
                        stream.write(chunk)
            measured = digest.hexdigest()
            if expected_sha and measured != expected_sha:
                raise requests.RequestException("Downloaded archive checksum mismatch")
            temporary.replace(path)
            frozen_json(receipt_path, {"url": url, "bytes": size, "sha256": measured})
            return measured
        except requests.RequestException:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("Unreachable")


def normalize(image):
    rgb = image.convert("RGB")
    width, height = rgb.size
    edge = min(width, height)
    if edge < 32:
        raise ValueError("Source image is too small")
    left, top = (width - edge) // 2, (height - edge) // 2
    normalized = rgb.crop((left, top, left + edge, top + edge)).resize((512, 512), Image.Resampling.LANCZOS)
    array = np.asarray(normalized)
    if (array.shape != (512, 512, 3) or array.dtype != np.uint8
            or not np.isfinite(array).all() or np.all(array == array[0, 0])):
        raise ValueError("Invalid or spatially constant normalized RGB image")
    return rgb, normalized


def prepare(args):
    if not 1 <= args.limit <= args.max_candidates <= 6000:
        raise ValueError("Require 1 <= limit <= max-candidates <= 6000")
    out = args.output_root
    out.mkdir(parents=True, exist_ok=True)
    archive_url = canonical_download_url(f"https://{HOST}{ARCHIVE_PATH}", args.transport)
    spec = {
        "version": VERSION, "seed": SEED, "limit": args.limit,
        "max_candidates": args.max_candidates, "archive_url": archive_url,
        "archive_sha256": ARCHIVE_SHA256, "archive_bytes": ARCHIVE_BYTES,
        "official_download_source": DOWNLOAD_SOURCE, "checksum_source": CHECKSUM_SOURCE,
        "transport": args.transport, "license_names": {str(k): v for k, v in LICENSE_NAMES.items()},
        "normalization": "RGB; centered integer square crop; 512x512 PIL LANCZOS; no EXIF transpose",
        "dedupe": {"exact_rgb": True, "phash_bits": 63, "dhash_bits": 64,
                   "reject_if_either_hamming_distance_at_most": HAMMING_LIMIT,
                   "views": ["original RGB", "normalized RGB"],
                   "references": ["all 510 official train RGB", "all 1020 official test RGB",
                                  "original and normalized previously accepted external RGB"]},
        "selection": "sorted COCO image IDs, random.Random(seed).shuffle, first eligible nonduplicates",
        "competition_labels_messages_metadata_read": False,
        "coco_object_annotations_used": False, "workers": 1, "get_attempts": 3,
        "versions": {"pillow": PIL.__version__, "numpy": np.__version__, "scipy": scipy.__version__},
        "disclaimer": DISCLAIMER,
    }
    frozen_json(out / "protocol.json", spec)  # Before accessing/downloading candidates.
    print("BUILDING_OFFICIAL_RGB_EXCLUSION_INDEX", flush=True)
    index, official_rows = official_exclusions(args.data_root)
    frozen_json(out / "official_rgb_exclusions.json", official_rows)
    print("OFFICIAL_RGB_EXCLUSION_INDEX_OK", len(official_rows), flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = "LesRooters-external-pretraining/1.0"
    archive = out / "cache" / "annotations_trainval2017.zip"
    download(session, archive_url, archive, ARCHIVE_BYTES, ARCHIVE_SHA256)
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise RuntimeError("Unexpected archive size")
    with zipfile.ZipFile(archive) as package:
        with package.open("annotations/instances_train2017.json") as stream:
            document = json.load(stream)
    licenses = {int(row["id"]): row for row in document["licenses"]}
    for license_id, expected in LICENSE_NAMES.items():
        if licenses[license_id]["name"] != expected:
            raise RuntimeError(f"Unexpected COCO license mapping: {licenses[license_id]}")
    frozen_json(out / "licenses.json", {"source": archive_url, "licenses": list(licenses.values()),
                                       "selected_ids": sorted(LICENSE_NAMES), "disclaimer": DISCLAIMER})
    candidates = sorted((row for row in document["images"] if int(row["license"]) in LICENSE_NAMES),
                        key=lambda row: int(row["id"]))
    del document
    random.Random(SEED).shuffle(candidates)
    candidates = candidates[:args.max_candidates]
    frozen_json(out / "candidate_ids.json", [int(row["id"]) for row in candidates])
    accepted, decisions, counts = [], [], Counter()
    image_dir = out / "train" / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        image_id = int(candidate["id"])
        source_id = f"coco_train2017_{image_id:012d}"
        name = f"ext_coco_{image_id:012d}"
        decision_path = out / "cache" / "decisions" / f"{image_id:012d}.json"
        previous = json.loads(decision_path.read_text()) if decision_path.exists() else None
        # Preserve network failures in the ordered sample on restart; no hidden resampling.
        if previous and previous.get("reason") == "download_failed":
            decision = previous
        else:
            url = canonical_download_url(candidate["coco_url"], args.transport, image_id)
            jpeg = out / "cache" / "jpeg" / f"{image_id:012d}.jpg"
            try:
                source_sha = download(session, url, jpeg, 20 * 1024 * 1024)
            except requests.RequestException as error:
                decision = {"source_id": source_id, "accepted": False,
                            "reason": "download_failed", "error_type": type(error).__name__}
            else:
                try:
                    with Image.open(jpeg) as image:
                        original, normalized = normalize(image)
                    original_fp, normalized_fp = fingerprints(original), fingerprints(normalized)
                except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
                    decision = {"source_id": source_id, "accepted": False,
                                "reason": "invalid_image", "error_type": type(error).__name__}
                else:
                    match = index.match(original_fp) or index.match(normalized_fp)
                    if match:
                        decision = {"source_id": source_id, "accepted": False,
                                    "reason": "duplicate", "match": match}
                    else:
                        png = image_dir / f"{name}.png"
                        if png.exists():
                            with Image.open(png) as cached:
                                if cached.mode != "RGB" or fingerprints(cached) != normalized_fp:
                                    raise RuntimeError(f"Cached normalized image changed: {png}")
                        else:
                            temporary = png.with_suffix(".png.part")
                            normalized.save(temporary, format="PNG")
                            temporary.replace(png)
                        license_record = licenses[int(candidate["license"])]
                        record = {
                            "image_name": name, "source_id": source_id, "coco_image_id": image_id,
                            "coco_url": candidate["coco_url"], "download_url": url,
                            "flickr_url": candidate.get("flickr_url"),
                            "license_id": int(candidate["license"]), "license_name": license_record["name"],
                            "license_url": license_record["url"],
                            "author_attribution": "Not supplied by COCO metadata; consult original source URL",
                            "original_sha256": source_sha, "normalized_sha256": sha256_file(png),
                            "original_fingerprint": original_fp, "normalized_fingerprint": normalized_fp,
                        }
                        decision = {"source_id": source_id, "accepted": True, "record": record}
                        accepted.append(record)
                        index.add(original_fp, source_id + "/original")
                        index.add(normalized_fp, source_id + "/normalized")
        frozen_json(decision_path, decision)
        decisions.append({k: v for k, v in decision.items() if k != "record"})
        counts["accepted" if decision["accepted"] else decision["reason"]] += 1
        if len(decisions) % 25 == 0 or len(accepted) == args.limit:
            print("COCO_PROGRESS", json.dumps({"attempted": len(decisions), **counts}), flush=True)
            write_json(out / "progress.json", {"attempted": len(decisions), "counts": dict(counts)})
        if len(accepted) >= args.limit:
            break
    if not accepted:
        raise RuntimeError("No external images accepted; inspect download failures before training")
    write_json(out / "train" / "metadata.json", accepted)
    write_json(out / "decisions.json", decisions)
    manifest = {
        "version": VERSION, "complete": True, "target_met": len(accepted) == args.limit,
        "accepted": len(accepted), "attempted": len(decisions), "counts": dict(counts),
        "metadata_sha256": sha256_file(out / "train" / "metadata.json"),
        "protocol_sha256": sha256_file(out / "protocol.json"),
        "official_rgb_exclusions_sha256": sha256_file(out / "official_rgb_exclusions.json"),
        "candidate_ids_sha256": sha256_file(out / "candidate_ids.json"),
        "archive_sha256": sha256_file(archive), "disclaimer": DISCLAIMER,
    }
    write_json(out / "manifest.json", manifest)
    session.close()
    print("EXTERNAL_COCO_READY", json.dumps(manifest), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Official release directory")
    parser.add_argument("--output-root", type=Path, default=Path("output/external_coco_v1"))
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--max-candidates", type=int, default=6000)
    parser.add_argument("--transport", choices=("https", "http"), default="https",
                        help="Explicit transport; official download page uses HTTP. No silent fallback.")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
