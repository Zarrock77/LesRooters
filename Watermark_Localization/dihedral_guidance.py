"""Cache decoder guidance for the actual eight transformed training inputs.

View ``2 * k + flip`` means a counter-clockwise ``k * 90`` degree rotation,
followed by a horizontal flip when ``flip == 1``. Guidance for a transformed
image is recomputed by the frozen decoder, never obtained by rotating its
original guidance. Importing this module does not load a model or use a GPU.
"""

import hashlib
import inspect
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np
import torch

import localization_v2 as experiment
from baseline import METHODS, message_scores


TRANSFORM_VERSION = "decoder-guidance-actual-dihedral-input-v1"


def transform_image(image, k, flip):
    """Apply the view convention to an HWC array and return contiguous data."""
    if k not in range(4) or flip not in (False, True):
        raise ValueError("Expected k in 0..3 and a boolean flip")
    result = np.rot90(image, k, axes=(0, 1))
    if flip:
        result = np.flip(result, axis=1)
    return np.ascontiguousarray(result)


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def asset_provenance(kit, rows):
    """Bind caches to verified model assets, wrapper, environment and train PNGs."""
    kit = Path(kit).resolve()
    code = Path(__file__).resolve().parent
    checksums_path = code / "cloud_asset_checksums.json"
    expected = json.loads(checksums_path.read_text(encoding="utf-8"))
    models = {}
    for relative, wanted in sorted(expected.items()):
        if not relative.startswith("models/"):
            continue  # Never read masks or test assets for guidance provenance.
        actual = _file_sha256(kit / relative)
        if actual != wanted:
            raise ValueError(f"Model checksum mismatch: {relative}")
        models[relative] = actual
    if not models:
        raise ValueError("No registered model assets found")
    images_root = kit / "release/train/images"
    sources = {}
    for row in rows:
        name = row["image_name"]
        path = images_root / (name + ".png")
        if path.resolve().parent != images_root.resolve():
            raise ValueError("Training image name escapes its source directory")
        sources[name] = _file_sha256(path)
    return {
        "registered_models_sha256": models,
        "model_checksum_registry_sha256": _file_sha256(checksums_path),
        "watermarks_wrapper_sha256": _file_sha256(kit / "wmloc/watermarks.py"),
        "cloud_environment_lock_sha256": _file_sha256(code / "cloud_environment.lock.txt"),
        "source_png_sha256": sources,
    }


def _freeze_manifest(path, value):
    """Reject stale or unversioned transformed caches, including partial ones."""
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Dihedral guidance provenance mismatch: {path}")
        return
    if any((path.parent / "guidance" / "train").glob("*.npz")):
        raise ValueError(f"Refusing unversioned transformed guidance: {path.parent}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_dihedral_guidance(kit_root, output_root=None, *, original_guidance_root=None):
    """Return ``name -> float16[8, 6, 128, 128]`` for real training images.

    Defaults are ``kit_root/output/guidance_dihedral_v1`` for new views and
    ``kit_root/output/guidance_v2`` for the existing view-zero cache. The latter
    must already contain all three method caches; this function never rebuilds
    or modifies it. Dataset reads are limited to ``release/train/metadata.json``
    and training images, alongside model/wrapper/environment provenance reads.
    Labels, masks, and test images are not used to construct guidance.

    Call serially, before constructing loaders/training: the existing guidance
    extractor's image reader is temporarily replaced and restored in a finally
    block, also if decoding or provenance checks fail. Missing nonzero views
    require a GPU; importing this module does not invoke this function.
    """
    kit_root = Path(kit_root).resolve()
    output_root = Path(output_root or kit_root / "output/guidance_dihedral_v1").resolve()
    original_guidance_root = Path(
        original_guidance_root or kit_root / "output/guidance_v2"
    ).resolve()
    metadata = kit_root / "release/train/metadata.json"
    metadata_bytes = metadata.read_bytes()
    # Project metadata immediately: full_watermark and any other annotation are
    # deliberately not passed to the decoder-guidance extractor.
    rows = [
        {key: row[key] for key in ("image_name", "method", "message")}
        for row in json.loads(metadata_bytes)
    ]
    names = [row["image_name"] for row in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError("Training metadata must contain unique image names")
    if set(row["method"] for row in rows) != set(METHODS):
        raise ValueError("Training metadata must contain all three known methods")
    original_files = [
        original_guidance_root / "guidance/train" / f"{method}.npz"
        for method in METHODS
    ]
    if not all(path.is_file() for path in original_files):
        raise FileNotFoundError("All original training guidance caches must exist before loading views")

    original_reader = experiment.read_image
    common = {
        "version": TRANSFORM_VERSION,
        "split": "train",
        "metadata_sha256": _sha256(metadata_bytes),
        "images": len(rows),
        "guidance_feature_version": experiment.FEATURE_VERSION,
        "guidance_source_sha256": _sha256(inspect.getsource(experiment.guidance).encode()),
        "read_image_source_sha256": _sha256(inspect.getsource(original_reader).encode()),
        "message_scores_source_sha256": _sha256(inspect.getsource(message_scores).encode()),
        "transform_source_sha256": _sha256(inspect.getsource(transform_image).encode()),
        "operation": "np.rot90(image,k,axes=(0,1)); optional np.flip(axis=1); contiguous",
        "view_index": "2*k+flip",
        "assets": asset_provenance(kit_root, rows),
    }
    # wmloc is imported lazily inside the existing extractor.
    kit_import = str(kit_root)
    added_import_path = kit_import not in sys.path
    if added_import_path:
        sys.path.insert(0, kit_import)
    views = []
    try:
        for index in range(8):
            k, flip = divmod(index, 2)
            folder = output_root / f"view_{index}"
            manifest = {**common, "index": index, "k": k, "flip": bool(flip)}
            if index == 0:
                manifest["original_cache_sha256"] = {
                    path.name: _sha256(path.read_bytes()) for path in original_files
                }
            _freeze_manifest(folder / "transform.json", manifest)

            def view_reader(data, split, name, _k=k, _flip=bool(flip)):
                if split != "train":
                    raise ValueError("Dihedral guidance is restricted to the real training split")
                return transform_image(original_reader(data, split, name), _k, _flip)

            experiment.read_image = view_reader
            args = SimpleNamespace(
                kit_root=kit_root,
                data_root=kit_root / "release",
                output_root=original_guidance_root if index == 0 else folder,
            )
            maps = experiment.guidance(args, rows, "train")
            if set(maps) != set(names):
                raise ValueError(f"Image membership mismatch in dihedral view {index}")
            for name, value in maps.items():
                if value.shape != (6, 128, 128) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid guidance for {name}, view {index}")
            views.append(maps)
            print("DIHEDRAL_GUIDANCE_VIEW_COMPLETE", index, len(maps), flush=True)
    finally:
        experiment.read_image = original_reader
        if added_import_path:
            sys.path.remove(kit_import)
    return {
        name: np.stack([view[name] for view in views]).astype(np.float16, copy=False)
        for name in names
    }


class DihedralImages(experiment.Images):
    """Use the guidance of the sampled input view without transforming it again.

    Training requires eight cached views per image. Non-augmented evaluation
    accepts either the ordinary maps or eight-view maps and uses view zero,
    preserving the existing prediction/export interface.
    """

    def __init__(self, rows, images, maps, masks=None, augment=False):
        self.dihedral_maps = maps
        ordinary = {}
        for row in rows:
            name = row["image_name"]
            value = maps[name]
            if value.shape == (8, 6, 128, 128):
                ordinary[name] = value[0]
            elif not augment and value.shape == (6, 128, 128):
                ordinary[name] = value
            else:
                raise ValueError(f"Invalid {'training' if augment else 'evaluation'} guidance for {name}")
        super().__init__(rows, images, ordinary, masks=masks, augment=augment)

    def __getitem__(self, index):
        if not self.augment:
            return super().__getitem__(index)
        row = self.rows[index]
        name = row["image_name"]
        k = random.randrange(4)
        flip = random.random() < .5
        view = 2 * k + int(flip)
        image = torch.from_numpy(self.images[name].copy()).permute(2, 0, 1).float() / 255
        guide = torch.from_numpy(self.dihedral_maps[name][view].astype(np.float32))
        mask = (
            torch.from_numpy(self.masks[name].astype(np.float32))
            if self.masks is not None else torch.zeros((128, 128))
        )
        image = torch.rot90(image, k, (-2, -1))
        mask = torch.rot90(mask, k, (-2, -1))
        if flip:
            image = image.flip(-1)
            mask = mask.flip(-1)
        return image, guide, METHODS.index(row["method"]), mask
