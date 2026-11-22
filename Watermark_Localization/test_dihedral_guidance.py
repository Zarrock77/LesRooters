"""CPU contracts for matching cached guidance to the transformed RGB input."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import dihedral_guidance as d


class DihedralGuidanceTests(unittest.TestCase):
    def setUp(self):
        self.row = {"image_name": "fixture", "method": d.METHODS[0], "message": "01"}
        self.rgb = (np.arange(512 * 512 * 3).reshape(512, 512, 3) % 251).astype(np.uint8)
        self.mask = (np.arange(128 * 128).reshape(128, 128) % 5 == 0).astype(np.float32)
        self.base = np.arange(6 * 128 * 128, dtype=np.float32).reshape(6, 128, 128) / 100
        # Deliberately non-equivariant fake decoder outputs: rotating view zero
        # cannot manufacture a different view. This catches stale guidance and
        # accidental double-transformation as well as wrong view indices.
        self.views = np.stack([self.base + 1000 * index for index in range(8)])

    def test_all_eight_views_align_image_mask_and_actual_view_guidance(self):
        dataset = d.DihedralImages(
            [self.row], {"fixture": self.rgb}, {"fixture": self.views},
            {"fixture": self.mask}, augment=True,
        )
        for k in range(4):
            for flip in (False, True):
                with self.subTest(k=k, flip=flip):
                    with patch.object(d.random, "randrange", return_value=k), patch.object(
                        d.random, "random", return_value=.1 if flip else .9
                    ):
                        image, guide, method, mask = dataset[0]
                    expected_image = np.rot90(self.rgb, k, axes=(0, 1))
                    expected_mask = np.rot90(self.mask, k, axes=(0, 1))
                    if flip:
                        expected_image = expected_image[:, ::-1]
                        expected_mask = expected_mask[:, ::-1]
                    actual_image = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
                    np.testing.assert_array_equal(actual_image, expected_image)
                    np.testing.assert_array_equal(mask.numpy(), expected_mask)
                    np.testing.assert_array_equal(guide.numpy(), self.views[2 * k + int(flip)])
                    np.testing.assert_array_equal(d.transform_image(self.rgb, k, flip), expected_image)
                    self.assertTrue(d.transform_image(self.rgb, k, flip).flags.c_contiguous)
                    self.assertEqual(method, 0)
        np.testing.assert_array_equal(dataset.images["fixture"], self.rgb)
        np.testing.assert_array_equal(dataset.dihedral_maps["fixture"], self.views)

    def test_nonaugmented_evaluation_matches_original_dataset(self):
        original = d.experiment.Images(
            [self.row], {"fixture": self.rgb}, {"fixture": self.base},
            {"fixture": self.mask}, augment=False,
        )
        for supplied in (self.base, self.views):
            with self.subTest(shape=supplied.shape):
                candidate = d.DihedralImages(
                    [self.row], {"fixture": self.rgb}, {"fixture": supplied},
                    {"fixture": self.mask}, augment=False,
                )
                for actual, expected in zip(candidate[0], original[0]):
                    if isinstance(expected, torch.Tensor):
                        self.assertTrue(torch.equal(actual, expected))
                    else:
                        self.assertEqual(actual, expected)

    def test_augmented_training_rejects_single_view_guidance(self):
        with self.assertRaisesRegex(ValueError, "Invalid training guidance"):
            d.DihedralImages(
                [self.row], {"fixture": self.rgb}, {"fixture": self.base}, augment=True,
            )

    def test_loader_restores_reader_and_import_path_after_decoder_failure(self):
        # The decoder is replaced entirely. This tests failure cleanup without
        # importing watermark packages, instantiating a model, or using CUDA.
        with tempfile.TemporaryDirectory(prefix="lesrooters-dihedral-cpu-") as temporary:
            root = Path(temporary)
            metadata = root / "release/train/metadata.json"
            metadata.parent.mkdir(parents=True)
            rows = [
                {"image_name": method, "method": method, "message": "01"}
                for method in d.METHODS
            ]
            metadata.write_text(json.dumps(rows), encoding="utf-8")
            original = root / "output/guidance_v2/guidance/train"
            original.mkdir(parents=True)
            for method in d.METHODS:
                np.savez_compressed(
                    original / f"{method}.npz", names=np.array([method]),
                    maps=np.zeros((1, 6, 128, 128), np.float16),
                    version=np.array(d.experiment.FEATURE_VERSION),
                )
            before = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in original.iterdir()}
            calls = []

            def fake_reader(data, split, name):
                self.assertEqual(split, "train")
                return self.rgb.copy()

            def failing_decoder(args, selected_rows, split):
                index = len(calls)
                calls.append(index)
                actual = d.experiment.read_image(args.data_root, split, selected_rows[0]["image_name"])
                k, flip = divmod(index, 2)
                expected = np.rot90(self.rgb, k, axes=(0, 1))
                if flip:
                    expected = expected[:, ::-1]
                np.testing.assert_array_equal(actual, expected)
                if index == 3:
                    raise RuntimeError("deliberate decoder failure")
                return {row["image_name"]: np.zeros((6, 128, 128), np.float16) for row in selected_rows}

            original_path = list(sys.path)
            with patch.object(d.experiment, "read_image", fake_reader), patch.object(
                d.experiment, "guidance", failing_decoder
            ), patch.object(d, "asset_provenance", return_value={"fixture": "CPU-only assets"}):
                with self.assertRaisesRegex(RuntimeError, "deliberate decoder failure"):
                    d.load_dihedral_guidance(root)
                self.assertIs(d.experiment.read_image, fake_reader)
            self.assertEqual(calls, [0, 1, 2, 3])
            self.assertEqual(sys.path, original_path)
            after = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in original.iterdir()}
            self.assertEqual(after, before)

    def test_cache_rejects_changed_or_missing_transformation_provenance(self):
        with tempfile.TemporaryDirectory(prefix="lesrooters-dihedral-provenance-") as temporary:
            root = Path(temporary)
            path = root / "view_1/transform.json"
            manifest = {
                "version": d.TRANSFORM_VERSION, "k": 0, "flip": True,
                "metadata_sha256": "original", "transform_source_sha256": "original-code",
            }
            d._freeze_manifest(path, manifest)
            before = path.read_bytes()
            d._freeze_manifest(path, manifest)
            self.assertEqual(path.read_bytes(), before)
            for key, value in (("k", 1), ("flip", False), ("metadata_sha256", "changed"),
                               ("transform_source_sha256", "changed-code")):
                with self.subTest(changed=key), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                    d._freeze_manifest(path, {**manifest, key: value})
            self.assertEqual(path.read_bytes(), before)
            unversioned = root / "view_2/guidance/train"
            unversioned.mkdir(parents=True)
            (unversioned / "pixelseal.npz").write_bytes(b"partial-cache")
            with self.assertRaisesRegex(ValueError, "unversioned transformed guidance"):
                d._freeze_manifest(root / "view_2/transform.json", manifest)
            self.assertFalse((root / "view_2/transform.json").exists())

    def test_asset_changes_invalidate_cache_and_changed_model_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lesrooters-dihedral-assets-") as temporary:
            root = Path(temporary)
            code, kit = root / "code", root / "kit"
            code.mkdir()
            checkpoint = kit / "models/fixture.ckpt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"known checkpoint")
            (code / "cloud_asset_checksums.json").write_text(json.dumps({
                "models/fixture.ckpt": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "release/test/metadata.json": "never-read-this-file",
            }), encoding="utf-8")
            (code / "cloud_environment.lock.txt").write_text("pinned-test-environment", encoding="utf-8")
            wrapper = kit / "wmloc/watermarks.py"
            wrapper.parent.mkdir()
            wrapper.write_text("original wrapper", encoding="utf-8")
            image = kit / "release/train/images/fixture.png"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"source image fixture")
            with patch.object(d, "__file__", str(code / "dihedral_guidance.py")):
                before = d.asset_provenance(kit, [self.row])
                self.assertEqual(set(before["source_png_sha256"]), {"fixture"})
                path = root / "cache/transform.json"
                d._freeze_manifest(path, {"assets": before})
                for changed in (wrapper, image, code / "cloud_environment.lock.txt"):
                    original = changed.read_bytes()
                    changed.write_bytes(original + b" changed")
                    after = d.asset_provenance(kit, [self.row])
                    with self.subTest(asset=changed.name), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                        d._freeze_manifest(path, {"assets": after})
                    changed.write_bytes(original)
                checkpoint.write_bytes(b"changed checkpoint")
                with self.assertRaisesRegex(ValueError, "Model checksum mismatch"):
                    d.asset_provenance(kit, [self.row])


if __name__ == "__main__":
    unittest.main()
