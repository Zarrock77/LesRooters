"""CPU-only leakage/geometry invariants; requires no watermark model or GPU."""
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image

from generate_synthetic import (BITS, MAX_ASPECT, METHODS, downsample_mask, file_sha256,
                                geometry_mask, make_geometry, pair_plan, prepare_sources,
                                read_valid_pair, render_positive, write_json)


class SyntheticGeometryTests(unittest.TestCase):
    def test_majority_tie_is_foreground(self):
        high = np.zeros((8, 8), dtype=bool)
        high[:2, :4] = True  # Exactly 8 of 16.
        high[:2, 4:] = True
        high[0, 4] = False  # Only 7 of 16.
        high[4:, :4] = True
        expected = np.array([[True, False], [True, False]])
        np.testing.assert_array_equal(downsample_mask(high), expected)

    def test_geometry_obeys_embedder_limits(self):
        for method in METHODS:
            rng = np.random.default_rng(123)
            full = holes = 0
            for _ in range(250):
                geometry = make_geometry(rng, (512, 512), method)
                left, top, width, height = geometry['rectangle_xywh']
                self.assertLessEqual(0, left)
                self.assertLessEqual(0, top)
                self.assertLessEqual(left + width, 512)
                self.assertLessEqual(top + height, 512)
                self.assertLessEqual(max(width / height, height / width), MAX_ASPECT[method])
                area = width * height / 512 ** 2
                if geometry['full_frame_before_holes']:
                    self.assertEqual(area, 1)
                    full += 1
                else:
                    self.assertGreaterEqual(area, .147)
                    self.assertLessEqual(area, .953)
                if geometry['holes']:
                    holes += 1
                    self.assertIn(len(geometry['holes']), (1, 2, 3))
                mask = geometry_mask(geometry)
                self.assertEqual(mask.shape, (512, 512))
                self.assertTrue(downsample_mask(mask).any())
            self.assertTrue(35 < full < 95)
            self.assertTrue(85 < holes < 165)

    def test_same_content_and_holes_are_preserved_exactly(self):
        source = np.random.default_rng(42).integers(0, 220, (512, 512, 3), dtype=np.uint8)
        geometry = {'shape': [512, 512], 'rectangle_xywh': [60, 90, 300, 350],
                    'full_frame_before_holes': False,
                    'holes': [{'cx': 210., 'cy': 240., 'rx': 45., 'ry': 65.}]}
        original = source.copy()
        seen = []

        def fake_embed(crop, bits):
            seen.append((crop.copy(), bits))
            return crop + np.uint8(7)

        result, low_mask = render_positive(source, geometry, [0, 1, 1], fake_embed)
        mask = geometry_mask(geometry)
        np.testing.assert_array_equal(source, original)
        np.testing.assert_array_equal(seen[0][0], original[90:440, 60:360])
        self.assertEqual(seen[0][1], [0, 1, 1])
        np.testing.assert_array_equal(result[~mask], original[~mask])
        np.testing.assert_array_equal(result[mask], original[mask] + np.uint8(7))
        np.testing.assert_array_equal(low_mask, downsample_mask(mask))
        self.assertFalse(mask[240, 210])

    def test_pair_lineage_target_message_and_reproducibility(self):
        source = {'source_name': 'parent_one', 'source_method': 'trustmark', 'source_sha256': 'a' * 64}
        plans = [pair_plan(source, method, 20261129) for method in METHODS if method != 'trustmark']
        for plan in plans:
            self.assertEqual(plan, pair_plan(source, plan['target_method'], 20261129))
            positive, negative = plan['rows']
            self.assertTrue(positive['positive'])
            self.assertFalse(negative['positive'])
            for key in ('source_name', 'source_method', 'source_sha256', 'method', 'message', 'seed', 'pair_id'):
                self.assertEqual(positive[key], negative[key])
            self.assertNotEqual(positive['image_name'], negative['image_name'])
            self.assertNotEqual(positive['method'], source['source_method'])
            self.assertEqual(len(positive['message']), BITS[positive['method']])
            self.assertEqual(set(positive['message']), {'0', '1'})
        self.assertNotEqual(plans[0]['seed'], plans[1]['seed'])
        with self.assertRaises(AssertionError):
            pair_plan(source, 'trustmark', 20261129)

    def test_resume_rejects_modified_outputs_or_specification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / 'parent.png'
            source = np.full((512, 512, 3), 30, dtype=np.uint8)
            Image.fromarray(source).save(source_path)
            # Unknown source labels are ignored, not propagated to descendants.
            sources = prepare_sources([{'image_name': 'parent', 'method': 'trustmark',
                                        'full_watermark': object(), 'message': object()}], root)
            self.assertEqual(set(sources[0]), {'source_name', 'source_method', 'source_sha256'})
            plan = pair_plan(sources[0], 'pixelseal', 123)
            positive, mask = render_positive(source, plan['geometry'], [1], lambda crop, bits: crop + 5)
            positive_path = root / (plan['rows'][0]['image_name'] + '.png')
            negative_path = root / (plan['rows'][1]['image_name'] + '.png')
            Image.fromarray(positive).save(positive_path)
            shutil.copyfile(source_path, negative_path)
            record = {'plan': plan, 'spec_sha256': 'test-spec',
                      'image_sha256': {path.stem: file_sha256(path) for path in (positive_path, negative_path)},
                      'positive_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
            record_path = root / 'pair.json'
            write_json(record_path, record)
            self.assertIsNotNone(read_valid_pair(record_path, plan, 'test-spec', root))
            self.assertIsNone(read_valid_pair(record_path, plan, 'different-spec', root))
            changed = positive.copy()
            changed[0, 0, 0] ^= 1
            Image.fromarray(changed).save(positive_path)
            self.assertIsNone(read_valid_pair(record_path, plan, 'test-spec', root))


if __name__ == '__main__':
    unittest.main()
