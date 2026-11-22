"""CPU checks for source leakage and synthetic positive/negative pair integrity."""
import copy
import unittest

import numpy as np

from baseline import METHODS
from run_synthetic_cv import validate_parent_split, validate_synthetic_rows


class SyntheticSplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.names = [f'image_{i:03d}' for i in range(510)]
        cls.split = {'fit': cls.names[:324], 'tune': cls.names[324:408],
                     'validation': cls.names[408:], 'seed': 20261122}
        cls.real = [{'image_name': name, 'method': METHODS[i % 3]} for i, name in enumerate(cls.names)]
        cls.rows, cls.masks = [], {}
        zero, one = np.zeros((128, 128), np.uint8), np.ones((128, 128), np.uint8)
        lengths = {'pixelseal': 256, 'trustmark': 100, 'stegastamp': 100}
        for index, source in enumerate(cls.real):
            for method in METHODS:
                if method == source['method']:
                    continue
                pair_id = source['image_name'] + '_' + method
                for positive in (False, True):
                    name = pair_id + '_' + str(positive)
                    cls.rows.append({'image_name': name, 'source_name': source['image_name'],
                                     'source_method': source['method'], 'method': method, 'target_method': method,
                                     'positive': positive, 'message': '01' * (lengths[method] // 2),
                                     'source_sha256': f'{index:064x}', 'pair_id': pair_id, 'seed': index})
                    cls.masks[name] = one if positive else zero

    def test_valid_parent_partition_and_pairs(self):
        validate_parent_split(self.split, set(self.names), self.names[408:], 20261122)
        validate_synthetic_rows(self.rows, self.real, self.masks)

    def test_inner_tune_parent_cannot_enter_fit(self):
        split = copy.deepcopy(self.split)
        split['fit'][0] = split['tune'][0]
        with self.assertRaises(AssertionError):
            validate_parent_split(split, set(self.names), self.names[408:], 20261122)

    def test_outer_parent_cannot_enter_fit(self):
        split = copy.deepcopy(self.split)
        split['fit'][0] = split['validation'][0]
        with self.assertRaises(AssertionError):
            validate_parent_split(split, set(self.names), self.names[408:], 20261122)

    def test_unknown_parent_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['source_name'] = 'unknown_source'
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(rows, self.real, self.masks)

    def test_original_method_cannot_be_synthetic_target(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['method'] = rows[0]['target_method'] = rows[0]['source_method']
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(rows, self.real, self.masks)

    def test_pair_message_must_match(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['message'] = '1' * len(rows[1]['message'])
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(rows, self.real, self.masks)

    def test_message_length_must_match_decoder(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['message'] = rows[0]['message'][:-1]
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(rows, self.real, self.masks)

    def test_pair_source_hash_must_match(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['source_sha256'] = 'f' * 64
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(rows, self.real, self.masks)

    def test_pair_source_identity_must_match(self):
        rows = copy.deepcopy(self.rows)
        # Another parent of the same original method still cannot replace a pair member.
        rows[1]['source_name'] = self.real[3]['image_name']
        rows[1]['source_sha256'] = f'{3:064x}'
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(rows, self.real, self.masks)

    def test_positive_presence_must_match_mask(self):
        masks = dict(self.masks)
        masks[self.rows[1]['image_name']] = masks[self.rows[0]['image_name']]
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(self.rows, self.real, masks)

    def test_duplicate_sample_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[1] = rows[0]
        with self.assertRaises(AssertionError):
            validate_synthetic_rows(rows, self.real, self.masks)


if __name__ == '__main__':
    unittest.main()
