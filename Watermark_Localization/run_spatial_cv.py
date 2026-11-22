"""Bounded spatial-feature experiment, with the incumbent's exact nested splits.

One architecture and one schedule are registered before results exist. The public
leaderboard is never an input. Reused development data is not a pristine test.
"""
import argparse
import gc
import hashlib
import json
from functools import partial
from pathlib import Path
import sys

import numpy as np
import torch

import localization_v2 as experiment
from baseline import METHODS, image_scores, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root', type=Path, required=True)
    ap.add_argument('--guidance-only', action='store_true')
    args = ap.parse_args()
    kit = args.kit_root
    sys.path.insert(0, str(kit))
    torch.set_num_threads(4)
    rows = json.loads((kit / 'release/train/metadata.json').read_text())
    guide_root = kit / 'output/guidance_v2'
    if args.guidance_only:
        experiment.guidance(argparse.Namespace(kit_root=kit, data_root=kit / 'release', output_root=guide_root), rows, 'train')
        print('GUIDANCE_COMPLETE', flush=True)
        return

    from spatial_segmenter import SpatialSegmenter
    root = kit / 'output/cv_spatial_v1'
    reference = kit / 'output/cv_adobe_v1'
    original = json.loads((kit / 'output/baseline_v1/split.json').read_text())
    ref_protocol = json.loads((reference / 'protocol.json').read_text())
    metadata_hash = hashlib.sha256((kit / 'release/train/metadata.json').read_bytes()).hexdigest()
    assert metadata_hash == ref_protocol['metadata_sha256'] == original['metadata_sha256']
    assert len(original['groups']) == 510 and all(len(group) == 1 for group in original['groups'])
    all_names = {r['image_name'] for r in rows}
    folds = ref_protocol['folds']
    assert len(folds) == 5 and all(len(fold) == 102 for fold in folds)
    assert set.union(*(set(fold) for fold in folds)) == all_names and sum(map(len, folds)) == 510
    spec = json.loads(Path(__file__).with_name('spatial_experiment.json').read_text())
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ('spatial_segmenter.py', 'adobe_segmenter.py', 'localization_v2.py', 'run_spatial_cv.py')}
    protocol = {'folds': folds, 'architecture': 'spatial', 'metadata_sha256': metadata_hash,
                'seed': 20261123, 'epochs': spec['epochs'], 'experiment': spec,
                'source_sha256': source_hashes, 'grouping': original['grouping'],
                'inner_selection': '324 fit / 84 tune, exact incumbent membership; checkpoint and thresholds on tune only',
                'outer_validation': '102 per fold, used only after training and checkpoint selection',
                'limitation': 'Reused development dataset; source IDs absent; not an untouched final test.'}
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Experiment changed after registration'
    else:
        write_json(protocol_path, protocol)
    experiment.ResidualSegmenter = partial(SpatialSegmenter, kit)
    with np.load(kit / 'release/train/masks_128.npz') as z:
        truth = dict(zip(z['names'].tolist(), z['masks'].astype(bool)))
    scores, predictions, probabilities = {}, {}, {}
    for fold, validation in enumerate(folds):
        folder = root / f'fold_{fold}'
        split_dir = folder / 'split'
        split_dir.mkdir(parents=True, exist_ok=True)
        split = {**original, 'train': sorted(all_names - set(validation)), 'validation': sorted(validation), 'seed': 20261123}
        write_json(split_dir / 'split.json', split)
        link = folder / 'guidance'
        if not link.exists():
            link.symlink_to(guide_root / 'guidance', target_is_directory=True)
        if not (folder / 'metrics.json').exists():
            sys.argv = ['localization_v2.py', '--kit-root', str(kit), '--baseline-root', str(split_dir),
                        '--output-root', str(folder), '--epochs', str(spec['epochs']), '--seed', str(20261122 + fold)]
            experiment.main()
            report = json.loads((folder / 'metrics.json').read_text())
            report.pop('baseline_validation_miou', None)  # legacy constant is not this fold's baseline
            write_json(folder / 'metrics.json', report)
        actual_split = json.loads((folder / 'inner_split.json').read_text())
        incumbent_split = json.loads((reference / f'fold_{fold}/inner_split.json').read_text())
        assert actual_split == incumbent_split, 'Nested split differs from incumbent'
        with np.load(folder / 'validation.npz') as z:
            names = z['names'].tolist()
            assert set(names) == set(validation)
            measured = image_scores(z['masks'].astype(bool), np.stack([truth[n] for n in names]))
            scores.update(zip(names, measured.tolist()))
            predictions.update(zip(names, z['masks']))
            probabilities.update(zip(names, z['probabilities']))
        print('FOLD_COMPLETE', fold, float(measured.mean()), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    names = sorted(all_names)
    write_json(root / 'image_scores.json', scores)
    write_json(root / 'metrics.json', {'architecture': 'spatial', 'oof_miou': float(np.mean(list(scores.values()))),
               'fold_miou': [float(np.mean([scores[n] for n in fold])) for fold in folds],
               'methods': {m: float(np.mean([scores[r['image_name']] for r in rows if r['method'] == m])) for m in METHODS},
               'images': 510, 'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest()})
    np.savez_compressed(root / 'oof.npz', names=np.array(names), masks=np.stack([predictions[n] for n in names]),
                        probabilities=np.stack([probabilities[n] for n in names]))
    print('SPATIAL_CV_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
