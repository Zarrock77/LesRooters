"""Synthetic pretraining with strict parent separation, then fixed real-data CV.

This is one registered combined variant: synthetic pretraining plus guidance
recomputed for each real-image augmentation. Synthetic descendants of inner-tune
and outer images never enter optimization. Checkpoints and thresholds use real
inner tune only, after fixed pretraining with no model selection.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

import localization_v2 as experiment
from baseline import METHODS, image_scores, write_json


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_parent_split(split, all_names, validation, seed):
    fit, tune, outer = (set(split[key]) for key in ('fit', 'tune', 'validation'))
    assert [len(split[key]) for key in ('fit', 'tune', 'validation')] == [324, 84, 102]
    assert (len(fit), len(tune), len(outer)) == (324, 84, 102)
    assert not fit & tune and not fit & outer and not tune & outer, 'Parent split leakage'
    assert fit | tune | outer == all_names and outer == set(validation)
    assert split['seed'] == seed
    return fit, tune, outer


def validate_synthetic_rows(rows, real_rows, masks):
    real = {row['image_name']: row for row in real_rows}
    names = [row['image_name'] for row in rows]
    assert len(rows) == len(set(names)) == 2040 and set(names) == set(masks)
    assert not set(names) & set(real)
    pairs = Counter()
    pair_identity = {}
    source_digests = {}
    bit_lengths = {'pixelseal': 256, 'trustmark': 100, 'stegastamp': 100}
    for row in rows:
        parent = row['source_name']
        assert parent in real and row['source_method'] == real[parent]['method']
        assert row['method'] in METHODS and row['method'] != row['source_method']
        assert row['target_method'] == row['method'] and isinstance(row['positive'], bool)
        assert len(row['message']) == bit_lengths[row['method']] and set(row['message']) <= {'0', '1'}
        source_hash = row['source_sha256']
        assert len(source_hash) == 64 and set(source_hash) <= set('0123456789abcdef')
        assert source_digests.setdefault(parent, source_hash) == source_hash, 'A parent has inconsistent source hashes'
        identity = tuple(row[key] for key in ('source_name', 'source_method', 'source_sha256',
                                               'method', 'target_method', 'message', 'seed'))
        assert pair_identity.setdefault(row['pair_id'], identity) == identity, 'Positive/negative pair differs in message or source'
        mask = masks[row['image_name']]
        assert mask.shape == (128, 128) and np.isin(mask, [0, 1]).all()
        assert bool(mask.any()) == row['positive']
        pairs[(parent, row['method'], row['positive'])] += 1
    expected = {(name, method, positive) for name, row in real.items()
                for method in METHODS if method != row['method'] for positive in (False, True)}
    assert set(pairs) == expected and all(count == 1 for count in pairs.values())
    assert len(pair_identity) == 1020


def pretrain(model, kit, synthetic_root, rows, maps, masks, split, folder, spec, seed):
    fit, tune, outer = (set(split[key]) for key in ('fit', 'tune', 'validation'))
    subset = [row for row in rows if row['source_name'] in fit]
    assert len(subset) == 1296
    assert {row['source_name'] for row in subset} == fit
    assert not {row['source_name'] for row in subset} & (tune | outer)
    images = {row['image_name']: experiment.read_image(synthetic_root, 'train', row['image_name'])
              for row in subset}
    dataset = experiment.Images(subset, images, maps, masks, augment=False)
    generator = torch.Generator().manual_seed(seed)
    sample_count = spec['batch_size'] * spec['pretrain_steps_per_epoch']
    sampler = RandomSampler(dataset, replacement=True, num_samples=sample_count, generator=generator)
    loader = DataLoader(dataset, batch_size=spec['batch_size'], sampler=sampler, num_workers=2,
                        pin_memory=True, persistent_workers=True,
                        generator=torch.Generator().manual_seed(seed + 100000))
    optimizer, scheduler = model.configure_optimization(spec['pretrain_epochs'])
    audit = {'source_parents': sorted(fit), 'synthetic_images': [row['image_name'] for row in subset],
             'seed': seed, 'epochs': spec['pretrain_epochs'], 'steps_per_epoch': len(loader),
             'batch_size': spec['batch_size'], 'augmentation': False,
             'selection': 'Fixed final pretraining epoch, no validation or early stopping',
             'forbidden_parent_overlap': 0}
    write_json(folder / 'pretraining_sources.json', audit)
    assert len(loader) == spec['pretrain_steps_per_epoch']
    history = []
    for epoch in range(1, spec['pretrain_epochs'] + 1):
        start = time.time()
        model.train()
        total = 0.0
        updates = 0
        for x, g, m, y in loader:
            x, g, m, y = (value.cuda(non_blocking=True) for value in (x, g, m, y))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = experiment.loss_fn(model(x, g, m).float(), y)
            assert torch.isfinite(loss), 'Non-finite synthetic pretraining loss'
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total += float(loss.detach())
            updates += 1
        assert updates == spec['pretrain_steps_per_epoch']
        scheduler.step()
        record = {'epoch': epoch, 'updates': updates, 'loss': total / updates,
                  'seconds': time.time() - start}
        history.append(record)
        write_json(folder / 'pretraining_history.json', history)
        print('SYNTHETIC_PRETRAIN', json.dumps(record), flush=True)
    assert sum(record['updates'] for record in history) == 820
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, loader, dataset, images
    gc.collect()
    torch.cuda.empty_cache()
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root', type=Path, required=True)
    ap.add_argument('--guidance-only', action='store_true')
    args = ap.parse_args()
    kit = args.kit_root
    sys.path.insert(0, str(kit))
    torch.set_num_threads(4)
    rows = json.loads((kit / 'release/train/metadata.json').read_text())
    synthetic_root = kit / 'output/synthetic_v1'
    manifest_path = synthetic_root / 'manifest.json'
    assert manifest_path.exists(), 'Synthetic generation must finish before training'
    manifest = json.loads(manifest_path.read_text())
    assert (manifest['rows'], manifest['pairs'], manifest['source_count']) == (2040, 1020, 510)
    assert manifest['all_pair_records_validated'] is True
    synthetic_rows = json.loads((synthetic_root / 'train/metadata.json').read_text())
    with np.load(synthetic_root / 'train/masks_128.npz') as z:
        assert len(z['names']) == len(set(z['names'].tolist()))
        synthetic_masks = dict(zip(z['names'].tolist(), z['masks']))
    validate_synthetic_rows(synthetic_rows, rows, synthetic_masks)
    assert manifest['metadata_sha256'] == digest(synthetic_root / 'train/metadata.json')
    assert manifest['masks_sha256'] == digest(synthetic_root / 'train/masks_128.npz')
    assert manifest['spec_sha256'] == digest(synthetic_root / 'spec.json')
    generation = json.loads((synthetic_root / 'spec.json').read_text())
    assert generation['source_metadata_sha256'] == digest(kit / 'release/train/metadata.json')
    guide_args = argparse.Namespace(kit_root=kit, data_root=synthetic_root, output_root=synthetic_root)
    if args.guidance_only:
        experiment.guidance(guide_args, synthetic_rows, 'train')
        print('SYNTHETIC_GUIDANCE_COMPLETE', flush=True)
        return

    from spatial_segmenter import SpatialSegmenter
    from dihedral_guidance import DihedralImages, load_dihedral_guidance
    root = kit / 'output/cv_synthetic_v1'
    reference = kit / 'output/cv_adobe_v1'
    original = json.loads((kit / 'output/baseline_v1/split.json').read_text())
    ref_protocol = json.loads((reference / 'protocol.json').read_text())
    metadata_hash = digest(kit / 'release/train/metadata.json')
    assert metadata_hash == ref_protocol['metadata_sha256'] == original['metadata_sha256']
    assert len(original['groups']) == 510 and all(len(group) == 1 for group in original['groups'])
    all_names = {row['image_name'] for row in rows}
    folds = ref_protocol['folds']
    assert len(folds) == 5 and all(len(fold) == 102 for fold in folds)
    assert set.union(*(set(fold) for fold in folds)) == all_names and sum(map(len, folds)) == 510
    spec = json.loads(Path(__file__).with_name('synthetic_experiment.json').read_text())
    assert (spec['pretrain_epochs'], spec['pretrain_steps_per_epoch'], spec['batch_size'], spec['epochs']) == (20, 41, 8, 120)
    source_hashes = {name: digest(Path(__file__).with_name(name)) for name in
                     ('spatial_segmenter.py', 'adobe_segmenter.py', 'localization_v2.py',
                      'run_synthetic_cv.py', 'generate_synthetic.py', 'dihedral_guidance.py')}
    protocol = {'folds': folds, 'architecture': 'synthetic', 'metadata_sha256': metadata_hash,
                'seed': 20261123, 'epochs': spec['epochs'], 'experiment': spec,
                'source_sha256': source_hashes, 'grouping': original['grouping'],
                'synthetic_manifest': manifest, 'synthetic_manifest_sha256': digest(manifest_path),
                'synthetic_generation': generation,
                'inner_selection': '324 fit / 84 tune, exact incumbent membership; real tune only',
                'outer_validation': '102 per fold, evaluated only after all checkpoint choices for that fold',
                'limitation': 'Reused development dataset; source IDs absent; synthetic domain shift remains.'}
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'Experiment changed after registration'
    else:
        write_json(protocol_path, protocol)
    maps = experiment.guidance(guide_args, synthetic_rows, 'train')
    view_maps = load_dihedral_guidance(kit)
    original_images = experiment.Images

    def images_factory(selected_rows, images, selected_maps, masks=None, augment=False):
        if augment:
            return DihedralImages(selected_rows, images, view_maps, masks, augment=True)
        return original_images(selected_rows, images, selected_maps, masks, augment=False)

    experiment.Images = images_factory
    with np.load(kit / 'release/train/masks_128.npz') as z:
        truth = dict(zip(z['names'].tolist(), z['masks'].astype(bool)))
    scores, predictions, probabilities = {}, {}, {}
    for fold, validation in enumerate(folds):
        seed = 20261122 + fold
        folder = root / f'fold_{fold}'
        split_dir = folder / 'split'
        split_dir.mkdir(parents=True, exist_ok=True)
        split = {**original, 'train': sorted(all_names - set(validation)),
                 'validation': sorted(validation), 'seed': 20261123}
        write_json(split_dir / 'split.json', split)
        incumbent_split = json.loads((reference / f'fold_{fold}/inner_split.json').read_text())
        validate_parent_split(incumbent_split, all_names, validation, seed)
        link = folder / 'guidance'
        if not link.exists():
            link.symlink_to(kit / 'output/guidance_v2/guidance', target_is_directory=True)

        def factory():
            # localization_v2 writes its exact split before constructing the model.
            actual_split = json.loads((folder / 'inner_split.json').read_text())
            assert actual_split == incumbent_split, 'Nested split differs before pretraining'
            validate_parent_split(actual_split, all_names, validation, seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            model = SpatialSegmenter(kit).cuda()
            return pretrain(model, kit, synthetic_root, synthetic_rows, maps,
                            synthetic_masks, actual_split, folder, spec, seed)

        experiment.ResidualSegmenter = factory
        if not (folder / 'metrics.json').exists() or not (folder / 'validation.npz').exists():
            sys.argv = ['localization_v2.py', '--kit-root', str(kit), '--baseline-root', str(split_dir),
                        '--output-root', str(folder), '--epochs', str(spec['epochs']), '--seed', str(seed)]
            experiment.main()
            report = json.loads((folder / 'metrics.json').read_text())
            report.pop('baseline_validation_miou', None)
            write_json(folder / 'metrics.json', report)
        assert json.loads((folder / 'inner_split.json').read_text()) == incumbent_split
        with np.load(folder / 'validation.npz') as z:
            names = z['names'].tolist()
            assert len(names) == len(set(names)) == 102 and set(names) == set(validation)
            measured = image_scores(z['masks'].astype(bool), np.stack([truth[name] for name in names]))
            scores.update(zip(names, measured.tolist()))
            predictions.update(zip(names, z['masks']))
            probabilities.update(zip(names, z['probabilities']))
        print('SYNTHETIC_FOLD_COMPLETE', fold, float(measured.mean()), flush=True)
        gc.collect()
        torch.cuda.empty_cache()
    names = sorted(all_names)
    assert set(scores) == all_names
    write_json(root / 'image_scores.json', scores)
    write_json(root / 'metrics.json', {'architecture': 'synthetic', 'oof_miou': float(np.mean(list(scores.values()))),
               'fold_miou': [float(np.mean([scores[name] for name in fold])) for fold in folds],
               'methods': {method: float(np.mean([scores[row['image_name']] for row in rows if row['method'] == method]))
                           for method in METHODS}, 'images': 510,
               'protocol_sha256': digest(protocol_path)})
    np.savez_compressed(root / 'oof.npz', names=np.array(names), masks=np.stack([predictions[name] for name in names]),
                        probabilities=np.stack([probabilities[name] for name in names]))
    print('SYNTHETIC_CV_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
