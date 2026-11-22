"""One external-only initialization, then the incumbent's five nested real folds.

Every fold loads the same final checkpoint into an ordinary SpatialSegmenter.
All PixelSeal parameters are frozen again. Real inner tune selects checkpoints
and thresholds; no per-fold external pretraining or outer-label selection occurs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

import localization_v2 as experiment
from baseline import METHODS, image_scores, write_json
from generate_external import asset_provenance, load_external_sources
from run_synthetic_cv import validate_parent_split


SOURCE_FILES = ('run_external_cv.py', 'pretrain_external.py', 'generate_external.py',
                'generate_synthetic.py', 'prepare_external_coco.py',
                'run_synthetic_cv.py', 'spatial_segmenter.py', 'adobe_segmenter.py',
                'localization_v2.py', 'baseline.py', 'dihedral_guidance.py',
                'external_experiment.json', 'cloud_asset_checksums.json',
                'cloud_environment.lock.txt')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pixel_state_digest(state_dict):
    """Hash all PixelSeal parameters and buffers from a complete model state."""
    result, count = hashlib.sha256(), 0
    for name, tensor in sorted(state_dict.items()):
        if name.startswith('pixel_encoder.'):
            value = tensor.detach().cpu().contiguous()
            result.update(name.encode())
            result.update(str((value.dtype, tuple(value.shape))).encode())
            result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
            count += 1
    require(count > 0, 'Checkpoint has no PixelSeal encoder state')
    return result.hexdigest()


def verify_external_pretraining(kit, code_root=None):
    """Verify the complete initialization chain on CPU before any real fitting."""
    kit = Path(kit)
    code = Path(code_root or Path(__file__).resolve().parent)
    root = kit / 'output/external_pretrain_v1'
    data = kit / 'output/external_watermarks_v1'
    corpus_root = kit / 'output/external_coco_v1'
    spec_path = code / 'external_experiment.json'
    spec = read_json(spec_path)
    require((spec['external_epochs'], spec['external_batch_size'], spec['epochs'], spec['seed'])
            == (10, 16, 120, 20261202), 'External experiment schedule changed')
    require((spec['external_parents'], spec['external_train_parents'], spec['external_validation_parents'])
            == (3000, 2700, 300), 'External parent counts changed')
    manifest_path, protocol_path = root / 'manifest.json', root / 'protocol.json'
    manifest, protocol = read_json(manifest_path), read_json(protocol_path)
    final_path = root / 'final.pt'
    require(manifest['checkpoint'] == 'final.pt', 'Unexpected external checkpoint')
    checkpoint_hash = digest(final_path)
    protocol_hash = digest(protocol_path)
    require(manifest['checkpoint_sha256'] == checkpoint_hash, 'External checkpoint checksum mismatch')
    require(manifest['checkpoint_bytes'] == final_path.stat().st_size, 'External checkpoint size changed')
    require(manifest['protocol_sha256'] == protocol_hash, 'External pretraining protocol changed')
    require(protocol['experiment'] == spec and protocol['experiment_sha256'] == digest(spec_path),
            'External pretraining used another experiment specification')
    require(protocol['competition_training_rows_used'] == 0 and protocol['competition_masks_used'] is False,
            'External initialization must not use competition training examples')
    require(protocol['checkpoint_selection'] == 'Fixed final epoch; no validation selection',
            'Unexpected external checkpoint selection')
    require(manifest['deployment_architecture'] == 'SpatialSegmenter'
            and manifest['strict_spatial_restore_verified'] is True
            and manifest['real_fold_pixel_encoder_frozen'] is True
            and manifest['frozen_encoders_unchanged'] is True
            and manifest['real_epochs'] == 120, 'External architecture or freezing proof is missing')
    require((manifest['train_images'], manifest['validation_images'],
             manifest['train_parents'], manifest['validation_parents'])
            == (16200, 1800, 2700, 300), 'External pretraining membership changed')
    expected_pretrain_sources = {'pretrain_external.py', 'spatial_segmenter.py',
                                'adobe_segmenter.py', 'localization_v2.py'}
    require(set(protocol['source_sha256']) == expected_pretrain_sources,
            'Unexpected external pretraining source provenance')
    for name, expected in protocol['source_sha256'].items():
        require(digest(code / name) == expected, f'Pretraining source changed: {name}')

    data_manifest_path, generation_path = data / 'manifest.json', data / 'spec.json'
    data_manifest, generation = read_json(data_manifest_path), read_json(generation_path)
    require(protocol['data_manifest'] == data_manifest
            and protocol['data_manifest_sha256'] == digest(data_manifest_path),
            'External data manifest changed after pretraining')
    require(protocol['data_spec_sha256'] == data_manifest['spec_sha256'] == digest(generation_path),
            'External generation specification changed')
    for key, relative in (('metadata_sha256', 'train/metadata.json'),
                          ('masks_sha256', 'train/masks_128.npz')):
        require(protocol[key] == data_manifest[key] == digest(data / relative),
                f'External generated data changed: {relative}')
    require(data_manifest['complete'] is True and data_manifest['all_pair_records_validated'] is True
            and (data_manifest['rows'], data_manifest['pairs'], data_manifest['source_count'])
            == (18000, 9000, 3000), 'External generation is incomplete')
    require(generation['version'] == 'external-clean-crop-embed-paired-v1'
            and generation['base_seed'] == 20261201 and generation['source_count'] == 3000,
            'Unexpected external generation protocol')
    require(generation['generator_sha256'] == digest(code / 'generate_external.py')
            and generation['synthetic_generator_sha256'] == digest(code / 'generate_synthetic.py'),
            'External generator source changed')
    require(generation['assets'] == asset_provenance(kit), 'Watermark assets changed after generation')
    sources, corpus = load_external_sources(corpus_root, 3000, generation['base_seed'])
    require(generation['sources'] == sources and generation['external_corpus'] == corpus,
            'External clean-source lineage changed')
    require(data_manifest['external_corpus_manifest_sha256'] == corpus['manifest_sha256'],
            'External clean corpus manifest changed')
    parent_split = {split: sorted(source['source_name'] for source in sources
                                  if source['parent_split'] == split)
                    for split in ('train', 'validation')}
    split_hash = hashlib.sha256(json.dumps(parent_split, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(protocol['parents'] == parent_split and protocol['parent_split_sha256'] == split_hash,
            'External parent partition changed')

    checkpoint = torch.load(final_path, weights_only=True, map_location='cpu')
    require(set(checkpoint) == {'state_dict', 'provenance', 'selection'},
            'Unexpected external checkpoint format')
    provenance = checkpoint['provenance']
    require(provenance == manifest['provenance'], 'Checkpoint and manifest provenance differ')
    for key, expected in (
        ('protocol_sha256', protocol_hash), ('experiment_sha256', digest(spec_path)),
        ('data_manifest_sha256', digest(data_manifest_path)),
        ('data_spec_sha256', digest(generation_path)), ('parent_split_sha256', split_hash),
        ('source_sha256', protocol['source_sha256']), ('seed', 20261202), ('epochs', 10),
        ('competition_training_rows_used', 0), ('competition_masks_used', False),
    ):
        require(provenance.get(key) == expected, f'Checkpoint provenance mismatch: {key}')
    require(checkpoint['selection'] == {'epoch': 10, 'rule': 'Fixed final epoch, no model selection'},
            'Unexpected external checkpoint selection record')
    pixel_hash = pixel_state_digest(checkpoint['state_dict'])
    bundle = {'experiment': spec, 'external_pretraining_manifest': manifest,
              'external_pretraining_manifest_sha256': digest(manifest_path),
              'external_pretraining_protocol_sha256': protocol_hash,
              'external_checkpoint_sha256': checkpoint_hash,
              'external_checkpoint_provenance': provenance,
              'external_pixel_encoder_sha256': pixel_hash,
              'external_data_manifest_sha256': digest(data_manifest_path),
              'external_generation_spec_sha256': digest(generation_path),
              'external_clean_corpus_manifest_sha256': corpus['manifest_sha256']}
    return checkpoint, bundle


def restore_initialization(model, checkpoint):
    """Strict state restore must preserve the ordinary model's frozen policy."""
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    require(all(not parameter.requires_grad for parameter in model.pixel_encoder.parameters()),
            'All PixelSeal parameters must remain frozen during real training')
    require(all(not parameter.requires_grad for parameter in model.encoder.body.parameters()),
            'Adobe body must remain frozen during real training')
    model.train()
    require(not model.pixel_encoder.training and not model.encoder.body.training,
            'Frozen encoders must remain in evaluation mode')
    require(pixel_state_digest(model.state_dict()) == pixel_state_digest(checkpoint['state_dict']),
            'Strict restore did not preserve external PixelSeal initialization')
    return model


def guidance_provenance(kit):
    paths = []
    for index in range(8):
        folder = kit / 'output/guidance_dihedral_v1' / f'view_{index}'
        paths.append(folder / 'transform.json')
        root = kit / 'output/guidance_v2' if index == 0 else folder
        paths.extend(root / 'guidance/train' / (method + '.npz') for method in METHODS)
    return {path.relative_to(kit).as_posix(): digest(path) for path in paths}


def freeze_json(path, value):
    if path.exists():
        require(read_json(path) == value, f'Existing experiment provenance differs: {path}')
    else:
        write_json(path, value)


def validate_fold_checkpoints(folder, initialization):
    require(read_json(folder / 'initialization.json') == initialization,
            'Fold initialization differs from the shared external checkpoint')
    selections = read_json(folder / 'selection.json')
    require(set(selections) == set(METHODS), 'Missing method-specific selection')
    history = read_json(folder / 'history.json')
    require([row['epoch'] for row in history] == list(range(1, 121)), 'Real fold schedule is incomplete')
    for method in METHODS:
        selected = selections[method]
        require(selected['epoch'] == 1 or selected['epoch'] in range(5, 121, 5),
                'Unexpected real checkpoint selection epoch')
        require(any(np.isclose(selected['threshold'], value, rtol=0, atol=1e-12)
                    for value in np.linspace(.2, .8, 13)), 'Unexpected real threshold grid')
        checkpoint = torch.load(folder / ('best_' + method + '.pt'), weights_only=True, map_location='cpu')
        require(checkpoint['method'] == method and checkpoint['selection'] == selected,
                'Method checkpoint selection metadata changed')
        require(pixel_state_digest(checkpoint['state_dict']) == initialization['pixel_encoder_sha256'],
                'Frozen PixelSeal encoder changed during real fitting')
        del checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    args = parser.parse_args()
    kit = args.kit_root.resolve()
    code = Path(__file__).resolve().parent
    sys.path.insert(0, str(kit))
    torch.set_num_threads(4)
    checkpoint, external = verify_external_pretraining(kit)
    rows = read_json(kit / 'release/train/metadata.json')
    root, reference = kit / 'output/cv_external_v1', kit / 'output/cv_adobe_v1'
    original = read_json(kit / 'output/baseline_v1/split.json')
    ref_protocol = read_json(reference / 'protocol.json')
    metadata_hash = digest(kit / 'release/train/metadata.json')
    require(metadata_hash == ref_protocol['metadata_sha256'] == original['metadata_sha256'],
            'Official training metadata differs from the incumbent')
    require(len(original['groups']) == 510 and all(len(group) == 1 for group in original['groups']),
            'Reference grouping changed')
    all_names = {row['image_name'] for row in rows}
    folds = ref_protocol['folds']
    require(len(rows) == len(all_names) == 510 and len(folds) == 5
            and all(len(fold) == len(set(fold)) == 102 for fold in folds)
            and set.union(*(set(fold) for fold in folds)) == all_names,
            'Outer folds do not partition the 510 official images')
    incumbent_splits = [read_json(reference / f'fold_{fold}/inner_split.json') for fold in range(5)]
    for fold, validation in enumerate(folds):
        validate_parent_split(incumbent_splits[fold], all_names, validation, 20261122 + fold)

    from spatial_segmenter import SpatialSegmenter
    from dihedral_guidance import DihedralImages, load_dihedral_guidance
    view_maps = load_dihedral_guidance(kit)
    protocol = {'folds': folds, 'architecture': 'external', 'metadata_sha256': metadata_hash,
                'masks_sha256': digest(kit / 'release/train/masks_128.npz'),
                'seed': 20261123, 'epochs': 120, **external,
                'source_sha256': {name: digest(code / name) for name in SOURCE_FILES},
                'reference_protocol_sha256': digest(reference / 'protocol.json'),
                'reference_inner_split_sha256': {str(fold): digest(reference / f'fold_{fold}/inner_split.json')
                                                 for fold in range(5)},
                'guidance_cache_sha256': guidance_provenance(kit), 'grouping': original['grouping'],
                'inner_selection': '324 fit / 84 tune, exact Adobe membership; method checkpoints and thresholds use real tune only',
                'outer_validation': '102 per fold, evaluated only after all checkpoint choices for that fold',
                'initialization': 'Identical external-only final checkpoint for all folds; ordinary SpatialSegmenter; all PixelSeal parameters frozen',
                'limitation': 'Reused development dataset; source IDs absent; external duplicate exclusion is heuristic, not proof of independence.'}
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / 'protocol.json'
    if not protocol_path.exists():
        require(not any(root.glob('fold_*')), 'Refusing existing folds without a registered protocol')
    freeze_json(protocol_path, protocol)
    initialization = {'external_checkpoint_sha256': external['external_checkpoint_sha256'],
                      'external_pretraining_protocol_sha256': external['external_pretraining_protocol_sha256'],
                      'strict_spatial_restore': True, 'pixel_encoder_frozen': True,
                      'pixel_encoder_sha256': external['external_pixel_encoder_sha256']}
    with np.load(kit / 'release/train/masks_128.npz') as archive:
        require(len(archive['names']) == len(set(archive['names'].tolist())) == 510,
                'Invalid official truth membership')
        truth = dict(zip(archive['names'].tolist(), archive['masks'].astype(bool)))
    require(set(truth) == all_names, 'Official masks and metadata differ')
    scores, predictions, probabilities = {}, {}, {}
    original_images, original_model, original_argv = experiment.Images, experiment.ResidualSegmenter, sys.argv

    def images_factory(selected_rows, images, selected_maps, masks=None, augment=False):
        if augment:
            return DihedralImages(selected_rows, images, view_maps, masks, augment=True)
        return original_images(selected_rows, images, selected_maps, masks, augment=False)

    experiment.Images = images_factory
    try:
        for fold, validation in enumerate(folds):
            seed = 20261122 + fold
            folder = root / f'fold_{fold}'
            split_dir = folder / 'split'
            split_dir.mkdir(parents=True, exist_ok=True)
            split = {**original, 'train': sorted(all_names - set(validation)),
                     'validation': sorted(validation), 'seed': 20261123}
            freeze_json(split_dir / 'split.json', split)
            incumbent_split = incumbent_splits[fold]
            link = folder / 'guidance'
            expected_guidance = kit / 'output/guidance_v2/guidance'
            if not link.exists():
                link.symlink_to(expected_guidance, target_is_directory=True)
            require(link.resolve() == expected_guidance.resolve(), 'Fold guidance points to another cache')

            def factory():
                actual_split = read_json(folder / 'inner_split.json')
                require(actual_split == incumbent_split, 'Nested split differs before model construction')
                validate_parent_split(actual_split, all_names, validation, seed)
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                model = restore_initialization(SpatialSegmenter(kit), checkpoint)
                freeze_json(folder / 'initialization.json', initialization)
                return model

            experiment.ResidualSegmenter = factory
            if not (folder / 'metrics.json').exists() or not (folder / 'validation.npz').exists():
                sys.argv = ['localization_v2.py', '--kit-root', str(kit), '--baseline-root', str(split_dir),
                            '--output-root', str(folder), '--epochs', '120', '--seed', str(seed)]
                experiment.main()
                report = read_json(folder / 'metrics.json')
                report.pop('baseline_validation_miou', None)
                write_json(folder / 'metrics.json', report)
            require(read_json(folder / 'inner_split.json') == incumbent_split, 'Completed fold split changed')
            validate_fold_checkpoints(folder, initialization)
            with np.load(folder / 'validation.npz') as archive:
                names = archive['names'].tolist()
                require(len(names) == len(set(names)) == 102 and set(names) == set(validation),
                        'Fold prediction membership differs')
                require(archive['masks'].shape == archive['probabilities'].shape == (102, 128, 128)
                        and np.isfinite(archive['probabilities']).all(), 'Invalid fold predictions')
                measured = image_scores(archive['masks'].astype(bool), np.stack([truth[name] for name in names]))
                scores.update(zip(names, measured.tolist()))
                predictions.update(zip(names, archive['masks']))
                probabilities.update(zip(names, archive['probabilities']))
            print('EXTERNAL_FOLD_COMPLETE', fold, float(measured.mean()), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        experiment.Images, experiment.ResidualSegmenter, sys.argv = original_images, original_model, original_argv
    names = sorted(all_names)
    require(set(scores) == all_names, 'Incomplete out-of-fold predictions')
    write_json(root / 'image_scores.json', scores)
    write_json(root / 'metrics.json', {'architecture': 'external', 'oof_miou': float(np.mean(list(scores.values()))),
               'fold_miou': [float(np.mean([scores[name] for name in fold])) for fold in folds],
               'methods': {method: float(np.mean([scores[row['image_name']] for row in rows if row['method'] == method]))
                           for method in METHODS}, 'images': 510, 'protocol_sha256': digest(protocol_path)})
    np.savez_compressed(root / 'oof.npz', names=np.array(names), masks=np.stack([predictions[name] for name in names]),
                        probabilities=np.stack([probabilities[name] for name in names]))
    print('EXTERNAL_CV_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
