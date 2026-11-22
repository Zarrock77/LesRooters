"""Generate fixed-embedder pairs from screened, licensed external COCO sources.

Only the prepared external corpus is read: no official images, metadata,
messages, masks or labels. Each parent supplies all three methods, each with
one embedded positive and one byte-identical clean negative. All six siblings
share one deterministic source-level pretraining/validation partition.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

import generate_synthetic as synthetic
from generate_synthetic import (BITS, MAX_ASPECT, METHODS, downsample_mask,
                                file_sha256, generate_pair, geometry_mask,
                                pair_plan, read_valid_pair, write_json)


VERSION = 'external-clean-crop-embed-paired-v1'
SEED = 20261201
TRUSTMARK_ASSETS = ('trustmark_Q.yaml', 'decoder_Q.ckpt', 'encoder_Q.ckpt',
                    'trustmark_rm_Q.yaml', 'trustmark_rm_Q.ckpt')
PROVENANCE_FIELDS = ('coco_image_id', 'coco_url', 'download_url', 'flickr_url',
                     'license_id', 'license_name', 'license_url',
                     'author_attribution', 'original_sha256', 'normalized_sha256')


def assign_parent_splits(sources, seed=SEED):
    """Partition parents before expansion; input order never changes membership."""
    sources = sorted(sources, key=lambda row: row['source_name'])
    if len(sources) < 2:
        raise ValueError('At least two external parents are required')
    if (len({row['source_name'] for row in sources}) != len(sources)
            or len({row['source_id'] for row in sources}) != len(sources)):
        raise ValueError('External parent names and IDs must be unique')
    indices = list(range(len(sources)))
    random.Random(seed).shuffle(indices)
    train = set(indices[:len(sources) * 9 // 10])
    return [{**row, 'parent_split': 'train' if index in train else 'validation'}
            for index, row in enumerate(sources)]


def load_external_sources(root, expected_sources=3000, seed=SEED):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest.get('version') != 'external-coco-clean-v1'
            or manifest.get('complete') is not True
            or manifest.get('target_met') is not True
            or manifest.get('accepted') != expected_sources):
        raise ValueError('External corpus is incomplete or has an unexpected source count')
    verified = {}
    for key, relative in (
        ('metadata_sha256', 'train/metadata.json'),
        ('protocol_sha256', 'protocol.json'),
        ('official_rgb_exclusions_sha256', 'official_rgb_exclusions.json'),
        ('candidate_ids_sha256', 'candidate_ids.json'),
    ):
        measured = file_sha256(root / relative)
        if measured != manifest[key]:
            raise ValueError(f'External corpus provenance changed: {relative}')
        verified[relative] = measured
    # This is external-preparation provenance, not competition metadata.
    protocol = json.loads((root / 'protocol.json').read_text(encoding='utf-8'))
    if protocol.get('competition_labels_messages_metadata_read') is not False:
        raise ValueError('External preparation must not use competition labels or messages')
    licenses_path = root / 'licenses.json'
    license_document = json.loads(licenses_path.read_text(encoding='utf-8'))
    licenses = {int(row['id']): row for row in license_document['licenses']}
    verified['licenses.json'] = file_sha256(licenses_path)
    metadata = json.loads((root / 'train/metadata.json').read_text(encoding='utf-8'))
    if len(metadata) != expected_sources:
        raise ValueError('External source count does not match its manifest')
    image_root = root / 'train/images'
    sources, seen_hashes = [], set()
    for row in metadata:
        name = row['image_name']
        if not name or Path(name).name != name:
            raise ValueError('Invalid external parent filename')
        path = image_root / (name + '.png')
        if path.resolve().parent != image_root.resolve():
            raise ValueError('External image escapes its source directory')
        measured = file_sha256(path)
        if measured != row['normalized_sha256'] or measured in seen_hashes:
            raise ValueError('External source PNG changed or is duplicated')
        seen_hashes.add(measured)
        with Image.open(path) as image:
            if image.mode != 'RGB' or image.size != (512, 512):
                raise ValueError('Expected normalized RGB512 external sources')
        license_id = int(row['license_id'])
        if (license_id not in license_document['selected_ids']
                or row['license_name'] != licenses[license_id]['name']
                or row['license_url'] != licenses[license_id]['url']):
            raise ValueError('External source license provenance does not match')
        sources.append({'source_name': name, 'source_id': row['source_id'],
                        'source_method': 'clean', 'source_sha256': measured,
                        'source_provenance': {key: row.get(key) for key in PROVENANCE_FIELDS}})
    sources = assign_parent_splits(sources, seed)
    provenance = {'manifest_sha256': file_sha256(manifest_path),
                  'verified_files_sha256': verified,
                  'archive_sha256': manifest['archive_sha256'],
                  'preparation_disclaimer': manifest.get('disclaimer')}
    return sources, provenance


def asset_provenance(kit):
    """Verify registered assets and upstream TrustMark checksums before generation."""
    kit = Path(kit)
    code = Path(__file__).resolve().parent
    registry = code / 'cloud_asset_checksums.json'
    registered = json.loads(registry.read_text(encoding='utf-8'))
    models = {}
    for relative, expected in sorted(registered.items()):
        if not relative.startswith('models/'):
            continue  # No official metadata/mask files are opened.
        measured = file_sha256(kit / relative)
        if measured != expected:
            raise ValueError(f'Model checksum mismatch: {relative}')
        models[relative] = measured
    if not models:
        raise ValueError('No registered model assets found')
    import trustmark.trustmark as implementation
    trustmark_root = Path(implementation.__file__).parent / 'models'
    trustmark_assets = {}
    for name in TRUSTMARK_ASSETS:
        path = trustmark_root / name
        with path.open('rb') as stream:
            measured_md5 = hashlib.file_digest(stream, 'md5').hexdigest()
        if measured_md5 != implementation.MODEL_CHECKSUMS[name]:
            raise ValueError(f'Upstream TrustMark checksum mismatch: {name}')
        trustmark_assets[name] = file_sha256(path)
    return {'registered_models_sha256': models,
            'model_checksum_registry_sha256': file_sha256(registry),
            'trustmark_assets_sha256': trustmark_assets,
            'trustmark_implementation_sha256': file_sha256(implementation.__file__),
            'pixelseal_card_sha256': file_sha256(kit / 'models/pixelseal_card.yaml'),
            'wrapper_sha256': file_sha256(kit / 'wmloc/watermarks.py'),
            'cloud_environment_lock_sha256': file_sha256(code / 'cloud_environment.lock.txt')}


def validate_rows(rows, sources):
    """Reject sibling leakage, missing targets, or mismatched pair messages."""
    by_source = {source['source_name']: source for source in sources}
    if len(by_source) != len(sources) or len(rows) != 6 * len(sources):
        raise ValueError('Expected exactly six rows per unique external parent')
    observed, names = {}, set()
    for row in rows:
        source = by_source.get(row['source_name'])
        if source is None or any(row.get(key) != value for key, value in source.items()):
            raise ValueError('Synthetic row has unknown or inconsistent parent provenance')
        method = row['method']
        if (method not in METHODS or row['target_method'] != method
                or row['source_method'] != 'clean'
                or type(row['positive']) is not bool
                or len(row['message']) != BITS[method]
                or not set(row['message']) <= {'0', '1'}):
            raise ValueError('Invalid external target or message')
        if row['image_name'] in names:
            raise ValueError('Duplicate synthetic image name')
        names.add(row['image_name'])
        key = (row['source_name'], method)
        observed.setdefault(key, []).append(row)
    for source in sources:
        for method in METHODS:
            pair = observed.get((source['source_name'], method), [])
            if (len(pair) != 2 or {row['positive'] for row in pair} != {False, True}
                    or any(pair[0][key] != pair[1][key]
                           for key in ('message', 'pair_id', 'seed', 'parent_split'))):
                raise ValueError('Missing or inconsistent external positive/negative pair')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', required=True, type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--expected-sources', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    args = parser.parse_args()
    args.kit_root = args.kit_root.resolve()
    source_root = (args.source_root or args.kit_root / 'output/external_coco_v1').resolve()
    output_root = (args.output_root or args.kit_root / 'output/external_watermarks_v1').resolve()
    if source_root == output_root or source_root.is_relative_to(output_root):
        raise ValueError('Generated output must be separate from its source corpus')
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    sources, corpus = load_external_sources(source_root, args.expected_sources, args.seed)
    assets = asset_provenance(args.kit_root)
    image_root, pair_root = output_root / 'train/images', output_root / 'pairs'
    image_root.mkdir(parents=True, exist_ok=True)
    pair_root.mkdir(parents=True, exist_ok=True)
    counts = dict(Counter(row['parent_split'] for row in sources))
    spec = {'version': VERSION, 'base_seed': args.seed, 'source_count': len(sources),
            'sources': sources, 'source_split_counts': counts, 'external_corpus': corpus,
            'generator_sha256': file_sha256(__file__),
            'synthetic_generator_sha256': file_sha256(synthetic.__file__),
            'assets': assets, 'device': args.device, 'torch_threads': 4,
            'partition': 'Sort source_name, random.Random(base_seed).shuffle indices; first floor(0.9*N) train, rest validation; all six descendants stay together.',
            'target_bits': BITS, 'maximum_crop_aspect_ratio': MAX_ASPECT,
            'crop_area_fraction_uniform': [.15, .95], 'full_frame_probability': .25,
            'holes_probability': .5, 'holes_count_uniform_inclusive': [1, 3],
            'hole_radius_fraction_uniform': [.08, .28], 'mask_downsample': '4x4 count >= 8',
            'official_images_metadata_messages_masks_labels_read': False,
            'distribution_caveat': 'External photos are assumed clean of the three target neural watermarks. Original photography and rectangular masks with restored holes differ from the edited official examples. Pretraining must be followed by real fit-only finetuning.'}
    spec_path = output_root / 'spec.json'
    if spec_path.exists():
        if json.loads(spec_path.read_text(encoding='utf-8')) != spec:
            raise ValueError('Existing external specification differs; use a new output root')
    elif list(pair_root.glob('*.json')) or list(image_root.glob('*.png')):
        raise ValueError('Refusing generated pairs without an immutable specification')
    else:
        write_json(spec_path, spec)
    spec_sha256 = file_sha256(spec_path)
    sys.path.insert(0, str(args.kit_root))
    from wmloc.watermarks import load_watermark
    rows, masks, pairs_done = [], [], 0
    for method in METHODS:
        wrapper = None
        for source in sources:
            plan = pair_plan(source, method, args.seed)
            record_path = pair_root / (plan['rows'][0]['pair_id'] + '.json')
            if read_valid_pair(record_path, plan, spec_sha256, image_root) is None:
                if wrapper is None:
                    if method == 'stegastamp' and args.device == 'cuda':
                        import onnxruntime as ort
                        ort.preload_dlls()
                    wrapper = load_watermark(method, models_dir=args.kit_root / 'models', device=args.device)
                    if wrapper.nbits != BITS[method]:
                        raise ValueError('Unexpected official embedder payload size')
                source_path = source_root / 'train/images' / (source['source_name'] + '.png')
                generate_pair(source_path, plan, wrapper, image_root, record_path, spec_sha256)
            mask = downsample_mask(geometry_mask(plan['geometry']))
            rows.extend(plan['rows'])
            masks.extend((mask, np.zeros_like(mask)))
            pairs_done += 1
            if pairs_done % 20 == 0:
                print('EXTERNAL_PAIRS', pairs_done, '/', 3 * len(sources), 'TARGET', method, flush=True)
        del wrapper
        gc.collect()
        if args.device == 'cuda':
            torch.cuda.empty_cache()
    validate_rows(rows, sources)
    order = sorted(range(len(rows)), key=lambda index: rows[index]['image_name'])
    rows = [rows[index] for index in order]
    masks = np.stack([masks[index] for index in order])
    names = [row['image_name'] for row in rows]
    metadata_path, masks_path = output_root / 'train/metadata.json', output_root / 'train/masks_128.npz'
    write_json(metadata_path, rows)
    temporary_masks = masks_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary_masks, names=np.array(names), masks=masks)
    temporary_masks.replace(masks_path)
    manifest = {'version': VERSION, 'complete': True, 'rows': len(rows), 'pairs': pairs_done,
                'source_count': len(sources), 'source_split_counts': counts,
                'row_split_counts': dict(Counter(row['parent_split'] for row in rows)),
                'spec_sha256': spec_sha256, 'metadata_sha256': file_sha256(metadata_path),
                'masks_sha256': file_sha256(masks_path),
                'external_corpus_manifest_sha256': corpus['manifest_sha256'],
                'all_pair_records_validated': True, 'negative_images_exact_source_png_copies': True,
                'official_images_metadata_messages_masks_labels_read': False}
    write_json(output_root / 'manifest.json', manifest)
    print('EXTERNAL_WATERMARKS_COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
