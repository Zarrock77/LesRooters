"""Generate paired, source-traceable watermark examples from TRAIN images only.

Every source supplies a positive/negative pair for each of the other two
watermark methods. The negative is the unchanged source; the positive embeds
the same newly drawn target message in a crop. Optional holes restore original
source pixels. Source masks and full_watermark labels are never used.

IMPORTANT: negatives contain a different method's watermark, and positives can
contain two watermark methods. Their validity assumes method-specific labels;
this distribution differs from the competition data. Use for fit-only synthetic
pretraining followed by real-data finetuning, with descendants grouped by source.
"""
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np
from PIL import Image


VERSION = 'synthetic-crop-embed-paired-v1'
METHODS = ('pixelseal', 'trustmark', 'stegastamp')
BITS = {'pixelseal': 256, 'trustmark': 100, 'stegastamp': 100}
MAX_ASPECT = {'pixelseal': 16., 'trustmark': 2., 'stegastamp': 16.}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def pair_seed(source_name, target_method, base_seed):
    identity = canonical_json([VERSION, base_seed, source_name, target_method])
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], 'big') % (2 ** 63)


def make_geometry(rng, shape, target_method):
    """Sample area uniformly, then a feasible log-uniform crop aspect ratio."""
    height, width = shape
    full_frame = bool(rng.random() < .25)
    if full_frame:
        left = top = 0
        crop_width, crop_height = width, height
    else:
        area = float(rng.uniform(.15, .95)) * height * width
        maximum = MAX_ASPECT[target_method]
        low = max(1 / maximum, area / height ** 2)
        high = min(maximum, width ** 2 / area)
        assert low <= high
        # Integer rounding can move a boundary aspect ratio slightly outside
        # the permitted interval. Reject that geometry without changing area.
        for _ in range(100):
            aspect = math.exp(float(rng.uniform(math.log(low), math.log(high))))
            crop_width = min(width, max(1, round(math.sqrt(area * aspect))))
            crop_height = min(height, max(1, round(math.sqrt(area / aspect))))
            if 1 / maximum <= crop_width / crop_height <= maximum:
                break
        else:
            raise RuntimeError('Cannot sample a permitted integer crop')
        left = int(rng.integers(0, width - crop_width + 1))
        top = int(rng.integers(0, height - crop_height + 1))
    holes = []
    if rng.random() < .5:
        for _ in range(int(rng.integers(1, 4))):
            rx = float(rng.uniform(.08, .28)) * crop_width
            ry = float(rng.uniform(.08, .28)) * crop_height
            cx = float(rng.uniform(left + rx, left + crop_width - rx))
            cy = float(rng.uniform(top + ry, top + crop_height - ry))
            holes.append({'cx': cx, 'cy': cy, 'rx': rx, 'ry': ry})
    return {'shape': [height, width], 'rectangle_xywh': [left, top, crop_width, crop_height],
            'full_frame_before_holes': full_frame, 'holes': holes}


def geometry_mask(geometry):
    height, width = geometry['shape']
    left, top, crop_width, crop_height = geometry['rectangle_xywh']
    assert 0 <= left < left + crop_width <= width
    assert 0 <= top < top + crop_height <= height
    mask = np.zeros((height, width), dtype=bool)
    mask[top:top + crop_height, left:left + crop_width] = True
    y, x = np.ogrid[:height, :width]
    for hole in geometry['holes']:
        inside = ((x + .5 - hole['cx']) / hole['rx']) ** 2 + ((y + .5 - hole['cy']) / hole['ry']) ** 2 <= 1
        mask[inside] = False
    assert mask.any()
    return mask


def downsample_mask(mask):
    """Official target convention: at least 8 of each 4x4 block are marked."""
    mask = np.asarray(mask, dtype=bool)
    assert mask.ndim == 2 and mask.shape[0] % 4 == mask.shape[1] % 4 == 0
    h, w = mask.shape
    return mask.reshape(h // 4, 4, w // 4, 4).sum(axis=(1, 3)) >= 8


def render_positive(source, geometry, bits, embed):
    """Paste only the source crop's own embedded residual; never a donor RGB."""
    assert source.dtype == np.uint8 and source.shape == (*geometry['shape'], 3)
    left, top, width, height = geometry['rectangle_xywh']
    crop = source[top:top + height, left:left + width].copy()
    embedded = np.asarray(embed(crop, bits))
    assert embedded.shape == crop.shape and embedded.dtype == np.uint8
    output = source.copy()
    output[top:top + height, left:left + width] = embedded
    mask = geometry_mask(geometry)
    output[~mask] = source[~mask]
    assert np.array_equal(output[~mask], source[~mask])
    assert np.any(output[mask] != source[mask]), 'Embedder did not change the marked region'
    return output, downsample_mask(mask)


def prepare_sources(metadata, image_root):
    """Whitelist fields: source mask, message and full/partial flag are unused."""
    sources = []
    names = set()
    image_root = Path(image_root).resolve()
    for row in metadata:
        name, method = row['image_name'], row['method']
        assert name and Path(name).name == name and name not in names
        assert method in METHODS
        names.add(name)
        path = (image_root / (name + '.png')).resolve()
        assert path.is_relative_to(image_root) and path.is_file()
        sources.append({'source_name': name, 'source_method': method, 'source_sha256': file_sha256(path)})
    return sorted(sources, key=lambda value: value['source_name'])


def pair_plan(source, target_method, base_seed):
    assert target_method != source['source_method']
    seed = pair_seed(source['source_name'], target_method, base_seed)
    rng = np.random.default_rng(seed)
    message = ''.join(map(str, rng.integers(0, 2, BITS[target_method]).tolist()))
    geometry = make_geometry(rng, (512, 512), target_method)
    pair_id = 'syn_' + target_method + '_' + hashlib.sha256(canonical_json([source['source_name'], target_method, seed])).hexdigest()[:20]
    common = {**source, 'method': target_method, 'target_method': target_method,
              'message': message, 'seed': seed, 'pair_id': pair_id}
    rows = [{**common, 'image_name': pair_id + suffix, 'positive': positive}
            for suffix, positive in (('_pos', True), ('_neg', False))]
    return {'version': VERSION, 'source': source, 'target_method': target_method,
            'seed': seed, 'geometry': geometry, 'rows': rows}


def read_valid_pair(path, plan, spec_sha256, image_root):
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['plan'] != plan or record['spec_sha256'] != spec_sha256:
            return None
        for row in plan['rows']:
            image_path = image_root / (row['image_name'] + '.png')
            if file_sha256(image_path) != record['image_sha256'][row['image_name']]:
                return None
        mask = downsample_mask(geometry_mask(plan['geometry']))
        if hashlib.sha256(mask.tobytes()).hexdigest() != record['positive_mask_sha256']:
            return None
        # Negatives must be an exact copy of the original source PNG.
        if record['image_sha256'][plan['rows'][1]['image_name']] != plan['source']['source_sha256']:
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError, AssertionError):
        return None


def generate_pair(source_path, plan, wrapper, image_root, record_path, spec_sha256):
    # Seeds also cover any stochastic operations in the released embedders.
    import torch
    seed = plan['seed']
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with Image.open(source_path) as image:
        source = np.array(image.convert('RGB'))
    positive, mask = render_positive(source, plan['geometry'],
                                    list(map(int, plan['rows'][0]['message'])), wrapper.embed)
    assert mask.shape == (128, 128) and mask.any()
    for row in plan['rows']:
        destination = image_root / (row['image_name'] + '.png')
        temporary = destination.with_suffix('.tmp.png')
        if row['positive']:
            Image.fromarray(positive).save(temporary)
        else:
            shutil.copyfile(source_path, temporary)
        temporary.replace(destination)
    record = {'plan': plan, 'spec_sha256': spec_sha256,
              'image_sha256': {row['image_name']: file_sha256(image_root / (row['image_name'] + '.png'))
                               for row in plan['rows']},
              'positive_mask_sha256': hashlib.sha256(mask.tobytes()).hexdigest()}
    write_json(record_path, record)
    assert read_valid_pair(record_path, plan, spec_sha256, image_root) is not None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=20261129)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    # This command has intentionally no test split or arbitrary input argument.
    source_root = args.kit_root / 'release/train'
    source_metadata = source_root / 'metadata.json'
    sources = prepare_sources(json.loads(source_metadata.read_text()), source_root / 'images')
    assert len(sources) == 510, 'Expected the complete official TRAIN source set'
    image_root = args.output_root / 'train/images'
    pair_root = args.output_root / 'pairs'
    image_root.mkdir(parents=True, exist_ok=True)
    pair_root.mkdir(parents=True, exist_ok=True)
    wrapper_source = args.kit_root / 'wmloc/watermarks.py'
    model_checksums = json.loads(Path(__file__).with_name('cloud_asset_checksums.json').read_text())
    model_checksums = {path: value for path, value in model_checksums.items() if path.startswith('models/')}
    for filename, expected in model_checksums.items():
        assert file_sha256(args.kit_root / filename) == expected
    spec = {'version': VERSION, 'base_seed': args.seed, 'source_split': 'train', 'source_count': len(sources),
            'source_metadata_sha256': file_sha256(source_metadata), 'sources': sources,
            'generator_sha256': file_sha256(__file__), 'wrapper_sha256': file_sha256(wrapper_source),
            'model_sha256': model_checksums, 'target_bits': BITS, 'maximum_crop_aspect_ratio': MAX_ASPECT,
            'crop_area_fraction_uniform': [.15, .95], 'full_frame_probability': .25,
            'holes_probability': .5, 'holes_count_uniform_inclusive': [1, 3],
            'hole_radius_fraction_uniform': [.08, .28], 'mask_downsample': '4x4 count >= 8',
            'source_labels_used': ['method'], 'source_masks_used': False, 'source_full_watermark_used': False,
            'lineage_requirement': 'Every descendant must have source_name in the current fold FIT parents; exclude inner-tune and outer parents.',
            'distribution_caveat': 'Unchanged negatives contain another method watermark; positives contain target plus foreign traces. Assumes method-specific target absence and may differ from competition generation. Synthetic pretraining must be followed by real fit finetuning.'}
    spec_path = args.output_root / 'spec.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text()) == spec, 'Existing synthetic specification differs; use a new output root'
    else:
        write_json(spec_path, spec)
    spec_sha256 = file_sha256(spec_path)
    sys.path.insert(0, str(args.kit_root))
    import torch
    from wmloc.watermarks import load_watermark
    torch.set_num_threads(4)
    # Stable algorithms/seed improve replayability; serialized pair hashes are
    # authoritative because exact numerical replay across GPUs is not promised.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    rows, masks = [], []
    pairs_done = 0
    for target_method in METHODS:
        wrapper = None
        plans = [pair_plan(source, target_method, args.seed) for source in sources
                 if source['source_method'] != target_method]
        for plan in plans:
            record_path = pair_root / (plan['rows'][0]['pair_id'] + '.json')
            if read_valid_pair(record_path, plan, spec_sha256, image_root) is None:
                if wrapper is None:
                    if target_method == 'stegastamp' and args.device.startswith('cuda'):
                        import onnxruntime as ort
                        ort.preload_dlls()
                    wrapper = load_watermark(target_method, models_dir=args.kit_root / 'models', device=args.device)
                    assert wrapper.nbits == BITS[target_method]
                source_path = source_root / 'images' / (plan['source']['source_name'] + '.png')
                generate_pair(source_path, plan, wrapper, image_root, record_path, spec_sha256)
            mask = downsample_mask(geometry_mask(plan['geometry']))
            assert mask.any()
            rows.extend(plan['rows'])
            masks.extend((mask, np.zeros_like(mask)))
            pairs_done += 1
            if pairs_done % 20 == 0:
                print('SYNTHETIC_PAIRS', pairs_done, '/', len(sources) * 2, 'TARGET', target_method, flush=True)
        del wrapper
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert pairs_done == 1020 and len(rows) == len(masks) == 2040
    order = sorted(range(len(rows)), key=lambda index: rows[index]['image_name'])
    rows = [rows[index] for index in order]
    masks = np.stack([masks[index] for index in order])
    names = [row['image_name'] for row in rows]
    assert len(set(names)) == len(names)
    metadata_path = args.output_root / 'train/metadata.json'
    masks_path = args.output_root / 'train/masks_128.npz'
    write_json(metadata_path, rows)
    temporary_masks = masks_path.with_suffix('.tmp.npz')
    np.savez_compressed(temporary_masks, names=np.array(names), masks=masks)
    temporary_masks.replace(masks_path)
    manifest = {'version': VERSION, 'rows': len(rows), 'pairs': pairs_done, 'source_count': len(sources),
                'spec_sha256': spec_sha256, 'metadata_sha256': file_sha256(metadata_path),
                'masks_sha256': file_sha256(masks_path), 'all_pair_records_validated': True,
                'negative_images_exact_source_png_copies': True}
    write_json(args.output_root / 'manifest.json', manifest)
    print('SYNTHETIC_COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
