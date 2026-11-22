"""Fixed external-only pretraining; one final initialization for every real fold.

The six guidance inputs are always 0.5 and become zero inside SpatialSegmenter.
Only the external training parents optimize weights. External validation is read
for one fixed-threshold report after epoch ten, never for model selection.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from baseline import METHODS, image_scores, write_json
from localization_v2 import loss_fn
from spatial_segmenter import SpatialSegmenter


LATE_PREFIXES = tuple('pixel_encoder.' + part + '.' + str(index) + '.'
                      for part in ('downsample_layers', 'stages') for index in (2, 3))


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ExternalSpatialSegmenter(SpatialSegmenter):
    """State-compatible training subclass; deployment uses SpatialSegmenter."""
    def __init__(self, kit_root):
        super().__init__(kit_root)
        for index in (2, 3):
            self.pixel_encoder.downsample_layers[index].requires_grad_(True)
            self.pixel_encoder.stages[index].requires_grad_(True)

    def pixel_features(self, image):
        features = []
        with torch.no_grad():
            x = self.pixel_input(image)
            for index in (0, 1):
                x = self.pixel_encoder.stages[index](self.pixel_encoder.downsample_layers[index](x))
                features.append(x)
        for index in (2, 3):
            x = self.pixel_encoder.stages[index](self.pixel_encoder.downsample_layers[index](x))
            features.append(x)
        return features

    def external_optimizer(self, spec):
        settings = spec['external_optimizer']
        fpn = [parameter for parameter in self.encoder.fpn.parameters() if parameter.requires_grad]
        late = [parameter for parameter in self.pixel_encoder.parameters() if parameter.requires_grad]
        head = [parameter for name, parameter in self.named_parameters()
                if not name.startswith(('encoder.', 'pixel_encoder.')) and parameter.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': fpn, 'lr': settings['fpn_lr'], 'name': 'fpn'},
            {'params': late, 'lr': settings['pixelseal_late_lr'], 'name': 'pixelseal_late'},
            {'params': head, 'lr': settings['head_lr'], 'name': 'head'},
        ], weight_decay=settings['weight_decay'])
        optimized = [parameter for group in optimizer.param_groups for parameter in group['params']]
        assert len(optimized) == len({id(parameter) for parameter in optimized})
        assert {id(parameter) for parameter in optimized} == {
            id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        floor = settings['cosine_final_lr_fraction']
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: floor + (1 - floor) *
            (1 + math.cos(math.pi * epoch / spec['external_epochs'])) / 2)
        return optimizer, scheduler


class ExternalImages(Dataset):
    """Read PNGs lazily; only the compact target masks stay in memory."""
    def __init__(self, rows, image_root, masks):
        self.rows, self.image_root, self.masks = rows, Path(image_root), masks
        self.guide = torch.full((6, 128, 128), .5)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        name = row['image_name']
        with Image.open(self.image_root / (name + '.png')) as image:
            rgb = np.array(image.convert('RGB'))
        assert rgb.shape == (512, 512, 3)
        rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        mask = torch.from_numpy(self.masks[name].astype(np.float32))
        return rgb, self.guide, METHODS.index(row['method']), mask


def validate_data(rows, masks, spec):
    names = [row['image_name'] for row in rows]
    assert len(names) == len(set(names)) == 18000 and set(names) == set(masks)
    parents, pairs, identities = {}, Counter(), {}
    bits = {'pixelseal': 256, 'trustmark': 100, 'stegastamp': 100}
    for row in rows:
        assert row['source_method'] == 'clean'
        assert row['method'] in METHODS and row['target_method'] == row['method']
        assert isinstance(row['positive'], bool)
        assert len(row['message']) == bits[row['method']] and set(row['message']) <= {'0', '1'}
        parent, split = row['source_name'], row['parent_split']
        assert split in ('train', 'validation')
        identity = (split, row['source_sha256'], row['source_id'])
        assert parents.setdefault(parent, identity) == identity, 'Parent split/source differs among descendants'
        pair_identity = tuple(row[key] for key in ('source_name', 'source_sha256', 'parent_split',
                                                   'method', 'message', 'seed'))
        assert identities.setdefault(row['pair_id'], pair_identity) == pair_identity
        pairs[(parent, row['method'], row['positive'])] += 1
        mask = masks[row['image_name']]
        assert mask.shape == (128, 128) and np.isin(mask, [0, 1]).all()
        assert bool(mask.any()) == row['positive']
    assert len(parents) == spec['external_parents'] and len(identities) == 9000
    expected = {(parent, method, positive) for parent in parents for method in METHODS for positive in (False, True)}
    assert set(pairs) == expected and all(count == 1 for count in pairs.values())
    train = {parent for parent, identity in parents.items() if identity[0] == 'train'}
    validation = {parent for parent, identity in parents.items() if identity[0] == 'validation'}
    assert not train & validation
    assert len(train) == spec['external_train_parents'] and len(validation) == spec['external_validation_parents']
    assert len({identity[2] for identity in parents.values()}) == len(parents), 'Duplicate external source ID'
    return train, validation


def frozen_digest(model):
    """Include frozen weights and buffers, especially Adobe batch-norm state."""
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.state_dict().items()):
        frozen = name.startswith('encoder.body.') or (name.startswith('pixel_encoder.') and not name.startswith(LATE_PREFIXES))
        if frozen:
            value = tensor.detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
            count += 1
    assert count > 0
    return digest.hexdigest()


def verify_first_gradients(model):
    assert not model.encoder.body.training and not model.pixel_encoder.training
    for index in (2, 3):
        for module in (model.pixel_encoder.downsample_layers[index], model.pixel_encoder.stages[index]):
            assert sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()
                       if parameter.grad is not None) > 0, 'Missing gradient in late PixelSeal block'
    for module in (model.encoder.fpn, model.fuse):
        assert sum(float(parameter.grad.abs().sum()) for parameter in module.parameters()
                   if parameter.grad is not None) > 0
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)


@torch.inference_mode()
def evaluate(model, dataset, batch_size, workers):
    model.eval()
    scores = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    for image, guide, method, truth in loader:
        image, guide, method = (value.cuda(non_blocking=True) for value in (image, guide, method))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            probability = model(image, guide, method).float().sigmoid()
        measured = image_scores(probability.cpu().numpy() >= .5, truth.numpy().astype(bool))
        scores.extend(measured.tolist())
    assert len(scores) == len(dataset.rows)
    values = np.array(scores)
    return {'miou': float(values.mean()), 'images': len(values), 'threshold': .5,
            'selection': 'Report only after fixed final epoch; no model selection',
            'methods': {method: float(values[[row['method'] == method for row in dataset.rows]].mean()) for method in METHODS},
            'presence': {str(positive): float(values[[row['positive'] == positive for row in dataset.rows]].mean())
                         for positive in (False, True)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    args = parser.parse_args()
    kit = args.kit_root
    sys.path.insert(0, str(kit))
    data_root = kit / 'output/external_watermarks_v1'
    output_root = kit / 'output/external_pretrain_v1'
    output_root.mkdir(parents=True, exist_ok=True)
    spec_path = Path(__file__).with_name('external_experiment.json')
    spec = json.loads(spec_path.read_text())
    assert (spec['external_epochs'], spec['external_batch_size'], spec['epochs'], spec['seed']) == (10, 16, 120, 20261202)
    data_manifest_path = data_root / 'manifest.json'
    data_manifest = json.loads(data_manifest_path.read_text())
    metadata_path = data_root / 'train/metadata.json'
    masks_path = data_root / 'train/masks_128.npz'
    assert data_manifest['metadata_sha256'] == file_sha256(metadata_path)
    assert data_manifest['masks_sha256'] == file_sha256(masks_path)
    assert data_manifest['spec_sha256'] == file_sha256(data_root / 'spec.json')
    assert data_manifest['all_pair_records_validated'] is True
    assert data_manifest['complete'] is True
    assert (data_manifest['rows'], data_manifest['pairs'], data_manifest['source_count']) == (18000, 9000, 3000)
    rows = json.loads(metadata_path.read_text())
    generation = json.loads((data_root / 'spec.json').read_text())
    from generate_external import validate_rows
    validate_rows(rows, generation['sources'])
    assert data_manifest['external_corpus_manifest_sha256'] == generation['external_corpus']['manifest_sha256']
    with np.load(masks_path) as archive:
        assert len(archive['names']) == len(set(archive['names'].tolist()))
        masks = dict(zip(archive['names'].tolist(), archive['masks']))
    train_parents, validation_parents = validate_data(rows, masks, spec)
    train_rows = [row for row in rows if row['source_name'] in train_parents]
    validation_rows = [row for row in rows if row['source_name'] in validation_parents]
    assert (len(train_rows), len(validation_rows)) == (16200, 1800)
    parent_split = {'train': sorted(train_parents), 'validation': sorted(validation_parents)}
    parent_split_sha256 = hashlib.sha256(json.dumps(parent_split, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    source_names = ('pretrain_external.py', 'spatial_segmenter.py', 'adobe_segmenter.py', 'localization_v2.py')
    protocol = {'experiment': spec, 'experiment_sha256': file_sha256(spec_path),
                'source_sha256': {name: file_sha256(Path(__file__).with_name(name)) for name in source_names},
                'data_manifest_sha256': file_sha256(data_manifest_path), 'data_manifest': data_manifest,
                'data_spec_sha256': file_sha256(data_root / 'spec.json'),
                'metadata_sha256': file_sha256(metadata_path), 'masks_sha256': file_sha256(masks_path),
                'parent_split_sha256': parent_split_sha256, 'parents': parent_split,
                'competition_training_rows_used': 0, 'competition_masks_used': False,
                'initialization': 'Released Adobe and PixelSeal weights only',
                'checkpoint_selection': 'Fixed final epoch; no validation selection'}
    protocol_path = output_root / 'protocol.json'
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol, 'External pretraining protocol changed'
    else:
        write_json(protocol_path, protocol)
    final_path = output_root / 'final.pt'
    manifest_path = output_root / 'manifest.json'
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        assert previous['protocol_sha256'] == file_sha256(protocol_path)
        assert previous['checkpoint_sha256'] == file_sha256(final_path)
        print('EXTERNAL_PRETRAIN_ALREADY_COMPLETE', json.dumps(previous), flush=True)
        return

    torch.set_num_threads(4)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    seed = spec['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    train_data = ExternalImages(train_rows, data_root / 'train/images', masks)
    train_loader = DataLoader(train_data, batch_size=spec['external_batch_size'], shuffle=True,
                              num_workers=spec['external_workers'], pin_memory=True, persistent_workers=True,
                              generator=torch.Generator().manual_seed(seed))
    model = ExternalSpatialSegmenter(kit).cuda()
    frozen_before = frozen_digest(model)
    optimizer, scheduler = model.external_optimizer(spec)
    history = []
    first_update = None
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, spec['external_epochs'] + 1):
        started = time.time()
        model.train()
        loss_sum = 0.
        seen = updates = 0
        for image, guide, method, truth in train_loader:
            image, guide, method, truth = (value.cuda(non_blocking=True) for value in (image, guide, method, truth))
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = loss_fn(model(image, guide, method).float(), truth)
            assert torch.isfinite(loss), 'Non-finite external loss'
            loss.backward()
            if first_update is None:
                assert image.shape[0] == 16
                verify_first_gradients(model)
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['external_optimizer']['gradient_clip_norm'])
            assert torch.isfinite(gradient_norm), 'Non-finite external gradients'
            optimizer.step()
            if first_update is None:
                first_update = {'batch_size': int(image.shape[0]), 'late_pixelseal_gradients': True,
                                'frozen_gradients_absent': True,
                                'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2 ** 30,
                                'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2 ** 30}
                write_json(output_root / 'first_update.json', first_update)
                print('EXTERNAL_FIRST_UPDATE_OK', json.dumps(first_update), flush=True)
            loss_sum += float(loss.detach()) * image.shape[0]
            seen += image.shape[0]
            updates += 1
        assert seen == len(train_rows)
        scheduler.step()
        record = {'epoch': epoch, 'images': seen, 'updates': updates,
                  'loss': loss_sum / seen, 'seconds': time.time() - started}
        history.append(record)
        write_json(output_root / 'history.json', history)
        print('EXTERNAL_PRETRAIN_EPOCH', json.dumps(record), flush=True)
    assert len(history) == 10 and sum(record['images'] for record in history) == 162000
    assert frozen_digest(model) == frozen_before, 'A frozen encoder weight or buffer changed'
    optimizer.zero_grad(set_to_none=True)
    del optimizer, scheduler, train_loader, train_data, image, guide, method, truth, loss
    gc.collect()
    torch.cuda.empty_cache()
    validation = evaluate(model, ExternalImages(validation_rows, data_root / 'train/images', masks),
                          spec['external_batch_size'], spec['external_workers'])
    write_json(output_root / 'validation.json', validation)
    provenance = {'protocol_sha256': file_sha256(protocol_path), 'experiment_sha256': file_sha256(spec_path),
                  'data_manifest_sha256': file_sha256(data_manifest_path),
                  'data_spec_sha256': file_sha256(data_root / 'spec.json'),
                  'parent_split_sha256': parent_split_sha256,
                  'source_sha256': protocol['source_sha256'], 'seed': seed, 'epochs': 10,
                  'competition_training_rows_used': 0, 'competition_masks_used': False,
                  'selection': 'Fixed final epoch; validation was report-only at threshold 0.5',
                  'frozen_encoder_sha256': frozen_before, 'torch': str(torch.__version__),
                  'cuda': str(torch.version.cuda), 'gpu': torch.cuda.get_device_name(0)}
    # JSON roundtrip guarantees only weights_only-compatible primitive metadata.
    provenance = json.loads(json.dumps(provenance))
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    restored = SpatialSegmenter(kit)
    restored.load_state_dict(state, strict=True)
    assert all(not parameter.requires_grad for parameter in restored.pixel_encoder.parameters())
    del restored
    gc.collect()
    temporary = final_path.with_suffix('.tmp.pt')
    torch.save({'state_dict': state, 'provenance': provenance,
                'selection': {'epoch': 10, 'rule': 'Fixed final epoch, no model selection'}}, temporary)
    temporary.replace(final_path)
    manifest = {'checkpoint': final_path.name, 'checkpoint_sha256': file_sha256(final_path),
                'checkpoint_bytes': final_path.stat().st_size, 'protocol_sha256': file_sha256(protocol_path),
                'provenance': provenance, 'train_images': len(train_rows), 'validation_images': len(validation_rows),
                'train_parents': len(train_parents), 'validation_parents': len(validation_parents),
                'first_update': first_update, 'external_validation': validation,
                'frozen_encoders_unchanged': True, 'deployment_architecture': 'SpatialSegmenter',
                'strict_spatial_restore_verified': True,
                'real_fold_pixel_encoder_frozen': True, 'real_epochs': 120}
    write_json(manifest_path, manifest)
    print('EXTERNAL_PRETRAIN_COMPLETE', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
