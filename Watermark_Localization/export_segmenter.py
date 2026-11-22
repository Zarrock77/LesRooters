"""Export a trained segmenter after reproducing its frozen validation masks."""
import argparse
import gc
import json
import lzma
import os
from pathlib import Path
import sys

import numpy as np
import torch

import localization_v2 as experiment
from baseline import METHODS, write_json
from prepare_submission import sha256, validate_payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root', type=Path, required=True)
    ap.add_argument('--model-root', type=Path, required=True)
    ap.add_argument('--guidance-root', type=Path, required=True)
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--architecture', choices=('residual', 'pretrained'), default='residual')
    ap.add_argument('--version', required=True)
    args = ap.parse_args()
    args.data_root = args.kit_root / 'release'
    sys.path.insert(0, str(args.kit_root))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    args.output_root.mkdir(parents=True, exist_ok=True)
    train = json.loads((args.data_root/'train/metadata.json').read_text())
    test = json.loads((args.data_root/'test/metadata.json').read_text())
    names = [r['image_name'] for r in test]
    with np.load(args.model_root/'validation.npz') as z:
        reference = dict(zip(z['names'].tolist(), z['masks']))
    assert len(reference) == 102
    inner = json.loads((args.model_root/'inner_split.json').read_text())
    assert set(reference) == set(inner['validation'])
    assert not (set(inner['fit']) | set(inner['tune'])) & set(reference)
    assert not set(names) & {r['image_name'] for r in train}
    # Cache guidance separately from the submission artifact.
    guide_args = argparse.Namespace(kit_root=args.kit_root, data_root=args.data_root,
                                    output_root=args.guidance_root)
    guides = experiment.guidance(guide_args, train, 'train')
    if args.architecture == 'pretrained':
        import pretrained_segmenter
        model = pretrained_segmenter.PretrainedSegmenter().cuda()
    else:
        model = experiment.ResidualSegmenter().cuda()
    verified = 0
    predictions = {}
    selections = {}
    checkpoints = {}
    for stage, rows in (('train', [r for r in train if r['image_name'] in reference]), ('test', test)):
        if stage == 'test':
            del guides
            gc.collect()
            guides = experiment.guidance(guide_args, test, 'test')
        for method in METHODS:
            path = args.model_root/('best_'+method+'.pt')
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            selections[method] = checkpoint['selection']
            checkpoints[method] = sha256(path)
            subset = [r for r in rows if r['method'] == method]
            images = {r['image_name']: experiment.read_image(args.data_root, stage, r['image_name']) for r in subset}
            prob = experiment.predict(model, experiment.Images(subset, images, guides))
            masks = prob >= checkpoint['selection']['threshold']
            for r, mask in zip(subset, masks):
                name = r['image_name']
                if stage == 'train':
                    assert np.array_equal(mask, reference[name]), 'Validation regression: '+name
                    verified += 1
                else:
                    predictions[name] = mask
            print('EXPORTED', stage, method, len(subset), flush=True)
            del images
    assert verified == 102 and set(predictions) == set(names)
    artifact = args.output_root/('lesrooters_watermark_'+args.version+'.pt')
    torch.save({'image_names': names, 'masks': torch.from_numpy(np.stack([predictions[n] for n in names])).bool()}, artifact)
    masks = validate_payload(artifact, names)
    manifest = {
        'artifact': artifact.name, 'sha256': sha256(artifact), 'bytes': artifact.stat().st_size,
        'task_id': '34-watermark-localisation-image', 'images': len(names),
        'shape': list(masks.shape), 'dtype': str(masks.dtype), 'binary': True,
        'unique_names_match_test_metadata': True, 'validation_predictions_reproduced_exactly': verified,
        'validation_reference': args.model_root.name, 'architecture': args.architecture,
        'selection': selections, 'thresholds_refitted': False, 'checkpoint_sha256': checkpoints,
        'metrics_sha256': sha256(args.model_root/'metrics.json'),
        'inner_split_sha256': sha256(args.model_root/'inner_split.json'),
        'test_metadata_sha256': sha256(args.data_root/'test/metadata.json'),
        'export_script_sha256': sha256(__file__), 'model_script_sha256': sha256(experiment.__file__),
        'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'uploaded': False,
    }
    write_json(args.output_root/'manifest.json', manifest)
    with lzma.open(str(artifact)+'.xz', 'wb', preset=6) as f:
        f.write(artifact.read_bytes())
    print('READY', json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
