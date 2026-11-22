"""Read-only OOF diagnostics and reproducibly sampled visual comparisons.

These summaries do not select thresholds, models, blend weights or submissions.
full_watermark is used only to describe errors on labeled training data.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from baseline import METHODS, image_scores, write_json


def load_predictions(folder):
    predictions = {}
    for fold in range(5):
        root = folder / ('fold_' + str(fold))
        split = json.loads((root / 'inner_split.json').read_text())
        assert not (set(split['fit']) | set(split['tune'])) & set(split['validation'])
        with np.load(root / 'validation.npz') as values:
            names = values['names'].tolist()
            assert set(names) == set(split['validation'])
            assert not set(names) & set(predictions)
            predictions.update(zip(names, values['masks'].astype(bool)))
    assert len(predictions) == 510
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    parser.add_argument('--candidate', required=True)
    args = parser.parse_args()
    data = args.kit_root / 'release/train'
    destination = args.kit_root / 'output' / ('cv_' + args.candidate + '_v1')
    rows = json.loads((data / 'metadata.json').read_text())
    by_name = {row['image_name']: row for row in rows}
    with np.load(data / 'masks_128.npz') as values:
        truth = dict(zip(values['names'].tolist(), values['masks'].astype(bool)))
    baseline = load_predictions(args.kit_root / 'output/cv_pretrained_v1')
    candidate = load_predictions(destination)
    assert set(baseline) == set(candidate) == set(truth)
    names = sorted(candidate)
    ground_truth = np.stack([truth[name] for name in names])
    predictions = np.stack([candidate[name] for name in names])
    scores = image_scores(predictions, ground_truth)
    reference = image_scores(np.stack([baseline[name] for name in names]), ground_truth)
    delta = scores - reference
    report = {
        'candidate': args.candidate,
        'oof_miou': float(scores.mean()),
        'reference_oof_miou': float(reference.mean()),
        'improved_images': int((delta > 0).sum()),
        'worse_images': int((delta < 0).sum()),
        'unchanged_images': int((delta == 0).sum()),
        'uniform_prediction_images': int((predictions.min((1, 2)) == predictions.max((1, 2))).sum()),
        'figure_selection': 'One seeded random image per watermark/full_watermark stratum; seed 20261127; no selection by prediction quality.',
        'use': 'Descriptive diagnostics only; no fitting or submission decisions.',
        'strata': [],
    }
    selected = []
    rng = np.random.default_rng(20261127)
    for method in METHODS:
        for full in (False, True):
            indices = [i for i, name in enumerate(names)
                       if by_name[name]['method'] == method and by_name[name]['full_watermark'] == full]
            report['strata'].append({'method': method, 'full_watermark': full, 'images': len(indices),
                                     'miou': float(scores[indices].mean()),
                                     'reference_miou': float(reference[indices].mean()),
                                     'gain': float(delta[indices].mean())})
            selected.append(int(rng.choice(indices)))
    width = 192
    sheet = Image.new('RGB', (width * 4, 28 + (width + 28) * len(selected)), 'white')
    draw = ImageDraw.Draw(sheet)
    for column, label in enumerate(('Image', 'Ground truth', 'Current model OOF', 'Learned Adobe OOF')):
        draw.text((column * width + 4, 8), label, fill='black')
    for row, index in enumerate(selected):
        name = names[index]
        y = 28 + row * (width + 28)
        label = (f"{by_name[name]['method']} full={by_name[name]['full_watermark']} "
                 f"mIoU {reference[index]:.3f} -> {scores[index]:.3f}")
        draw.text((4, y + 4), label, fill='black')
        with Image.open(data / 'images' / (name + '.png')) as source:
            sheet.paste(source.convert('RGB').resize((width, width)), (0, y + 28))
        for column, mask in enumerate((truth[name], baseline[name], candidate[name]), 1):
            tile = Image.fromarray(mask.astype(np.uint8) * 255).resize((width, width), Image.Resampling.NEAREST)
            sheet.paste(tile.convert('RGB'), (column * width, y + 28))
    sheet.save(destination / 'oof_examples.jpg', quality=85)
    write_json(destination / 'review.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
