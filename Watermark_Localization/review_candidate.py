"""Descriptive OOF audit with a seeded sample, never model selection."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from baseline import METHODS, image_scores, write_json
from review_cv import load_predictions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root', type=Path, required=True)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--reference', required=True)
    args = ap.parse_args()
    data = args.kit_root / 'release/train'
    root = args.kit_root / 'output' / f'cv_{args.candidate}_v1'
    rows = json.loads((data / 'metadata.json').read_text())
    by_name = {r['image_name']: r for r in rows}
    candidate = load_predictions(root)
    reference = json.loads((args.kit_root / 'output' / f'cv_{args.reference}_v1/image_scores.json').read_text())
    with np.load(data / 'masks_128.npz') as z:
        truth = dict(zip(z['names'].tolist(), z['masks'].astype(bool)))
    names = sorted(candidate)
    assert set(names) == set(reference) == set(truth)
    pred = np.stack([candidate[n] for n in names])
    gt = np.stack([truth[n] for n in names])
    scores = image_scores(pred, gt)
    delta = scores - np.array([reference[n] for n in names])
    report = {'candidate': args.candidate, 'reference': args.reference, 'oof_miou': float(scores.mean()),
              'improved_images': int((delta > 0).sum()), 'worse_images': int((delta < 0).sum()),
              'unchanged_images': int((delta == 0).sum()),
              'uniform_predictions': int((pred.min((1, 2)) == pred.max((1, 2))).sum()),
              'figure_selection': 'One seeded random image per method/full_watermark stratum, seed20261127',
              'use': 'Descriptive only; no thresholds, models or submissions selected by these diagnostics', 'strata': []}
    rng = np.random.default_rng(20261127)
    chosen = []
    for method in METHODS:
        for full in (False, True):
            ix = [i for i, n in enumerate(names) if by_name[n]['method'] == method and by_name[n]['full_watermark'] == full]
            report['strata'].append({'method': method, 'full_watermark': full, 'images': len(ix),
                                     'miou': float(scores[ix].mean()), 'paired_gain': float(delta[ix].mean())})
            chosen.append(int(rng.choice(ix)))
    sheet = Image.new('RGB', (768, 28 + 286 * len(chosen)), 'white')
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(('Image', 'Ground truth', args.candidate + ' OOF')):
        draw.text((column * 256 + 4, 6), title, fill='black')
    for row, index in enumerate(chosen):
        name = names[index]
        y = 28 + row * 286
        draw.text((4, y + 3), f"{by_name[name]['method']} full={by_name[name]['full_watermark']}  mIoU {scores[index]:.3f}  gain {delta[index]:+.3f}", fill='black')
        with Image.open(data / 'images' / (name + '.png')) as source:
            sheet.paste(source.convert('RGB').resize((256, 256)), (0, y + 26))
        for column, mask in enumerate((truth[name], candidate[name]), 1):
            tile = Image.fromarray(mask.astype(np.uint8) * 255).resize((256, 256), Image.Resampling.NEAREST)
            sheet.paste(tile.convert('RGB'), (column * 256, y + 26))
    sheet.save(root / 'oof_examples.jpg', quality=88)
    write_json(root / 'review.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
