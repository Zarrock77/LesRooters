"""Five-fold nested model selection, frozen before inspecting fold results.

This is a robustness check after exploratory architecture development, not an
untouched final test. Source identities are absent; audited pHash groups are used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

import localization_v2 as experiment
from baseline import image_scores, write_json, METHODS


def baseline_fold(rows, truth, validation, feature_root):
    """Refit the original baseline on each outer training partition only."""
    measured={}
    for method in METHODS:
        with np.load(feature_root/(method+'.npz')) as z:
            names=z['names'].tolist(); maps=z['maps']
        gt=np.stack([truth[n] for n in names])
        va=np.array([n in validation for n in names]); tr=~va
        prior=np.broadcast_to(gt[tr].mean(0),gt.shape)
        best_score=-1; best_pred=None
        candidates=[(np.zeros_like(gt,dtype=float),[.5]),(np.ones_like(gt,dtype=float),[.5]),
                    (prior,np.linspace(.05,.95,37)),(maps[:,0],np.linspace(.30,.95,66)),
                    (maps[:,1],np.linspace(.30,.95,66))]
        for values,thresholds in candidates:
            for threshold in thresholds:
                score=float(image_scores(values[tr]>=threshold,gt[tr]).mean())
                if score>best_score:
                    best_score=score; best_pred=values[va]>=threshold
        measured.update(zip(np.array(names)[va].tolist(),image_scores(best_pred,gt[va]).tolist()))
    return measured


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    ap.add_argument('--baseline-root',type=Path,required=True)
    ap.add_argument('--guidance-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    ap.add_argument('--architecture',choices=('residual','pretrained','detector','adobe'),required=True)
    args=ap.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    metadata=args.kit_root/'release/train/metadata.json'
    rows=json.loads(metadata.read_text())
    original=json.loads((args.baseline_root/'split.json').read_text())
    assert hashlib.sha256(metadata.read_bytes()).hexdigest()==original['metadata_sha256']
    # Current audit found 510 singleton groups. Fail rather than split any future duplicate group.
    assert len(original['groups'])==510 and all(len(g)==1 for g in original['groups'])
    strata=defaultdict(list)
    for r in rows: strata[(r['method'],r['full_watermark'])].append(r['image_name'])
    folds=[[] for _ in range(5)]
    for key in sorted(strata):
        names=sorted(strata[key]); random.Random(20261123).shuffle(names)
        for i,name in enumerate(names): folds[i%5].append(name)
    assert all(len(f)==102 for f in folds)
    all_names={r['image_name'] for r in rows}
    assert set.union(*(set(f) for f in folds))==all_names
    epochs=60 if args.architecture=='residual' else 120
    adobe_protocol=None
    if args.architecture=='adobe':
        adobe_protocol=json.loads(Path(__file__).with_name('adobe_experiment.json').read_text())
        epochs=adobe_protocol['epochs']
    protocol={'folds':folds,'seed':20261123,'architecture':args.architecture,'epochs':epochs,
              'inner_selection':'324 fit / 84 tune, checkpoint and threshold selected on tune only',
              'outer_validation':'102 per fold; each image predicted exactly once by a model that did not train or tune on it',
              'limitation':'Architectures developed on this dataset before CV; not a pristine final test. Source IDs absent.',
              'grouping':original['grouping'],'metadata_sha256':original['metadata_sha256']}
    if adobe_protocol is not None:
        protocol['learned_adobe_extension']=adobe_protocol
        protocol['adobe_provenance']=json.loads((args.kit_root/'models/trustmark_bbox/provenance.json').read_text())
    protocol_path=args.output_root/'protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text())==protocol
    else: write_json(protocol_path,protocol)
    if args.architecture=='pretrained':
        from pretrained_segmenter import PretrainedSegmenter
        experiment.ResidualSegmenter=PretrainedSegmenter
    elif args.architecture=='detector':
        from detector_segmenter import DetectorSegmenter, detection_guidance
        experiment.ResidualSegmenter=DetectorSegmenter
        experiment.guidance=detection_guidance
    elif args.architecture=='adobe':
        from functools import partial
        from adobe_segmenter import AdobeSegmenter
        experiment.ResidualSegmenter=partial(AdobeSegmenter,args.kit_root)
    scores={}; probabilities={}; predictions={}; summaries=[]; baseline_scores={}
    with np.load(args.kit_root/'release/train/masks_128.npz') as z:
        truth=dict(zip(z['names'].tolist(),z['masks'].astype(bool)))
    for index,val in enumerate(folds):
        folder=args.output_root/('fold_'+str(index)); folder.mkdir(exist_ok=True)
        split_dir=folder/'split'; split_dir.mkdir(exist_ok=True)
        split={**original,'train':sorted(all_names-set(val)),'validation':sorted(val),'seed':20261123}
        write_json(split_dir/'split.json',split)
        baseline_path=folder/'baseline_image_scores.json'
        if not baseline_path.exists():
            write_json(baseline_path,baseline_fold(rows,truth,set(val),args.baseline_root/'features'))
        baseline_scores.update(json.loads(baseline_path.read_text()))
        link=folder/'guidance'
        if not link.exists(): link.symlink_to(args.guidance_root/'guidance',target_is_directory=True)
        if not (folder/'metrics.json').exists():
            sys.argv=['localization_v2.py','--kit-root',str(args.kit_root),'--baseline-root',str(split_dir),
                      '--output-root',str(folder),'--epochs',str(epochs),'--seed',str(20261122+index)]
            experiment.main()
        result=json.loads((folder/'metrics.json').read_text()); summaries.append(result)
        with np.load(folder/'validation.npz') as z:
            ns=z['names'].tolist(); assert set(ns)==set(val)
            measured=image_scores(z['masks'].astype(bool),np.stack([truth[n] for n in ns]))
            scores.update(zip(ns,measured.tolist())); predictions.update(zip(ns,z['masks']))
            probabilities.update(zip(ns,z['probabilities']))
        print('FOLD_COMPLETE',index,result['validation_miou'],flush=True)
        torch.cuda.empty_cache()
    names=sorted(all_names)
    values=np.array([scores[n] for n in names])
    rng=np.random.default_rng(20261123)
    boots=values[rng.integers(0,510,size=(10000,510))].mean(1)
    baseline_values=np.array([baseline_scores[n] for n in names]); differences=values-baseline_values
    paired=differences[rng.integers(0,510,size=(10000,510))].mean(1)
    report={'architecture':args.architecture,'oof_miou':float(values.mean()),
            'baseline_oof_miou':float(baseline_values.mean()),'paired_gain':float(differences.mean()),
            'paired_gain_bootstrap_95pct_interval':np.quantile(paired,[.025,.975]).tolist(),
            'image_bootstrap_95pct_interval':np.quantile(boots,[.025,.975]).tolist(),
            'interval_caveat':'Conditional on these trained folds; does not include all training or architecture-selection uncertainty.',
            'fold_miou':[s['validation_miou'] for s in summaries],
            'methods':{m:float(np.mean([scores[r['image_name']] for r in rows if r['method']==m])) for m in METHODS},
            'images':510,'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    write_json(args.output_root/'metrics.json',report)
    write_json(args.output_root/'image_scores.json',scores)
    write_json(args.output_root/'baseline_image_scores.json',baseline_scores)
    np.savez_compressed(args.output_root/'oof.npz',names=np.array(names),masks=np.stack([predictions[n] for n in names]),
                        probabilities=np.stack([probabilities[n] for n in names]))
    print('CV_RESULTS',json.dumps(report),flush=True)


if __name__=='__main__': main()
