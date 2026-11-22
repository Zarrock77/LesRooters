"""Require paired OOF evidence before exporting a fine-tuned replacement."""
import argparse
import json
from pathlib import Path
import numpy as np
from baseline import image_scores, write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    args=ap.parse_args(); out=args.kit_root/'output'
    previous=json.loads((out/'cv_comparison_v1/metrics.json').read_text())['candidates']
    eligible={k:v for k,v in previous.items() if min(v['fold_gain'])>0 and v['paired_gain_95pct'][0]>0}
    incumbent=max(eligible,key=lambda k:eligible[k]['oof_miou'])
    fine=json.loads((out/'cv_finetuned_v1/image_scores.json').read_text())
    if incumbent=='equal_blend':
        with np.load(args.kit_root/'release/train/masks_128.npz') as z: truth=dict(zip(z['names'].tolist(),z['masks']))
        with np.load(out/'cv_comparison_v1/blend_oof.npz') as z:
            ns=z['names'].tolist(); scores=image_scores(z['masks'],np.stack([truth[n] for n in ns])); base=dict(zip(ns,scores.tolist()))
    else: base=json.loads((out/('cv_'+incumbent+'_v1')/'image_scores.json').read_text())
    names=sorted(fine); assert set(names)==set(base)
    difference=np.array([fine[n]-base[n] for n in names]); rng=np.random.default_rng(20261125)
    ci=np.quantile(difference[rng.integers(0,510,size=(10000,510))].mean(1),[.025,.975])
    folds=json.loads((out/'cv_finetuned_v1/protocol.json').read_text())['folds']
    fold_gain=[float(np.mean([fine[n]-base[n] for n in f])) for f in folds]
    result={'incumbent':incumbent,'paired_gain':float(difference.mean()),'paired_gain_95pct':ci.tolist(),
            'fold_gain':fold_gain,'export_recommended':bool(ci[0]>0 and min(fold_gain)>0),
            'rule':'Positive paired-bootstrap lower bound and positive gain in every fold over current CV winner.',
            'caveat':'Conditional bootstrap; exploratory architectures, not a pristine final test.'}
    write_json(out/'cv_finetuned_v1/replacement_decision.json',result); print(json.dumps(result),flush=True)
    if not result['export_recommended']: raise SystemExit(3)


if __name__=='__main__':main()
