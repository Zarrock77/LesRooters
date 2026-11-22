"""Compare paired out-of-fold predictions, including a fixed equal-weight blend.

The blend uses checkpoint thresholds selected inside each fold. No OOF label
is used to adjust a threshold or weight. Architecture selection is still exploratory.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from baseline import image_scores, write_json, METHODS


def calibrated_probability(probability, threshold):
    p=np.clip(probability.astype(np.float32),1e-6,1-1e-6)
    t=float(threshold)
    # Shift logits so each independently tuned decision threshold maps to 0.5.
    return p*(1-t)/(p*(1-t)+(1-p)*t)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    ap.add_argument('--residual-root',type=Path,required=True)
    ap.add_argument('--pretrained-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    args=ap.parse_args(); args.output_root.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.kit_root/'release/train/metadata.json').read_text()); by_name={r['image_name']:r for r in rows}
    with np.load(args.kit_root/'release/train/masks_128.npz') as z: truth=dict(zip(z['names'].tolist(),z['masks'].astype(bool)))
    probabilities=[]; model_scores={}; folds=None
    for model_root in (args.residual_root,args.pretrained_root):
        protocol=json.loads((model_root/'protocol.json').read_text())
        if folds is None: folds=protocol['folds']
        else: assert folds==protocol['folds']
        calibrated={}; measured={}
        for i,expected in enumerate(folds):
            folder=model_root/('fold_'+str(i)); selections=json.loads((folder/'selection.json').read_text())
            with np.load(folder/'validation.npz') as z:
                names=z['names'].tolist(); assert set(names)==set(expected)
                scores=image_scores(z['masks'],np.stack([truth[n] for n in names])); measured.update(zip(names,scores.tolist()))
                for n,p in zip(names,z['probabilities']):
                    calibrated[n]=calibrated_probability(p,selections[by_name[n]['method']]['threshold'])
        probabilities.append(calibrated); model_scores[protocol['architecture']]=measured
    names=sorted(truth); blend=np.stack([(probabilities[0][n]+probabilities[1][n])/2 for n in names])
    scores=image_scores(blend>=.5,np.stack([truth[n] for n in names])); model_scores['equal_blend']=dict(zip(names,scores.tolist()))
    baseline=json.loads((args.pretrained_root/'baseline_image_scores.json').read_text())
    report={'weights':{'residual':.5,'pretrained':.5},'threshold':.5,'selection':'Fixed before seeing CV results; per-model thresholds from inner tune only.',
            'caveat':'CV follows exploratory development and is not an untouched final test. Bootstrap intervals condition on trained folds.', 'candidates':{}}
    rng=np.random.default_rng(20261123); bootstrap=rng.integers(0,510,size=(10000,510))
    for label,values in model_scores.items():
        s=np.array([values[n] for n in names]); delta=s-np.array([baseline[n] for n in names])
        report['candidates'][label]={'oof_miou':float(s.mean()),'paired_gain':float(delta.mean()),
            'paired_gain_95pct':np.quantile(delta[bootstrap].mean(1),[.025,.975]).tolist(),
            'fold_miou':[float(np.mean([values[n] for n in f])) for f in folds],
            'fold_gain':[float(np.mean([values[n]-baseline[n] for n in f])) for f in folds],
            'methods':{m:float(np.mean([values[r['image_name']] for r in rows if r['method']==m])) for m in METHODS}}
    np.savez_compressed(args.output_root/'blend_oof.npz',names=np.array(names),masks=blend>=.5,probabilities=blend)
    write_json(args.output_root/'metrics.json',report); print(json.dumps(report),flush=True)


if __name__=='__main__':main()
