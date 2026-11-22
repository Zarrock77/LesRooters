"""Frozen improvement gate for the next bounded experiment batch.

All five paired fold gains must be positive, mean gain at least .002, and
the paired bootstrap 99% lower bound above zero. No public score is read.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from baseline import METHODS, write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    ap.add_argument('--candidate',required=True)
    ap.add_argument('--reference',default='pretrained',help='OOF incumbent to beat; use adobe after accepted submission 1337.')
    args=ap.parse_args(); root=args.kit_root/'output'; folder=root/('cv_'+args.candidate+'_v1')
    reference_root=root/('cv_'+args.reference+'_v1')
    scores=json.loads((folder/'image_scores.json').read_text()); reference=json.loads((reference_root/'image_scores.json').read_text())
    protocol=json.loads((folder/'protocol.json').read_text()); folds=protocol['folds']
    assert folds==json.loads((reference_root/'protocol.json').read_text())['folds']
    assert set(scores)==set(reference) and len(scores)==510
    names=sorted(scores); values=np.array([scores[n] for n in names]); delta=values-np.array([reference[n] for n in names])
    rng=np.random.default_rng(20261126); boots=delta[rng.integers(0,510,size=(20000,510))].mean(1)
    ci=np.quantile(boots,[.005,.995]); gain=[float(np.mean([scores[n]-reference[n] for n in f])) for f in folds]
    rows=json.loads((args.kit_root/'release/train/metadata.json').read_text())
    metrics={'oof_miou':float(values.mean()),'paired_gain':float(delta.mean()),'paired_gain_99pct':ci.tolist(),
        'fold_miou':[float(np.mean([scores[n] for n in f])) for f in folds],'fold_gain':gain,
        'methods':{m:float(np.mean([scores[r['image_name']] for r in rows if r['method']==m])) for m in METHODS}}
    decision={'candidate':args.candidate,'reference':args.reference,'export_recommended':bool(min(gain)>0 and ci[0]>0 and delta.mean()>=.002),
        'rule':'All five fold gains positive, paired bootstrap 99% lower bound positive, mean gain >= 0.002.',
        'caveat':'Conditional image bootstrap. Reused development data; not an untouched final test.',**metrics}
    write_json(folder/'replacement_decision.json',decision)
    write_json(folder/'metrics.json',{'candidates':{args.candidate:metrics},'comparison':args.reference+' incumbent','confidence_level':.99})
    print(json.dumps(decision),flush=True)


if __name__=='__main__':main()
