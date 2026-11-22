"""Predeclared 50/50 calibrated blend; neither weights nor thresholds fitted."""
import argparse
import json
from pathlib import Path
import numpy as np
from baseline import METHODS, image_scores, write_json
from compare_cv import calibrated_probability


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kit-root',type=Path,required=True)
    args=ap.parse_args();root=args.kit_root/'output';out=root/'cv_detector_blend_v1';out.mkdir(exist_ok=True)
    protocol=json.loads((root/'cv_pretrained_v1/protocol.json').read_text())
    assert protocol['folds']==json.loads((root/'cv_detector_v1/protocol.json').read_text())['folds']
    protocol.update(architecture='detector_blend',blend='Equal threshold-calibrated pretrained and detector probabilities, threshold 0.5; fixed in advance.')
    write_json(out/'protocol.json',protocol)
    rows=json.loads((args.kit_root/'release/train/metadata.json').read_text());methods={r['image_name']:r['method'] for r in rows}
    with np.load(args.kit_root/'release/train/masks_128.npz') as z:truth=dict(zip(z['names'].tolist(),z['masks']))
    all_scores={}
    for fold in range(5):
        name='fold_'+str(fold);folder=out/name;folder.mkdir(exist_ok=True);parts=[]
        for architecture in ('pretrained','detector'):
            origin=root/('cv_'+architecture+'_v1')/name
            selection=json.loads((origin/'selection.json').read_text())
            with np.load(origin/'validation.npz') as z:
                names=z['names'].tolist();assert set(names)==set(protocol['folds'][fold])
                parts.append({n:calibrated_probability(p,selection[methods[n]]['threshold']) for n,p in zip(names,z['probabilities'])})
        names=sorted(parts[0]);probabilities=np.stack([(parts[0][n]+parts[1][n])/2 for n in names]);masks=probabilities>=.5
        scores=image_scores(masks,np.stack([truth[n] for n in names]));all_scores.update(zip(names,scores.tolist()))
        np.savez_compressed(folder/'validation.npz',names=np.array(names),masks=masks,probabilities=probabilities)
        print('BLEND_FOLD_COMPLETE',fold,float(scores.mean()),flush=True)
    write_json(out/'image_scores.json',all_scores)


if __name__=='__main__':main()
