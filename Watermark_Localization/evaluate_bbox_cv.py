"""Choose an Adobe-box prior weight on inner tune only; outer folds stay held out.

Fixed grid alpha in {0,.25,.5,.75,1}; no box-coordinate tuning. Other methods
remain identical. TrustMark detector confidence scales the selected alpha.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import localization_v2 as experiment
from baseline import METHODS, image_scores, write_json
from bbox_features import apply_prior
from compare_cv import calibrated_probability


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kit-root',type=Path,required=True)
    args=ap.parse_args();root=args.kit_root;data=root/'release';source=root/'output/cv_pretrained_v1';out=root/'output/cv_bbox_v1';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(root));torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    from pretrained_segmenter import PretrainedSegmenter
    protocol=json.loads((source/'protocol.json').read_text());protocol.update(architecture='bbox',alpha_grid=[0,.25,.5,.75,1],selection='Inner tune only; frozen Adobe box coordinates and confidence; TrustMark only.')
    write_json(out/'protocol.json',protocol)
    rows=json.loads((data/'train/metadata.json').read_text());by_name={r['image_name']:r for r in rows}
    with np.load(data/'train/masks_128.npz') as z:truth=dict(zip(z['names'].tolist(),z['masks']))
    images={r['image_name']:experiment.read_image(data,'train',r['image_name']) for r in rows if r['method']=='trustmark'}
    guides=experiment.guidance(argparse.Namespace(kit_root=root,data_root=data,output_root=root/'output/localization_v2'),rows,'train')
    boxes=json.loads((root/'output/bbox_features_v1/train.json').read_text());model=PretrainedSegmenter().cuda();all_scores={}
    for fold in range(5):
        origin=source/('fold_'+str(fold));folder=out/origin.name;folder.mkdir(exist_ok=True)
        split=json.loads((origin/'inner_split.json').read_text());write_json(folder/'inner_split.json',split)
        assert not (set(split['fit'])|set(split['tune']))&set(split['validation'])
        selection=json.loads((origin/'selection.json').read_text());new_selection={}
        for method in METHODS:
            target=origin/('best_'+method+'.pt');link=folder/target.name
            if not link.exists():link.symlink_to(target)
            alpha=0.;t=selection[method]['threshold'];tune_scores=[]
            if method=='trustmark':
                ckpt=torch.load(target,weights_only=True,map_location='cpu');model.load_state_dict(ckpt['state_dict'])
                tune=[r for r in rows if r['image_name'] in split['tune'] and r['method']==method]
                p=calibrated_probability(experiment.predict(model,experiment.Images(tune,images,guides)),t)
                gt=np.stack([truth[r['image_name']] for r in tune])
                tune_scores=[float(image_scores(apply_prior(p,tune,boxes,a)>=.5,gt).mean()) for a in protocol['alpha_grid']]
                alpha=protocol['alpha_grid'][int(np.argmax(tune_scores))]
            new_selection[method]={'threshold':.5,'source_threshold':t,'alpha':alpha,'inner_tune_scores':tune_scores}
        write_json(folder/'selection.json',new_selection)
        with np.load(origin/'validation.npz') as z:
            names=z['names'].tolist();p=np.stack([calibrated_probability(q,selection[by_name[n]['method']]['threshold']) for n,q in zip(names,z['probabilities'])])
        val=[by_name[n] for n in names];p=apply_prior(p,val,boxes,new_selection['trustmark']['alpha']);masks=p>=.5
        scores=image_scores(masks,np.stack([truth[n] for n in names]));all_scores.update(zip(names,scores.tolist()))
        np.savez_compressed(folder/'validation.npz',names=np.array(names),masks=masks,probabilities=p)
        print('BBOX_FOLD_COMPLETE',fold,'alpha',new_selection['trustmark']['alpha'],'miou',float(scores.mean()),flush=True)
    write_json(out/'image_scores.json',all_scores)


if __name__=='__main__':main()
