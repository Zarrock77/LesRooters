"""Measure a fixed dihedral average on existing independent outer folds."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import localization_v2 as experiment
from baseline import METHODS, image_scores, write_json
from dihedral import predict_dihedral


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    args=ap.parse_args(); root=args.kit_root; data=root/'release'
    source=root/'output/cv_pretrained_v1'; out=root/'output/cv_pretrained_tta_v1'; out.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(root)); torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    from pretrained_segmenter import PretrainedSegmenter
    protocol=json.loads((source/'protocol.json').read_text()); protocol.update(architecture='pretrained_tta',inference='Equal average of all eight dihedral views; existing inner thresholds unchanged.')
    write_json(out/'protocol.json',protocol)
    rows=json.loads((data/'train/metadata.json').read_text())
    with np.load(data/'train/masks_128.npz') as z:truth=dict(zip(z['names'].tolist(),z['masks']))
    images={r['image_name']:experiment.read_image(data,'train',r['image_name']) for r in rows}
    guides=experiment.guidance(argparse.Namespace(kit_root=root,data_root=data,output_root=root/'output/localization_v2'),rows,'train')
    model=PretrainedSegmenter().cuda(); scores={}
    for fold in range(5):
        folder=out/('fold_'+str(fold)); folder.mkdir(exist_ok=True); origin=source/folder.name
        split=json.loads((origin/'inner_split.json').read_text()); write_json(folder/'inner_split.json',split)
        write_json(folder/'selection.json',json.loads((origin/'selection.json').read_text()))
        assert not (set(split['fit'])|set(split['tune']))&set(split['validation'])
        probabilities={}
        for method in METHODS:
            path=origin/('best_'+method+'.pt'); link=folder/path.name
            if not link.exists():link.symlink_to(path)
            ckpt=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ckpt['state_dict'])
            subset=[r for r in rows if r['image_name'] in split['validation'] and r['method']==method]
            p=predict_dihedral(model,experiment.Images(subset,images,guides))
            s=image_scores(p>=ckpt['selection']['threshold'],np.stack([truth[r['image_name']] for r in subset]))
            scores.update(zip([r['image_name'] for r in subset],s.tolist()));probabilities.update(zip([r['image_name'] for r in subset],p))
        names=sorted(probabilities); selection=json.loads((folder/'selection.json').read_text()); by_name={r['image_name']:r for r in rows}
        masks=np.stack([probabilities[n]>=selection[by_name[n]['method']]['threshold'] for n in names])
        np.savez_compressed(folder/'validation.npz',names=np.array(names),masks=masks,probabilities=np.stack([probabilities[n] for n in names]))
        print('TTA_FOLD_COMPLETE',fold,float(np.mean([scores[n] for n in names])),flush=True)
    assert len(scores)==510;write_json(out/'image_scores.json',scores)
    print('TTA_COMPLETE',float(np.mean(list(scores.values()))),flush=True)


if __name__=='__main__':main()
