"""Diagnose GPU inference drift against stored, held-out fold predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import localization_v2 as experiment
from baseline import METHODS, image_scores, write_json


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    ap.add_argument('--fold',type=int,default=0)
    args=ap.parse_args(); sys.path.insert(0,str(args.kit_root))
    torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    from pretrained_segmenter import PretrainedSegmenter
    root=args.kit_root; data=root/'release'; folder=root/'output/cv_pretrained_v1'/('fold_'+str(args.fold))
    rows=json.loads((data/'train/metadata.json').read_text())
    with np.load(folder/'validation.npz') as z:
        reference=dict(zip(z['names'].tolist(),z['masks']))
        old_prob=dict(zip(z['names'].tolist(),z['probabilities']))
    with np.load(data/'train/masks_128.npz') as z: truth=dict(zip(z['names'].tolist(),z['masks']))
    guides=experiment.guidance(argparse.Namespace(kit_root=root,data_root=data,output_root=root/'output/localization_v2'),rows,'train')
    images={r['image_name']:experiment.read_image(data,'train',r['image_name']) for r in rows if r['image_name'] in reference}
    model=PretrainedSegmenter().cuda(); results={}
    for method in METHODS:
        ckpt=torch.load(folder/('best_'+method+'.pt'),weights_only=True,map_location='cpu'); model.load_state_dict(ckpt['state_dict'])
        threshold=ckpt['selection']['threshold']; subset=[r for r in rows if r['image_name'] in reference and r['method']==method]
        names=[r['image_name'] for r in subset]; p=experiment.predict(model,experiment.Images(subset,images,guides)); q=np.stack([old_prob[n] for n in names])
        a=p>=threshold; b=np.stack([reference[n] for n in names]); gt=np.stack([truth[n] for n in names])
        changed=(a!=b).sum((1,2)); difference=abs(p-q)
        results[method]={'threshold':threshold,'images_changed':int((changed>0).sum()),'pixels_changed':int(changed.sum()),
            'fraction_changed':float((a!=b).mean()),'max_pixels_per_image':int(changed.max()),
            'probability_mean_abs_diff':float(difference.mean()),'probability_max_abs_diff':float(difference.max()),
            'old_miou':float(image_scores(b,gt).mean()),'new_miou':float(image_scores(a,gt).mean()),
            'max_image_miou_diff':float(abs(image_scores(a,gt)-image_scores(b,gt)).max())}
    report={'fold':args.fold,'device':torch.cuda.get_device_name(),'torch':torch.__version__,'results':results}
    write_json(root/'output/cv_comparison_v1/reproduction_diagnostic.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
