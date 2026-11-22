"""Frozen official Adobe region detector, with no hackathon-label fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image


def apply_prior(probabilities,rows,records,alpha):
    output=probabilities.copy()
    if alpha==0:return output
    for i,row in enumerate(rows):
        if row['method']!='trustmark':continue
        candidates=records[row['image_name']]
        if not candidates:continue
        best=max(candidates,key=lambda r:r['score'])
        if best['score']<.5:continue
        x1,y1,x2,y2=np.clip(np.array(best['box'])/4,0,128)
        # Rasterize using pixel centers, without fitting box coordinates to labels.
        x=np.arange(128)+.5;y=x[:,None]
        prior=((x>=x1)&(x<x2)&(y>=y1)&(y<y2)).astype(np.float32)
        weight=alpha*best['score'];output[i]=(1-weight)*output[i]+weight*prior
    return output


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kit-root',type=Path,required=True)
    args=ap.parse_args();root=args.kit_root;models=root/'models/trustmark_bbox';out=root/'output/bbox_features_v1';out.mkdir(exist_ok=True)
    sys.path.insert(0,str(models/'source/python'))
    from trustmark.yolo_wo_decoder_multibox import fasterRCNN
    torch.set_num_threads(4)
    checkpoint=models/'trustmark_bbox_Q.ckpt';assert hashlib.md5(checkpoint.read_bytes()).hexdigest()=='9d15428a33e15140ea16aa378416d304'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if 'state_dict' in state:state=state['state_dict']
    filtered={k[5:]:v for k,v in state.items() if k.startswith('yolo.')}
    assert filtered,'No detector weights found'
    model=fasterRCNN(num_classes=3);model.load_state_dict(filtered,strict=True);model=model.cuda().eval()
    for split in ('train','test'):
        path=out/(split+'.json')
        if path.exists():continue
        rows=json.loads((root/'release'/split/'metadata.json').read_text());rows=[r for r in rows if r['method']=='trustmark'];records={}
        with torch.inference_mode():
            for start in range(0,len(rows),4):
                batch=rows[start:start+4];images=[]
                for r in batch:
                    with Image.open(root/'release'/split/'images'/(r['image_name']+'.png')) as im:x=np.array(im.convert('RGB'))
                    images.append(torch.from_numpy(x).permute(2,0,1).float().cuda()/127.5-1)
                preds,_=model(images)
                for row,boxes,scores,labels in zip(batch,preds['boxes'],preds['scores'],preds['labels']):
                    records[row['image_name']]=[{'box':b,'score':float(s),'label':int(l)} for b,s,l in zip(boxes.cpu().tolist(),scores.cpu().tolist(),labels.cpu().tolist()) if s>0 and l!=0]
                if start%40==0:print('BBOX_FEATURES',split,min(start+4,len(rows)),flush=True)
        path.write_text(json.dumps(records)+'\n')
    print('BBOX_FEATURES_READY',flush=True)


if __name__=='__main__':main()
