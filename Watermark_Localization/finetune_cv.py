"""Conservative nested fine-tuning of the last two watermark encoder stages.

Fixed protocol: 40 epochs, head LR 2e-4, encoder LR 1e-5. Each outer fold
starts from its own pretrained checkpoints; all selection remains on inner tune.
The previous per-method checkpoint is retained unless inner mIoU improves.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import localization_v2 as experiment
from baseline import METHODS, image_scores, write_json
from pretrained_segmenter import PretrainedSegmenter


class FinetunedSegmenter(PretrainedSegmenter):
    def __init__(self):
        super().__init__()
        self.encoder.layer3.requires_grad_(True)
        self.encoder.layer4.requires_grad_(True)

    def forward(self,image,guide,method):
        n=self.encoder
        with torch.no_grad():
            x=n.relu(n.bn1(n.conv1(image*2-1))); features=[x]
            x=n.layer1(n.maxpool(x)); features.append(x)
            x=n.layer2(x); features.append(x)
        x=n.layer3(x); features.append(x)
        x=n.layer4(x); features.append(x)
        pyramids=[F.interpolate(p(f),(128,128),mode='bilinear',align_corners=False) for p,f in zip(self.project,features)]
        residuals=[(image-F.avg_pool2d(image,k,1,k//2))*8 for k in (3,7,15)]
        noise=self.noise(F.pixel_unshuffle(torch.cat([image-.5]+residuals,1),4))
        condition=F.one_hot(method,3).to(noise.dtype)[:,:,None,None].expand(-1,-1,128,128)
        return self.fuse(torch.cat(pyramids+[noise,(guide-.5)*4,condition],1))[:,0]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root',type=Path,required=True)
    ap.add_argument('--source-root',type=Path,required=True)
    ap.add_argument('--guidance-root',type=Path,required=True)
    ap.add_argument('--output-root',type=Path,required=True)
    args=ap.parse_args(); args.data_root=args.kit_root/'release'
    sys.path.insert(0,str(args.kit_root)); torch.set_num_threads(4); torch.backends.cudnn.benchmark=True
    args.output_root.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((args.source_root/'protocol.json').read_text())
    protocol.update(architecture='finetuned',fine_tune_epochs=40,head_lr=.0002,encoder_lr=.00001,
                    initialization='Own fold best_stegastamp checkpoint; own per-method checkpoints as inner-selection fallbacks')
    protocol_path=args.output_root/'protocol.json'
    if protocol_path.exists(): assert json.loads(protocol_path.read_text())==protocol
    else: write_json(protocol_path,protocol)
    rows=json.loads((args.data_root/'train/metadata.json').read_text())
    with np.load(args.data_root/'train/masks_128.npz') as z: truth=dict(zip(z['names'].tolist(),z['masks'].astype(bool)))
    images={r['image_name']:experiment.read_image(args.data_root,'train',r['image_name']) for r in rows}
    guide_args=argparse.Namespace(kit_root=args.kit_root,data_root=args.data_root,output_root=args.guidance_root)
    guides=experiment.guidance(guide_args,rows,'train'); all_scores={}; all_pred={}; all_probs={}; fold_scores=[]
    for fold in range(5):
        source=args.source_root/('fold_'+str(fold)); folder=args.output_root/('fold_'+str(fold)); folder.mkdir(exist_ok=True)
        split=json.loads((source/'inner_split.json').read_text()); write_json(folder/'inner_split.json',split)
        if not (folder/'validation.npz').exists():
            seed=20261124+fold; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            fit=[r for r in rows if r['image_name'] in split['fit']]
            tune=[r for r in rows if r['image_name'] in split['tune']]
            val=[r for r in rows if r['image_name'] in split['validation']]
            assert len(fit)==324 and len(tune)==84 and len(val)==102
            assert not (set(split['fit'])|set(split['tune']))&set(split['validation'])
            train=DataLoader(experiment.Images(fit,images,guides,truth,True),batch_size=8,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
            tuning=experiment.Images(tune,images,guides,truth)
            model=FinetunedSegmenter().cuda()
            initial=torch.load(source/'best_stegastamp.pt',weights_only=True,map_location='cpu')
            model.load_state_dict(initial['state_dict']); del initial
            best=json.loads((source/'selection.json').read_text())
            for method in METHODS: shutil.copyfile(source/('best_'+method+'.pt'),folder/('best_'+method+'.pt'))
            encoder=[p for n,p in model.named_parameters() if n.startswith('encoder.') and p.requires_grad]
            head=[p for n,p in model.named_parameters() if not n.startswith('encoder.') and p.requires_grad]
            optimizer=torch.optim.AdamW([{'params':head,'lr':.0002},{'params':encoder,'lr':.00001}],weight_decay=.0001)
            scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda epoch:.1+.9*(1+math.cos(math.pi*epoch/40))/2)
            history=[]
            for epoch in range(1,41):
                model.train(); total=0; started=time.time()
                for x,g,m,y in train:
                    x,g,m,y=x.cuda(),g.cuda(),m.cuda(),y.cuda(); optimizer.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16): loss=experiment.loss_fn(model(x,g,m).float(),y)
                    assert torch.isfinite(loss)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); total+=float(loss.detach())
                scheduler.step(); record={'epoch':epoch,'loss':total/len(train),'seconds':time.time()-started}
                if epoch==1 or epoch%5==0:
                    prob=experiment.predict(model,tuning); selection=experiment.choose_thresholds(prob,tune,truth); record['tune']=selection
                    for method,s in selection.items():
                        if s['miou']>best[method]['miou']:
                            best[method]={**s,'epoch':epoch,'phase':'fine_tuning'}
                            torch.save({'state_dict':model.state_dict(),'selection':best[method],'method':method},folder/('best_'+method+'.pt'))
                    write_json(folder/'selection.json',best)
                history.append(record); print('FINE_TUNE',fold,json.dumps(record),flush=True)
            write_json(folder/'history.json',history)
            predictions={}; probabilities={}; measured={}
            for method in METHODS:
                checkpoint=torch.load(folder/('best_'+method+'.pt'),weights_only=True,map_location='cpu'); model.load_state_dict(checkpoint['state_dict'])
                subset=[r for r in val if r['method']==method]
                prob=experiment.predict(model,experiment.Images(subset,images,guides,truth)); masks=prob>=best[method]['threshold']
                measured[method]=float(image_scores(masks,np.stack([truth[r['image_name']] for r in subset])).mean())
                for r,p,q in zip(subset,masks,prob): predictions[r['image_name']]=p; probabilities[r['image_name']]=q
            names=sorted(predictions)
            np.savez_compressed(folder/'validation.npz',names=np.array(names),masks=np.stack([predictions[n] for n in names]),probabilities=np.stack([probabilities[n] for n in names]))
            write_json(folder/'metrics.json',{'methods':measured,'validation_miou':float(np.mean(list(measured.values()))),'selection':best})
            del model,optimizer,scheduler,train,tuning,checkpoint; gc.collect(); torch.cuda.empty_cache()
        with np.load(folder/'validation.npz') as z:
            names=z['names'].tolist(); assert set(names)==set(protocol['folds'][fold])
            scores=image_scores(z['masks'],np.stack([truth[n] for n in names])); all_scores.update(zip(names,scores.tolist()))
            all_pred.update(zip(names,z['masks'])); all_probs.update(zip(names,z['probabilities'])); fold_scores.append(float(scores.mean()))
        print('FINETUNE_FOLD_COMPLETE',fold,fold_scores[-1],flush=True)
    baseline=json.loads((args.source_root/'baseline_image_scores.json').read_text()); names=sorted(truth)
    values=np.array([all_scores[n] for n in names]); delta=values-np.array([baseline[n] for n in names])
    rng=np.random.default_rng(20261124); bootstrap=delta[rng.integers(0,510,size=(10000,510))].mean(1)
    result={'oof_miou':float(values.mean()),'paired_gain':float(delta.mean()),'paired_gain_95pct':np.quantile(bootstrap,[.025,.975]).tolist(),
            'fold_miou':fold_scores,'fold_gain':[float(np.mean([all_scores[n]-baseline[n] for n in f])) for f in protocol['folds']],
            'methods':{m:float(np.mean([all_scores[r['image_name']] for r in rows if r['method']==m])) for m in METHODS}}
    report={'candidates':{'finetuned':result},'caveat':protocol['limitation'],'selection':'Fine-tuning retained only when inner tune improves; outer masks never fitted.'}
    write_json(args.output_root/'metrics.json',report); write_json(args.output_root/'image_scores.json',all_scores)
    write_json(args.output_root/'baseline_image_scores.json',baseline)
    np.savez_compressed(args.output_root/'oof.npz',names=np.array(names),masks=np.stack([all_pred[n] for n in names]),probabilities=np.stack([all_probs[n] for n in names]))
    print('FINETUNE_RESULTS',json.dumps(report),flush=True)


if __name__=='__main__':main()
