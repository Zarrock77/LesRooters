"""Test message-conditioned watermark residual matching and pixel classification.

Compare re-embedding the known message with re-embedding its complement. Fit
only on the fixed inner training split; use inner tune for model selection.
"""
import argparse
import gc
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from baseline import METHODS, message_scores, image_scores, write_json
from localization_v2 import read_image, choose_thresholds


def residual_features(image, good, bad, global_scores):
    x=torch.from_numpy(image.copy()).permute(2,0,1)[None].float().cuda()/255
    residual=torch.from_numpy((good.astype(np.float32)-bad.astype(np.float32))/255).permute(2,0,1)[None].cuda()
    features=[]
    for scale in (3,7,15):
        hx=x-F.avg_pool2d(x,scale,1,scale//2)
        hr=residual-F.avg_pool2d(residual,scale,1,scale//2)
        xx=(hx*hx).mean(1,keepdim=True); rr=(hr*hr).mean(1,keepdim=True); xr=(hx*hr).mean(1,keepdim=True)
        for window in (9,21,41):
            def smooth(v): return F.avg_pool2d(v,window,1,window//2,count_include_pad=False)
            cov=smooth(xr); vx=smooth(xx); vr=smooth(rr)
            features += [cov/(vx*vr+1e-12).sqrt(),(cov/(vr+1e-7)).clamp(-5,5)/5]
        features += [(F.avg_pool2d(xx,21,1,10)+1e-8).log()/10]
    f=F.interpolate(torch.cat(features,1),(128,128),mode="bilinear",align_corners=False)[0]
    g=global_scores.cuda()[:,None,None].expand(-1,128,128)
    return torch.cat((f,g),0).cpu().numpy().astype(np.float16)


def extract(args,rows):
    from wmloc.watermarks import load_watermark
    output={}; folder=args.output_root/"features"; folder.mkdir(exist_ok=True)
    for method in METHODS:
        subset=[r for r in rows if r["method"]==method]; names=[r["image_name"] for r in subset]
        path=folder/(method+".npz")
        if path.exists():
            with np.load(path) as z:
                assert z["names"].tolist()==names; output.update(zip(names,z["maps"]))
            continue
        if method=="stegastamp":
            import onnxruntime as ort; ort.preload_dlls()
        wm=load_watermark(method,models_dir=args.kit_root/"models",device="cuda")
        features=[]
        with torch.inference_mode():
            for i,r in enumerate(subset):
                image=read_image(args.kit_root/"release","train",r["image_name"])
                bits=[int(b) for b in r["message"]]
                good=wm.embed(image,bits); bad=wm.embed(image,[1-b for b in bits])
                x=torch.from_numpy(image.copy()).permute(2,0,1)[None].float()/255
                global_scores=message_scores(wm.logits(x),torch.tensor([bits]))[0]
                features.append(residual_features(image,good,bad,global_scores))
                if i%20==0: print("MATCHED_FEATURES",method,i+1,"/",len(subset),flush=True)
        values=np.stack(features); assert values.shape==(len(subset),23,128,128) and np.isfinite(values).all()
        np.savez_compressed(path,names=np.array(names),maps=values)
        output.update(zip(names,values)); del wm,values,features; gc.collect(); torch.cuda.empty_cache()
    return output


def fit_pixel_model(features,masks,fit,tune,method,out):
    rng=np.random.default_rng(20261122)
    xs=[]; ys=[]
    for r in fit:
        n=r["image_name"]; ix=rng.choice(128*128,1024,replace=False)
        xs.append(features[n].reshape(23,-1)[:,ix].T); ys.append(masks[n].ravel()[ix])
    x=torch.from_numpy(np.concatenate(xs).astype(np.float32)).cuda()
    y=torch.from_numpy(np.concatenate(ys).astype(np.float32)).cuda()
    mean=x.mean(0); std=x.std(0).clamp_min(.01); x=(x-mean)/std
    net=nn.Sequential(nn.Linear(23,32),nn.SiLU(),nn.Linear(32,16),nn.SiLU(),nn.Linear(16,1)).cuda()
    opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.001)
    best={"miou":-1}
    def predict(rows):
        result=[]
        with torch.inference_mode():
            for r in rows:
                z=torch.from_numpy(features[r["image_name"]].astype(np.float32)).cuda().flatten(1).T
                result.append(net((z-mean)/std).sigmoid().reshape(128,128).cpu().numpy())
        return np.stack(result)
    for step in range(1,801):
        ix=torch.randint(len(x),(4096,),device="cuda")
        opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(net(x[ix])[:,0],y[ix]); loss.backward(); opt.step()
        if step%200==0:
            prob=predict(tune); score=choose_thresholds(prob,tune,masks)[method]
            if score["miou"]>best["miou"]:
                best={**score,"step":step}
                torch.save({"state_dict":net.state_dict(),"mean":mean,"std":std,"selection":best},out/("best_"+method+".pt"))
            print("MATCHED_TUNE",method,step,score,flush=True)
    checkpoint=torch.load(out/("best_"+method+".pt"),weights_only=True)
    net.load_state_dict(checkpoint["state_dict"])
    return predict,best


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit-root",type=Path,required=True); ap.add_argument("--baseline-root",type=Path,required=True)
    ap.add_argument("--output-root",type=Path,required=True); args=ap.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True); sys.path.insert(0,str(args.kit_root))
    torch.set_num_threads(4); torch.manual_seed(20261122)
    rows=json.loads((args.kit_root/"release/train/metadata.json").read_text())
    split=json.loads((args.baseline_root/"split.json").read_text())
    with np.load(args.kit_root/"release/train/masks_128.npz") as z: masks=dict(zip(z["names"].tolist(),z["masks"]))
    features=extract(args,rows); results={}; predictions={}; probabilities={}
    for method in METHODS:
        calibration=[r for r in rows if r["method"]==method and r["image_name"] in split["train"]]
        tune_names=[]
        for full in (False,True):
            names=sorted(r["image_name"] for r in calibration if r["full_watermark"]==full)
            random.Random(20261122).shuffle(names); tune_names.extend(names[:14])
        fit=[r for r in calibration if r["image_name"] not in tune_names]
        tune=[r for r in calibration if r["image_name"] in tune_names]
        validation=[r for r in rows if r["method"]==method and r["image_name"] in split["validation"]]
        predict,best=fit_pixel_model(features,masks,fit,tune,method,args.output_root)
        prob=predict(validation); pred=prob>=best["threshold"]
        score=float(image_scores(pred,np.stack([masks[r["image_name"]] for r in validation])).mean())
        results[method]={"validation_miou":score,"inner_selection":best}
        for r,p,v in zip(validation,pred,prob): predictions[r["image_name"]]=p; probabilities[r["image_name"]]=v
    report={"methods":results,"validation_miou":float(np.mean([v["validation_miou"] for v in results.values()])),
            "fit_images":324,"inner_tune_images":84,"validation_images":102}
    write_json(args.output_root/"metrics.json",report)
    names=sorted(predictions)
    np.savez_compressed(args.output_root/"validation.npz",names=np.array(names),masks=np.stack([predictions[n] for n in names]),probabilities=np.stack([probabilities[n] for n in names]))
    print("MATCHED_RESULTS",json.dumps(report),flush=True)


if __name__=="__main__": main()
