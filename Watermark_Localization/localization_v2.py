"""Supervised residual segmentation with frozen watermark-decoder guidance.

Model/threshold selection uses an inner split of the 408 calibration images.
The original 102-image validation set is evaluated only after training.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from baseline import METHODS, message_scores, image_scores, write_json

FEATURE_VERSION = "guidance-native-512-global-v2"


def read_image(data, split, name):
    with Image.open(data / split / "images" / (name + ".png")) as im:
        return np.array(im.convert("RGB"))


def guidance(args, rows, split_name):
    from wmloc.watermarks import load_watermark
    destination = args.output_root / "guidance" / split_name
    destination.mkdir(parents=True, exist_ok=True)
    all_maps = {}
    for method in METHODS:
        subset = [r for r in rows if r["method"] == method]
        names = [r["image_name"] for r in subset]
        path = destination / (method + ".npz")
        if path.exists():
            with np.load(path) as z:
                assert z["names"].tolist() == names and z["version"].item() == FEATURE_VERSION
                all_maps.update(zip(names, z["maps"]))
            continue
        print("GUIDANCE_LOAD", method, split_name, flush=True)
        if method == "stegastamp":
            import onnxruntime as ort
            ort.preload_dlls()
        wm = load_watermark(method, models_dir=args.kit_root / "models", device="cuda")
        if method == "stegastamp":
            assert "CUDAExecutionProvider" in wm.sess.get_providers()
        output = []
        with torch.inference_mode():
            for start in range(0, len(subset), 4):
                batch = subset[start:start+4]
                x = torch.from_numpy(np.stack([read_image(args.data_root, split_name, r["image_name"]) for r in batch])).permute(0,3,1,2).float()/255
                bits = torch.tensor([[int(b) for b in r["message"]] for r in batch])
                if method == "stegastamp":
                    global_scores = message_scores(wm.logits(x), bits).to("cuda")
                    # The global decoder is aligned to the full image; arbitrary crops destroy its geometry.
                    maps = global_scores.repeat(1,3)[:,:,None,None].expand(-1,-1,128,128)
                else:
                    original_resolution = wm.resolution
                    logits = wm.features(x)["per_position_logits"]
                    global_scores = message_scores(logits.mean((-1,-2)), bits)
                    native = F.interpolate(message_scores(logits,bits), (128,128), mode="bilinear", align_corners=False)
                    wm.resolution = 512
                    high = F.interpolate(message_scores(wm.features(x)["per_position_logits"],bits), (128,128), mode="bilinear", align_corners=False)
                    wm.resolution = original_resolution
                    maps = torch.cat((native, high, global_scores[:,:,None,None].expand(-1,-1,128,128)), 1)
                output.append(maps.cpu().numpy().astype(np.float16))
                if start % 80 == 0:
                    print("GUIDANCE", method, split_name, min(start+4,len(subset)), "/", len(subset), flush=True)
        values = np.concatenate(output)
        assert values.shape == (len(subset),6,128,128) and np.isfinite(values).all()
        np.savez_compressed(path, names=np.array(names), maps=values, version=np.array(FEATURE_VERSION))
        all_maps.update(zip(names,values))
        del wm, values, output, maps
        gc.collect(); torch.cuda.empty_cache()
    return all_maps


def block(cin, cout, stride=1):
    return nn.Sequential(nn.Conv2d(cin,cout,3,stride,1,bias=False),nn.GroupNorm(8,cout),nn.SiLU(),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.SiLU())


class ResidualSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = block(48,16)
        self.down = block(16,32,2)
        self.fuse = block(41,32)
        self.e2 = block(32,48,2); self.e3 = block(48,64,2); self.e4 = block(64,96,2)
        self.d3 = block(96+64,64); self.d2 = block(64+48,48); self.d1 = block(48+32,32)
        self.head = nn.Conv2d(32,1,1)

    def forward(self, image, guide, method):
        residuals = [(image-F.avg_pool2d(image,k,1,k//2))*8 for k in (3,7,15)]
        x = torch.cat([image-.5]+residuals,1)
        x = self.down(self.stem(F.pixel_unshuffle(x,2)))
        condition = F.one_hot(method,3).to(x.dtype)[:,:,None,None].expand(-1,-1,128,128)
        e1 = self.fuse(torch.cat((x,(guide-.5)*4,condition),1))
        e2=self.e2(e1); e3=self.e3(e2); e4=self.e4(e3)
        def up(a,b):
            return torch.cat((F.interpolate(a,b.shape[-2:],mode="bilinear",align_corners=False),b),1)
        return self.head(self.d1(up(self.d2(up(self.d3(up(e4,e3)),e2)),e1)))[:,0]


class Images(Dataset):
    def __init__(self, rows, images, maps, masks=None, augment=False):
        self.rows,self.images,self.maps,self.masks,self.augment=rows,images,maps,masks,augment

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]; name=row["image_name"]
        image=torch.from_numpy(self.images[name].copy()).permute(2,0,1).float()/255
        guide=torch.from_numpy(self.maps[name].astype(np.float32))
        mask=torch.from_numpy(self.masks[name].astype(np.float32)) if self.masks is not None else torch.zeros((128,128))
        if self.augment:
            k=random.randrange(4)
            image=torch.rot90(image,k,(-2,-1)); guide=torch.rot90(guide,k,(-2,-1)); mask=torch.rot90(mask,k,(-2,-1))
            if random.random()<.5:
                image=image.flip(-1); guide=guide.flip(-1); mask=mask.flip(-1)
        return image,guide,METHODS.index(row["method"]),mask


def loss_fn(logits, truth):
    bce=F.binary_cross_entropy_with_logits(logits,truth)
    p=logits.sigmoid()
    fg=(2*(p*truth).sum((-1,-2))+1)/(p.sum((-1,-2))+truth.sum((-1,-2))+1)
    bg=(2*((1-p)*(1-truth)).sum((-1,-2))+1)/((1-p).sum((-1,-2))+(1-truth).sum((-1,-2))+1)
    return bce+(1-(fg+bg)/2).mean()


@torch.inference_mode()
def predict(model, dataset, tta=False):
    model.eval(); output=[]
    loader=DataLoader(dataset,batch_size=8,num_workers=0,pin_memory=True)
    for x,g,m,_ in loader:
        x,g,m=x.cuda(non_blocking=True),g.cuda(non_blocking=True),m.cuda(non_blocking=True)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            p=model(x,g,m).float().sigmoid()
            if tta:
                q=model(x.flip(-1),g.flip(-1),m).float().sigmoid().flip(-1)
                r=model(x.flip(-2),g.flip(-2),m).float().sigmoid().flip(-2)
                p=(p+q+r)/3
        output.append(p.cpu().numpy())
    return np.concatenate(output)


def choose_thresholds(prob, rows, masks):
    result={}
    for method in METHODS:
        ix=[i for i,r in enumerate(rows) if r["method"]==method]
        if not ix: continue
        gt=np.stack([masks[rows[i]["image_name"]] for i in ix])
        candidates=np.linspace(.2,.8,13)
        scores=[float(image_scores(prob[ix]>=t,gt).mean()) for t in candidates]
        best=int(np.argmax(scores))
        result[method]={"threshold":float(candidates[best]),"miou":scores[best]}
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit-root",type=Path,required=True)
    ap.add_argument("--baseline-root",type=Path,required=True)
    ap.add_argument("--output-root",type=Path,required=True)
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--seed",type=int,default=20261122)
    args=ap.parse_args(); args.data_root=args.kit_root/"release"
    args.output_root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(args.kit_root))
    from wmloc.utils import miou_image
    torch.set_num_threads(4); torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.backends.cudnn.benchmark=True
    rows=json.loads((args.data_root/"train/metadata.json").read_text())
    split=json.loads((args.baseline_root/"split.json").read_text())
    with np.load(args.data_root/"train/masks_128.npz") as z: masks=dict(zip(z["names"].tolist(),z["masks"]))
    images={r["image_name"]:read_image(args.data_root,"train",r["image_name"]) for r in rows}
    maps=guidance(args,rows,"train")
    groups=defaultdict(list)
    for r in rows:
        if r["image_name"] in split["train"]: groups[(r["method"],r["full_watermark"])].append(r["image_name"])
    tune=[]
    for k in sorted(groups):
        names=sorted(groups[k]); random.Random(args.seed).shuffle(names); tune.extend(names[:14])
    fit=[r for r in rows if r["image_name"] in split["train"] and r["image_name"] not in tune]
    tune_rows=[r for r in rows if r["image_name"] in tune]
    validation=[r for r in rows if r["image_name"] in split["validation"]]
    assert len(fit)==324 and len(tune_rows)==84 and len(validation)==102
    write_json(args.output_root/"inner_split.json",{"fit":[r["image_name"] for r in fit],"tune":tune,"validation":split["validation"],"seed":args.seed})
    train_data=Images(fit,images,maps,masks,True); tune_data=Images(tune_rows,images,maps,masks)
    train_loader=DataLoader(train_data,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,persistent_workers=True)
    model=ResidualSegmenter().cuda()
    if hasattr(model,'configure_optimization'):
        optimizer,scheduler=model.configure_optimization(args.epochs)
    else:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=.0001)
    best={m:{"miou":-1} for m in METHODS}; history=[]
    print("TRAINING",len(fit),"TUNE",len(tune_rows),"PARAMETERS",sum(p.numel() for p in model.parameters()),flush=True)
    for epoch in range(1,args.epochs+1):
        start=time.time(); model.train(); total=0
        for x,g,m,y in train_loader:
            x,g,m,y=x.cuda(non_blocking=True),g.cuda(non_blocking=True),m.cuda(non_blocking=True),y.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): loss=loss_fn(model(x,g,m).float(),y)
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step()
            total+=float(loss.detach())
        scheduler.step()
        record={"epoch":epoch,"loss":total/len(train_loader),"seconds":time.time()-start}
        if epoch==1 or epoch%5==0:
            probs=predict(model,tune_data)
            scores=choose_thresholds(probs,tune_rows,masks); record["tune"]=scores
            for method,s in scores.items():
                if s["miou"]>best[method]["miou"]:
                    best[method]={**s,"epoch":epoch}
                    torch.save({"state_dict":model.state_dict(),"selection":best[method],"method":method},args.output_root/("best_"+method+".pt"))
            write_json(args.output_root/"selection.json",best)
        history.append(record); write_json(args.output_root/"history.json",history)
        print("EPOCH",json.dumps(record),flush=True)
    # Evaluate the outer validation only after the checkpoint decisions are complete.
    measured={}; all_predictions={}; all_probs={}
    for method in METHODS:
        checkpoint=torch.load(args.output_root/("best_"+method+".pt"),weights_only=True,map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        subset=[r for r in validation if r["method"]==method]
        probs=predict(model,Images(subset,images,maps,masks))
        pred=probs>=best[method]["threshold"]
        scores=[miou_image(p,masks[r["image_name"]]) for p,r in zip(pred,subset)]
        measured[method]={"validation_miou":float(np.mean(scores)),"n":len(subset),"inner_selection":best[method]}
        for r,p,v in zip(subset,pred,probs): all_predictions[r["image_name"]]=p; all_probs[r["image_name"]]=v
    report={"methods":measured,"validation_miou":float(np.mean([v["validation_miou"] for v in measured.values()])),
            "baseline_validation_miou":.42147506656998046,"fit_images":324,"inner_tune_images":84,"validation_images":102,
            "checkpoint_selection":"Inner tune only; original validation evaluated after training.","epochs":args.epochs,"seed":args.seed}
    write_json(args.output_root/"metrics.json",report)
    names=sorted(all_predictions)
    np.savez_compressed(args.output_root/"validation.npz",names=np.array(names),masks=np.stack([all_predictions[n] for n in names]),probabilities=np.stack([all_probs[n] for n in names]))
    print("RESULTS",json.dumps(report),flush=True)


if __name__=="__main__": main()
