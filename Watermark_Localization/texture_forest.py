"""CPU alternative: local residual statistics plus decoder guidance, boosted trees.

Feature extraction and fitting can run in an isolated local CPU environment.
"""
import argparse
import json
from pathlib import Path
import random
import time
import numpy as np

METHODS=("pixelseal","trustmark","stegastamp")


def features(image,guide):
    import cv2
    x=image.astype(np.float32)/255
    channels=[]
    def small(y): return cv2.resize(y,(128,128),interpolation=cv2.INTER_AREA)
    channels.extend(small(x)[:,:,c] for c in range(3))
    for scale in (3,7,15):
        hp=x-cv2.GaussianBlur(x,(scale,scale),0)
        for window in (5,13,29):
            energy=cv2.boxFilter(hp*hp,-1,(window,window),normalize=True)
            for c in range(3): channels.append(np.log(small(energy[:,:,c])+1e-7))
            for a,b in ((0,1),(2,1)):
                chroma=hp[:,:,a]-hp[:,:,b]
                channels.append(np.log(small(cv2.boxFilter(chroma*chroma,-1,(window,window),normalize=True))+1e-7))
    med=image.astype(np.float32)-cv2.medianBlur(image,3).astype(np.float32)
    for c in range(3):
        channels.append(np.log(small(cv2.boxFilter(med[:,:,c]**2,-1,(13,13),normalize=True))+1e-3))
    channels.extend(guide.astype(np.float32))
    grid=np.linspace(-1,1,128,dtype=np.float32)
    xx,yy=np.meshgrid(grid,grid)
    channels.extend((xx,yy,np.minimum(1-abs(xx),1-abs(yy))))
    result=np.stack(channels).astype(np.float16)
    assert result.shape==(60,128,128) and np.isfinite(result).all(),result.shape
    return result


def score(pred,gt):
    result=np.zeros(len(gt))
    for p,g in ((pred,gt),(~pred,~gt)):
        union=(p|g).sum((1,2)); inter=(p&g).sum((1,2))
        result+=np.divide(inter,union,out=np.ones(len(gt)),where=union!=0)/2
    return float(result.mean())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kit-root",type=Path,required=True); ap.add_argument("--baseline-root",type=Path,required=True)
    ap.add_argument("--guidance-root",type=Path); ap.add_argument("--output-root",type=Path,required=True)
    ap.add_argument("--stage",choices=("features","fit"),required=True); args=ap.parse_args()
    args.output_root.mkdir(parents=True,exist_ok=True)
    rows=json.loads((args.kit_root/"release/train/metadata.json").read_text())
    split=json.loads((args.baseline_root/"split.json").read_text())
    with np.load(args.kit_root/"release/train/masks_128.npz") as z: masks=dict(zip(z["names"].tolist(),z["masks"].astype(bool)))
    if args.stage=="features":
        from PIL import Image
        import cv2
        cv2.setNumThreads(8)
        for method in METHODS:
            subset=[r for r in rows if r["method"]==method]; dest=args.output_root/("features_"+method+".npz")
            if dest.exists(): continue
            if args.guidance_root:
                with np.load(args.guidance_root/"train"/(method+".npz")) as z: guide=dict(zip(z["names"].tolist(),z["maps"]))
            else:
                guide={r['image_name']:np.zeros((6,128,128),dtype=np.float16) for r in subset}
            values=[]
            for i,r in enumerate(subset):
                n=r["image_name"]
                with Image.open(args.kit_root/"release/train/images"/(n+".png")) as im: image=np.array(im.convert("RGB"))
                values.append(features(image,guide[n]))
                if i%40==0: print("TEXTURE",method,i+1,flush=True)
            np.savez_compressed(dest,names=np.array([r["image_name"] for r in subset]),maps=np.stack(values))
        return
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    import joblib
    threadpool_limits(limits=8)
    all_prob={}; report={}
    for method in METHODS:
        with np.load(args.output_root/("features_"+method+".npz")) as z: bank=dict(zip(z["names"].tolist(),z["maps"]))
        calibration=[r for r in rows if r["method"]==method and r["image_name"] in split["train"]]
        tune=[]
        for full in (False,True):
            names=sorted(r["image_name"] for r in calibration if r["full_watermark"]==full)
            random.Random(20261122).shuffle(names); tune.extend(names[:14])
        fit=[r["image_name"] for r in calibration if r["image_name"] not in tune]
        val=[r["image_name"] for r in rows if r["method"]==method and r["image_name"] in split["validation"]]
        rng=np.random.default_rng(20261122); xs=[];ys=[]
        for name in fit:
            ix=rng.choice(128*128,1024,replace=False)
            xs.append(bank[name].reshape(60,-1)[:,ix].T.astype(np.float32)); ys.append(masks[name].ravel()[ix])
        x=np.concatenate(xs); y=np.concatenate(ys)
        tree=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,learning_rate=.08,min_samples_leaf=150,
                                            l2_regularization=10,early_stopping=False,max_bins=128,warm_start=True,random_state=20261122)
        gt=np.stack([masks[n] for n in tune]); best={"miou":-1}
        for iterations in (100,300,600):
            start=time.time(); tree.set_params(max_iter=iterations); tree.fit(x,y)
            prob=np.stack([tree.predict_proba(bank[n].reshape(60,-1).T.astype(np.float32))[:,1].reshape(128,128) for n in tune])
            thresholds=np.linspace(.2,.8,13); scores=[score(prob>=t,gt) for t in thresholds]; k=int(np.argmax(scores))
            current={"miou":scores[k],"threshold":float(thresholds[k]),"iterations":iterations}
            if current["miou"]>best["miou"]:
                best=current; joblib.dump(tree,args.output_root/("best_"+method+".joblib"))
            print("TEXTURE_TUNE",method,current,"seconds",round(time.time()-start,1),flush=True)
        tree=joblib.load(args.output_root/("best_"+method+".joblib"))
        prob=np.stack([tree.predict_proba(bank[n].reshape(60,-1).T.astype(np.float32))[:,1].reshape(128,128) for n in val])
        report[method]={"inner_selection":best,"validation_miou":score(prob>=best["threshold"],np.stack([masks[n] for n in val]))}
        all_prob.update(zip(val,prob))
    results={"methods":report,"validation_miou":float(np.mean([v["validation_miou"] for v in report.values()]))}
    (args.output_root/"metrics.json").write_text(json.dumps(results,indent=2)+"\n")
    names=sorted(all_prob)
    np.savez_compressed(args.output_root/"validation.npz",names=np.array(names),probabilities=np.stack([all_prob[n] for n in names]))
    print("TEXTURE_RESULTS",json.dumps(results),flush=True)


if __name__=="__main__":main()
