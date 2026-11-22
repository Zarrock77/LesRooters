"""Add PixelSeal's detection channel, omitted by the bit-only guidance.

The two extra maps are computed without labels at native and 512 resolution.
Backbones remain frozen; the training protocol and model size stay comparable.
"""
import gc
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import localization_v2 as experiment
from pretrained_segmenter import PretrainedSegmenter

base_guidance=experiment.guidance
FEATURE_VERSION='pixelseal-detection-native-512-v1'


def detection_guidance(args,rows,split_name):
    maps=base_guidance(args,rows,split_name)
    folder=args.output_root/'guidance'/split_name
    path=folder/'detection_extra.npz'; names=[r['image_name'] for r in rows]
    if path.exists():
        with np.load(path) as z:
            assert z['names'].tolist()==names and z['version'].item()==FEATURE_VERSION
            extras=z['maps']
    else:
        from wmloc.watermarks import load_watermark
        wm=load_watermark('pixelseal',models_dir=args.kit_root/'models',device='cuda'); outputs=[]
        with torch.inference_mode():
            for start in range(0,len(rows),8):
                batch=rows[start:start+8]
                x=torch.from_numpy(np.stack([experiment.read_image(args.data_root,split_name,r['image_name']) for r in batch])).permute(0,3,1,2).float().cuda()/255
                features=[]
                for resolution in (256,512):
                    wm.resolution=resolution
                    image=wm.det.preprocess(wm._resize(x))
                    embedding=wm.det.pixel_decoder.output_upscaling(wm.det.convnext(image))
                    linear=wm.det.pixel_decoder.linear
                    logits=torch.einsum('bchw,c->bhw',embedding,linear.weight[0])+linear.bias[0]
                    features.append(F.interpolate(logits[:,None],(128,128),mode='bilinear',align_corners=False).sigmoid())
                outputs.append(torch.cat(features,1).cpu().numpy().astype(np.float16))
                if start%80==0:print('DETECTION_GUIDANCE',split_name,min(start+8,len(rows)),flush=True)
        extras=np.concatenate(outputs);assert extras.shape==(len(rows),2,128,128) and np.isfinite(extras).all()
        np.savez_compressed(path,names=np.array(names),maps=extras,version=np.array(FEATURE_VERSION))
        del wm;gc.collect();torch.cuda.empty_cache()
    return {n:np.concatenate((maps[n],extra),axis=0) for n,extra in zip(names,extras)}


class DetectorSegmenter(PretrainedSegmenter):
    def __init__(self):
        super().__init__()
        self.fuse=nn.Sequential(experiment.block(107,64),experiment.block(64,32),nn.Conv2d(32,1,1))
