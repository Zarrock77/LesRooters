"""Reuse TrustMark's frozen, watermark-trained ResNet features for segmentation.

Runs the same fixed inner/outer split protocol as localization_v2.py, with a
multi-scale feature pyramid and an additional image-residual branch.
"""
import gc
import torch
from torch import nn
import torch.nn.functional as F
import localization_v2 as experiment


class PretrainedSegmenter(nn.Module):
    def __init__(self):
        super().__init__()
        from wmloc.watermarks import load_watermark
        wrapper=load_watermark("trustmark",device="cuda")
        self.encoder=wrapper.net
        self.encoder.requires_grad_(False)
        del wrapper; gc.collect()
        self.project=nn.ModuleList(nn.Conv2d(ch,16,1) for ch in (64,256,512,1024,2048))
        self.noise=nn.Sequential(nn.Conv2d(192,16,1),nn.SiLU(),experiment.block(16,16))
        self.fuse=nn.Sequential(experiment.block(105,64),experiment.block(64,32),nn.Conv2d(32,1,1))

    def train(self,mode=True):
        super().train(mode); self.encoder.eval(); return self

    def forward(self,image,guide,method):
        with torch.no_grad():
            n=self.encoder
            x=n.relu(n.bn1(n.conv1(image*2-1))); features=[x]
            x=n.layer1(n.maxpool(x)); features.append(x)
            x=n.layer2(x); features.append(x)
            x=n.layer3(x); features.append(x)
            x=n.layer4(x); features.append(x)
        pyramids=[F.interpolate(p(f),(128,128),mode="bilinear",align_corners=False)
                  for p,f in zip(self.project,features)]
        residuals=[(image-F.avg_pool2d(image,k,1,k//2))*8 for k in (3,7,15)]
        noise=self.noise(F.pixel_unshuffle(torch.cat([image-.5]+residuals,1),4))
        condition=F.one_hot(method,3).to(noise.dtype)[:,:,None,None].expand(-1,-1,128,128)
        return self.fuse(torch.cat(pyramids+[noise,(guide-.5)*4,condition],1))[:,0]


if __name__=="__main__":
    experiment.ResidualSegmenter=PretrainedSegmenter
    experiment.main()
