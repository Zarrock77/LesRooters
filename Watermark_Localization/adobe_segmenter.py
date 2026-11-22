"""Extend Adobe's pretrained detector backbone into a learned dense segmenter.

The official ResNet/FPN weights initialize visual features, not output boxes.
The FPN and a new pixel head learn from the fit masks; ResNet and its batch-norm
statistics stay frozen. No image identifiers, box rasterization or label-derived
spatial templates enter forward().
"""
import hashlib
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor

import localization_v2 as experiment

CHECKPOINT_MD5 = '9d15428a33e15140ea16aa378416d304'
BACKBONE_PREFIX = 'yolo.yolo_model.backbone.'


class AdobeSegmenter(nn.Module):
    def __init__(self, kit_root):
        super().__init__()
        checkpoint = Path(kit_root) / 'models/trustmark_bbox/trustmark_bbox_Q.ckpt'
        assert hashlib.md5(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_MD5
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state.get('state_dict', state)
        backbone_state = {k[len(BACKBONE_PREFIX):]: v for k, v in state.items()
                          if k.startswith(BACKBONE_PREFIX)}
        assert backbone_state, 'Missing Adobe backbone weights'
        # This is the same ResNet50/FPN construction as the pinned Adobe source.
        self.encoder = _resnet_fpn_extractor(
            resnet50(weights=None, norm_layer=nn.BatchNorm2d), trainable_layers=5)
        self.encoder.load_state_dict(backbone_state, strict=True)
        self.encoder.body.requires_grad_(False)
        self.project = nn.ModuleList(nn.Conv2d(256, 16, 1) for _ in range(5))
        self.noise = nn.Sequential(nn.Conv2d(192, 16, 1), nn.SiLU(), experiment.block(16, 16))
        self.fuse = nn.Sequential(experiment.block(105, 64), experiment.block(64, 32), nn.Conv2d(32, 1, 1))
        # Adobe's wrapper converts [-1,1] to [0,1], then applies this normalization.
        self.register_buffer('image_mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('image_std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def train(self, mode=True):
        super().train(mode)
        self.encoder.body.eval()
        return self

    def configure_optimization(self, epochs):
        fpn = list(self.encoder.fpn.parameters())
        head = [p for name, p in self.named_parameters() if not name.startswith('encoder.') and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': fpn, 'lr': .00005},
            {'params': head, 'lr': .002},
        ], weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda epoch: .05 + .95 * (1 + math.cos(math.pi * epoch / epochs)) / 2)
        return optimizer, scheduler

    def forward(self, image, guide, method):
        with torch.no_grad():
            body = self.encoder.body((image - self.image_mean) / self.image_std)
        features = self.encoder.fpn(body)
        assert tuple(features) == ('0', '1', '2', '3', 'pool')
        pyramids = [F.interpolate(projection(feature), (128, 128), mode='bilinear', align_corners=False)
                    for projection, feature in zip(self.project, features.values())]
        residuals = [(image - F.avg_pool2d(image, k, 1, k // 2)) * 8 for k in (3, 7, 15)]
        noise = self.noise(F.pixel_unshuffle(torch.cat([image - .5] + residuals, 1), 4))
        condition = F.one_hot(method, 3).to(noise.dtype)[:, :, None, None].expand(-1, -1, 128, 128)
        return self.fuse(torch.cat(pyramids + [noise, (guide - .5) * 4, condition], 1))[:, 0]
