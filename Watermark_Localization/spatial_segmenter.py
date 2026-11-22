"""Learn spatial fusion of Adobe features and frozen native PixelSeal features.

The PixelSeal sensor runs at its released 256-pixel decoder resolution. Its
four intermediate stages retain information discarded by the six bit-agreement
guidance maps. A small U-Net learns context and boundaries from the fit masks;
neither pretrained image encoder nor its running statistics are updated.
"""
import gc
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from adobe_segmenter import AdobeSegmenter
import localization_v2 as experiment


PIXEL_CHECKPOINT_SHA256 = '0c5665cff20eb6ce1b5aaa7d91c19dafb418bfee32d02dd3344e4ed60d9d75bd'


class SpatialFusion(nn.Module):
    """Modest learned context pyramid; all outputs remain dense pixel logits."""
    def __init__(self, channels=137):
        super().__init__()
        self.down0 = experiment.block(channels, 32)
        self.down1 = experiment.block(32, 48, stride=2)
        self.down2 = experiment.block(48, 64, stride=2)
        self.down3 = experiment.block(64, 96, stride=2)
        self.up2 = experiment.block(96 + 64, 64)
        self.up1 = experiment.block(64 + 48, 48)
        self.up0 = experiment.block(48 + 32, 32)
        self.dropout = nn.Dropout2d(.1)
        self.output = nn.Conv2d(32, 1, 1)

    @staticmethod
    def join(coarse, skip):
        return torch.cat((F.interpolate(coarse, size=skip.shape[-2:],
                                         mode='bilinear', align_corners=False), skip), 1)

    def forward(self, inputs):
        x0 = self.down0(inputs)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.dropout(self.down3(x2))
        x = self.up2(self.join(x3, x2))
        x = self.up1(self.join(x, x1))
        return self.output(self.up0(self.join(x, x0)))


class SpatialSegmenter(AdobeSegmenter):
    def __init__(self, kit_root):
        super().__init__(kit_root)
        from wmloc.watermarks import load_watermark
        checkpoint = Path(kit_root) / 'models/pixelseal/checkpoint.pth'
        with checkpoint.open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == PIXEL_CHECKPOINT_SHA256
        # Keep only the official detector backbone; discard the unused embedder.
        wrapper = load_watermark('pixelseal', models_dir=Path(kit_root) / 'models', device='cpu')
        assert wrapper.resolution == 256
        self.pixel_encoder = wrapper.det.convnext
        assert len(self.pixel_encoder.stages) == len(self.pixel_encoder.downsample_layers) == 4
        self.pixel_encoder.requires_grad_(False).eval()
        del wrapper
        gc.collect()
        self.pixel_project = nn.ModuleList(nn.Conv2d(channels, 8, 1)
                                          for channels in (96, 192, 384, 768))
        self.fuse = SpatialFusion(137)

    def train(self, mode=True):
        super().train(mode)
        self.pixel_encoder.eval()
        return self

    @staticmethod
    def pixel_input(image):
        # Exact preprocessing used by the pinned hackathon PixelSeal wrapper.
        return F.interpolate(image, size=(256, 256), mode='bilinear',
                             align_corners=False, antialias=True) * 2 - 1

    def pixel_features(self, image):
        features = []
        with torch.no_grad():
            x = self.pixel_input(image)
            for downsample, stage in zip(self.pixel_encoder.downsample_layers,
                                         self.pixel_encoder.stages):
                x = stage(downsample(x))
                features.append(x)
        return features

    def forward(self, image, guide, method):
        with torch.no_grad():
            body = self.encoder.body((image - self.image_mean) / self.image_std)
        features = self.encoder.fpn(body)
        assert tuple(features) == ('0', '1', '2', '3', 'pool')
        pyramids = [F.interpolate(projection(feature), (128, 128), mode='bilinear', align_corners=False)
                    for projection, feature in zip(self.project, features.values())]
        pixel_pyramids = [F.interpolate(projection(feature), (128, 128), mode='bilinear', align_corners=False)
                          for projection, feature in zip(self.pixel_project, self.pixel_features(image))]
        residuals = [(image - F.avg_pool2d(image, k, 1, k // 2)) * 8 for k in (3, 7, 15)]
        noise = self.noise(F.pixel_unshuffle(torch.cat([image - .5] + residuals, 1), 4))
        condition = F.one_hot(method, 3).to(noise.dtype)[:, :, None, None].expand(-1, -1, 128, 128)
        return self.fuse(torch.cat(pyramids + pixel_pyramids +
                                   [noise, (guide - .5) * 4, condition], 1))[:, 0]
