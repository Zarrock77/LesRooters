"""GPU checks for spatial fusion, preprocessing and frozen sensor invariants."""
import argparse
import gc
from pathlib import Path
import sys

import torch

from localization_v2 import loss_fn
from spatial_segmenter import SpatialSegmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.kit_root))
    from wmloc.watermarks import load_watermark
    torch.set_num_threads(4)
    torch.manual_seed(20261128)
    image = torch.rand(2, 3, 512, 512)
    wrapper = load_watermark('pixelseal', models_dir=args.kit_root / 'models', device='cpu')
    assert torch.equal(SpatialSegmenter.pixel_input(image), wrapper.det.preprocess(wrapper._resize(image)))
    del wrapper
    gc.collect()
    model = SpatialSegmenter(args.kit_root).cuda().train()
    assert not model.encoder.body.training and not model.pixel_encoder.training
    assert all(not p.requires_grad for p in model.encoder.body.parameters())
    assert all(not p.requires_grad for p in model.pixel_encoder.parameters())
    assert all(p.requires_grad for p in model.encoder.fpn.parameters())
    frozen = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
              if name.startswith(('encoder.body.', 'pixel_encoder.'))}
    original_fpn = next(model.encoder.fpn.parameters()).detach().clone()
    optimizer, _ = model.configure_optimization(120)
    optimized = [p for group in optimizer.param_groups for p in group['params']]
    assert len({id(p) for p in optimized}) == len(optimized)
    assert {id(p) for p in optimized} == {id(p) for p in model.parameters() if p.requires_grad}
    image = image.cuda()
    guides = torch.rand(2, 6, 128, 128, device='cuda')
    method = torch.tensor([0, 2], device='cuda')
    truth = torch.randint(0, 2, (2, 128, 128), device='cuda').float()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        sensor_features = model.pixel_features(image)
        assert [tuple(x.shape[1:]) for x in sensor_features] == [
            (96, 64, 64), (192, 32, 32), (384, 16, 16), (768, 8, 8)]
        assert all(not x.requires_grad for x in sensor_features)
        logits = model(image, guides, method)
        assert logits.shape == truth.shape and torch.isfinite(logits).all()
        loss = loss_fn(logits.float(), truth)
    loss.backward()
    for module in (model.encoder.fpn, model.pixel_project, model.fuse, model.noise):
        assert sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in model.encoder.body.parameters())
    assert all(p.grad is None for p in model.pixel_encoder.parameters())
    optimizer.step()
    assert not torch.equal(original_fpn, next(model.encoder.fpn.parameters()))
    assert all(torch.equal(value.detach().cpu(), frozen[name]) for name, value in model.state_dict().items()
               if name in frozen)
    model.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        expected = model(image, guides, method)
    # A fresh instance must accept the complete trained checkpoint strictly.
    restored = SpatialSegmenter(args.kit_root).cuda().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        actual = restored(image, guides, method)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print('SPATIAL_SMOKE_OK: official preprocessing, native sensor stages, finite dense output, '
          'trainable FPN/projections/head, unchanged frozen encoders, strict checkpoint reproduction', flush=True)


if __name__ == '__main__':
    main()
