"""GPU smoke checks for the learned Adobe extension; run under Slurm."""
import argparse
from pathlib import Path

import torch
from adobe_segmenter import AdobeSegmenter
from localization_v2 import loss_fn


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261127)
    model = AdobeSegmenter(args.kit_root).cuda().train()
    assert not model.encoder.body.training
    assert all(not p.requires_grad for p in model.encoder.body.parameters())
    assert all(p.requires_grad for p in model.encoder.fpn.parameters())
    original_body = {k: v.detach().clone() for k, v in model.encoder.body.state_dict().items()}
    original_fpn = next(model.encoder.fpn.parameters()).detach().clone()
    optimizer, _ = model.configure_optimization(120)
    image = torch.rand(2, 3, 512, 512, device='cuda')
    guides = torch.rand(2, 6, 128, 128, device='cuda')
    method = torch.tensor([0, 2], device='cuda')
    truth = torch.randint(0, 2, (2, 128, 128), device='cuda').float()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = model(image, guides, method)
        assert logits.shape == truth.shape and torch.isfinite(logits).all()
        loss = loss_fn(logits.float(), truth)
    loss.backward()
    assert sum(float(p.grad.abs().sum()) for p in model.encoder.fpn.parameters() if p.grad is not None) > 0
    assert model.fuse[-1].weight.grad.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(original_fpn, next(model.encoder.fpn.parameters()))
    assert all(torch.equal(v, original_body[k]) for k, v in model.encoder.body.state_dict().items())
    print('ADOBE_SMOKE_OK: strict pretrained load, finite dense output, learned FPN/head, unchanged frozen ResNet/BN', flush=True)


if __name__ == '__main__':
    main()
