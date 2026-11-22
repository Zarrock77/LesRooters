"""Brief per-GPU CUDA/BF16 and NCCL checks; no experiment training or upload.

Run with torchrun --standalone --nproc_per_node=<device count> and --report-dir.
"""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path

import torch
import torch.distributed as distributed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ['LOCAL_RANK'])
    world = int(os.environ['WORLD_SIZE'])
    torch.set_num_threads(2)
    torch.cuda.set_device(rank)
    assert torch.cuda.device_count() >= world and torch.cuda.is_bf16_supported()
    device = torch.device('cuda', rank)
    values = torch.ones((256, 256), device=device, dtype=torch.bfloat16, requires_grad=True)
    product = values @ values
    assert torch.isfinite(product).all() and (product == 256).all()
    product.float().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    distributed.init_process_group('nccl', timeout=timedelta(seconds=60))
    try:
        collective = torch.tensor([float(rank + 1)], device=device)
        distributed.all_reduce(collective)
        expected = world * (world + 1) / 2
        assert collective.item() == expected
        torch.cuda.synchronize()
        properties = torch.cuda.get_device_properties(rank)
        report = {'device': rank, 'name': properties.name, 'memory_bytes': properties.total_memory,
                  'compute_capability': [properties.major, properties.minor],
                  'torch': str(torch.__version__), 'cuda_runtime': str(torch.version.cuda),
                  'bf16_matmul': True, 'backward_finite': True,
                  'nccl_world_size': world, 'nccl_all_reduce': collective.item()}
        args.report_dir.mkdir(parents=True, exist_ok=True)
        (args.report_dir / f'gpu_device_{rank}.json').write_text(json.dumps(report, indent=2) + '\n')
        print('GPU_ENVIRONMENT_VERIFIED', json.dumps(report), flush=True)
    finally:
        distributed.destroy_process_group()


if __name__ == '__main__':
    main()
