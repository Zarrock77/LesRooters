"""Measure real GPU memory and verify the three provided watermark decoders."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch

from adobe_segmenter import AdobeSegmenter
from localization_v2 import loss_fn, read_image


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root', type=Path, required=True)
    ap.add_argument('--report', type=Path, required=True)
    args = ap.parse_args()
    sys.path.insert(0, str(args.kit_root))
    torch.set_num_threads(4)
    torch.manual_seed(20261128)
    torch.backends.cudnn.benchmark = True
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    asset_hashes = json.loads(Path(__file__).with_name('cloud_asset_checksums.json').read_text())
    for filename, expected in asset_hashes.items():
        with (args.kit_root / filename).open('rb') as stream:
            measured_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert measured_hash == expected, ('JURECA asset mismatch', filename)
    original = json.loads(Path(__file__).with_name('splits').joinpath('baseline_v1.json').read_text())
    metadata_path = args.kit_root / 'release/train/metadata.json'
    assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == original['metadata_sha256']
    rows = json.loads(metadata_path.read_text())
    assert len(rows) == 510
    report = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
              'cuda_runtime': torch.version.cuda, 'bf16': True,
              'source_metadata_verified': True, 'jureca_asset_hashes_verified': asset_hashes,
              'batch_tests': [], 'decoders': {}}
    for batch in (8,):
        model = AdobeSegmenter(args.kit_root).cuda().train()
        optimizer, _ = model.configure_optimization(120)
        images = torch.rand(batch, 3, 512, 512, device='cuda')
        guides = torch.rand(batch, 6, 128, 128, device='cuda')
        methods = torch.arange(batch, device='cuda') % 3
        truth = torch.randint(0, 2, (batch, 128, 128), device='cuda').float()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        for step in range(4):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(images, guides, methods).float()
                loss = loss_fn(logits, truth)
            assert logits.shape == truth.shape and torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        measured = {'batch_size': batch, 'steps': len(timings),
                    'peak_allocated_GiB': torch.cuda.max_memory_allocated() / 2**30,
                    'peak_reserved_GiB': torch.cuda.max_memory_reserved() / 2**30,
                    'warm_step_seconds': float(np.median(timings[1:]))}
        report['batch_tests'].append(measured)
        print('BATCH_OK', json.dumps(measured), flush=True)
        del model, optimizer, images, guides, methods, truth, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
    from wmloc.watermarks import load_watermark
    for method in ('pixelseal', 'trustmark', 'stegastamp'):
        if method == 'stegastamp':
            import onnxruntime as ort
            ort.preload_dlls()
        wrapper = load_watermark(method, models_dir=args.kit_root / 'models', device='cuda')
        if method == 'stegastamp':
            assert 'CUDAExecutionProvider' in wrapper.sess.get_providers()
        row = next(r for r in rows if r['method'] == method)
        image = read_image(args.kit_root / 'release', 'train', row['image_name'])
        tensor = torch.from_numpy(image).permute(2, 0, 1)[None].float() / 255
        logits = wrapper.logits(tensor)
        assert logits.shape == (1, wrapper.nbits) and torch.isfinite(logits).all()
        report['decoders'][method] = {'output_shape': list(logits.shape), 'finite': True}
        print('DECODER_OK', method, list(logits.shape), flush=True)
        del wrapper, logits, tensor
        gc.collect()
        torch.cuda.empty_cache()
    report['process_peak_RSS_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    report['caveat'] = 'Short synthetic-batch memory/timing test, not full-epoch throughput or model validation.'
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print('CLOUD_BENCHMARK_OK', flush=True)


if __name__ == '__main__':
    main()
