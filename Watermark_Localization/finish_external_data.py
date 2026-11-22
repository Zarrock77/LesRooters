"""Prepare fixed external sources and synthetic pairs using CPU only."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    kit = args.kit_root
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    subprocess.run([sys.executable, '-u', str(code / 'prepare_external_coco.py'),
                    '--data-root', str(kit / 'release'),
                    '--output-root', str(kit / 'output/external_coco_v1'),
                    '--limit', '3000', '--max-candidates', '6000', '--transport', 'http'], check=True)
    subprocess.run([sys.executable, '-u', str(code / 'generate_external.py'),
                    '--kit-root', str(kit), '--expected-sources', '3000', '--device', 'cpu'], check=True)


if __name__ == '__main__':
    main()
