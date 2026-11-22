"""Run the registered spatial CV and export immediately only if its gate passes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root', type=Path, required=True)
    args = ap.parse_args()
    code = Path(__file__).resolve().parent
    os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')

    def run(script, *arguments):
        subprocess.run([sys.executable, '-u', str(code / script), '--kit-root', str(args.kit_root),
                        *map(str, arguments)], check=True, cwd=code)

    run('run_spatial_cv.py')
    run('assess_candidate.py', '--candidate', 'spatial', '--reference', 'adobe')
    folder = args.kit_root / 'output/cv_spatial_v1'
    decision = json.loads((folder / 'replacement_decision.json').read_text())
    if not decision['export_recommended']:
        print('NO_ROBUST_GAIN', json.dumps(decision), flush=True)
        return
    run('export_cv_ensemble.py', '--architecture', 'spatial', '--guidance-root', args.kit_root / 'output/guidance_v2',
        '--comparison-root', folder, '--output-root', args.kit_root / 'output/submission_spatial_v1', '--version', 'spatial_v1')
    print('SPATIAL_READY_FOR_LOCAL_UPLOAD', flush=True)


if __name__ == '__main__':
    main()
