"""Review the learned Adobe candidate and export only after the frozen gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).parent
    root = args.kit_root / 'output'
    subprocess.run([sys.executable, str(code / 'review_cv.py'), '--kit-root', str(args.kit_root),
                    '--candidate', 'adobe'], check=True)
    decision = json.loads((root / 'cv_adobe_v1/replacement_decision.json').read_text())
    if not decision['export_recommended']:
        print('NO_ROBUST_GAIN: learned Adobe candidate not exported', flush=True)
        return
    subprocess.run([sys.executable, str(code / 'export_cv_ensemble.py'),
                    '--kit-root', str(args.kit_root), '--guidance-root', str(root / 'localization_v2'),
                    '--comparison-root', str(root / 'cv_adobe_v1'),
                    '--output-root', str(root / 'submission_adobe_v1'),
                    '--version', 'adobe_v1', '--allow-numerical-replay'], check=True)


if __name__ == '__main__':
    main()
