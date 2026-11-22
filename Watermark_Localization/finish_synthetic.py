"""Queue the second registered experiment on the same L4, then gate its export."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def wait_success(unit, completion_log, markers):
    while True:
        result = subprocess.run(['systemctl', 'show', unit, '-p', 'ActiveState', '-p', 'ExecMainStatus', '-p', 'LoadState'],
                                check=True, capture_output=True, text=True)
        status = dict(line.split('=', 1) for line in result.stdout.splitlines())
        if status['ActiveState'] in ('active', 'activating', 'deactivating'):
            print('WAITING_FOR', unit, flush=True)
            time.sleep(30)
            continue
        # Successful transient services are garbage-collected by systemd. A
        # persisted completion marker is required even when the unit is gone.
        assert status['ActiveState'] == 'inactive' and status.get('ExecMainStatus', '0') == '0', status
        log = Path(completion_log).read_text()
        assert any(marker in log for marker in markers), ('Missing successful completion marker', unit)
        return


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kit-root', type=Path, required=True)
    args = ap.parse_args()
    code = Path(__file__).resolve().parent
    os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    setup = args.kit_root.parent / 'setup'
    wait_success('lesrooters-spatial-cv.service', setup / 'spatial-cv.log',
                 ('SPATIAL_READY_FOR_LOCAL_UPLOAD', 'NO_ROBUST_GAIN'))
    wait_success('lesrooters-synthetic-data.service', setup / 'synthetic-data.log', ('SYNTHETIC_COMPLETE',))
    assert (args.kit_root / 'output/synthetic_v1/manifest.json').is_file()
    spatial = json.loads((args.kit_root / 'output/cv_spatial_v1/replacement_decision.json').read_text())
    reference = 'spatial' if spatial['export_recommended'] else 'adobe'

    def run(script, *arguments):
        subprocess.run([sys.executable, '-u', str(code / script), '--kit-root', str(args.kit_root),
                        *map(str, arguments)], check=True, cwd=code)

    run('run_synthetic_cv.py')
    run('assess_candidate.py', '--candidate', 'synthetic', '--reference', reference)
    folder = args.kit_root / 'output/cv_synthetic_v1'
    decision = json.loads((folder / 'replacement_decision.json').read_text())
    run('review_candidate.py', '--candidate', 'synthetic', '--reference', reference)
    if not decision['export_recommended']:
        print('NO_ROBUST_GAIN', json.dumps(decision), flush=True)
        return
    run('export_cv_ensemble.py', '--architecture', 'synthetic', '--guidance-root', args.kit_root / 'output/guidance_v2',
        '--comparison-root', folder, '--output-root', args.kit_root / 'output/submission_synthetic_v1', '--version', 'synthetic_v1')
    print('SYNTHETIC_READY_FOR_LOCAL_UPLOAD', flush=True)


if __name__ == '__main__':
    main()
