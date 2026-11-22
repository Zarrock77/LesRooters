"""Queue external pretraining and CV after the two required predecessors."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from finish_synthetic import wait_success


def file_sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prior_external_decisions(kit_root):
    """Validate the complete promotion chain without reading public scores."""
    reference = 'adobe'
    decisions, hashes = {}, {}
    for candidate in ('spatial', 'synthetic'):
        path = Path(kit_root) / 'output' / ('cv_' + candidate + '_v1') / 'replacement_decision.json'
        decision = json.loads(path.read_text())
        assert decision['candidate'] == candidate and decision['reference'] == reference, 'Invalid prior promotion reference'
        gains, bounds = decision['fold_gain'], decision['paired_gain_99pct']
        mean_gain = decision['paired_gain']
        assert len(gains) == 5 and len(bounds) == 2 and bounds[0] <= bounds[1]
        assert all(math.isfinite(value) for value in [*gains, *bounds, mean_gain])
        expected = min(gains) > 0 and bounds[0] > 0 and mean_gain >= .002
        assert type(decision['export_recommended']) is bool and decision['export_recommended'] == expected, 'Prior decision contradicts frozen gate'
        decisions[candidate] = decision
        hashes[candidate] = file_sha256(path)
        if expected:
            reference = candidate
    return reference, decisions, hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit-root', type=Path, required=True)
    args = parser.parse_args()
    code = Path(__file__).resolve().parent
    os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    setup = args.kit_root.parent / 'setup'
    wait_success('lesrooters-synthetic-cv.service', setup / 'synthetic-cv.log',
                 ('SYNTHETIC_READY_FOR_LOCAL_UPLOAD', 'NO_ROBUST_GAIN'))
    wait_success('lesrooters-external-data.service', setup / 'external-data.log',
                 ('EXTERNAL_WATERMARKS_COMPLETE',))
    generated = json.loads((args.kit_root / 'output/external_watermarks_v1/manifest.json').read_text())
    assert generated['complete'] is True and generated['all_pair_records_validated'] is True
    assert (generated['rows'], generated['pairs'], generated['source_count']) == (18000, 9000, 3000)
    reference, prior_decisions, prior_hashes = prior_external_decisions(args.kit_root)

    def run(script, *arguments):
        subprocess.run([sys.executable, '-u', str(code / script), '--kit-root', str(args.kit_root),
                        *map(str, arguments)], check=True, cwd=code)

    # Both services above must have succeeded before any GPU subprocess starts.
    run('pretrain_external.py')
    run('run_external_cv.py')
    assert prior_external_decisions(args.kit_root) == (reference, prior_decisions, prior_hashes), 'Prior decisions changed while external training ran'
    run('assess_candidate.py', '--candidate', 'external', '--reference', reference)
    folder = args.kit_root / 'output/cv_external_v1'
    decision = json.loads((folder / 'replacement_decision.json').read_text())
    assert decision['candidate'] == 'external' and decision['reference'] == reference
    run('review_candidate.py', '--candidate', 'external', '--reference', reference)
    if not decision['export_recommended']:
        print('NO_ROBUST_GAIN', json.dumps(decision), flush=True)
        return
    run('export_cv_ensemble.py', '--architecture', 'external',
        '--guidance-root', args.kit_root / 'output/guidance_v2', '--comparison-root', folder,
        '--output-root', args.kit_root / 'output/submission_external_v1', '--version', 'external_v1')
    print('EXTERNAL_READY_FOR_LOCAL_UPLOAD', flush=True)


if __name__ == '__main__':
    main()
