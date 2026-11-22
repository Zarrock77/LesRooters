"""Export only the strongest eligible member of the predeclared batch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--kit-root',type=Path,required=True)
    args=ap.parse_args();out=args.kit_root/'output';eligible=[]
    for candidate in ('pretrained_tta','detector','detector_blend'):
        decision=json.loads((out/('cv_'+candidate+'_v1')/'replacement_decision.json').read_text())
        if decision['export_recommended']:eligible.append(decision)
    if not eligible:print('NO_ROBUST_GAIN: no submission artifact produced',flush=True);return
    best=max(eligible,key=lambda d:d['oof_miou']);candidate=best['candidate']
    subprocess.run([sys.executable,str(Path(__file__).with_name('export_cv_ensemble.py')),
        '--kit-root',str(args.kit_root),'--guidance-root',str(out/'localization_v2'),
        '--comparison-root',str(out/('cv_'+candidate+'_v1')),'--output-root',str(out/'submission_next_v1'),
        '--version','next_'+candidate+'_v1','--allow-numerical-replay'],check=True)


if __name__=='__main__':main()
