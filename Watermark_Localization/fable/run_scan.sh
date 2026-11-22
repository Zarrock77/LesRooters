#!/bin/bash
# usage: run_scan.sh <folds run name> <gpu> <split> <methods>
# Whole-image decoder search for rectangles the detector did not propose; results go to runs/scan/.
set -euo pipefail
NAME=$1; GPU=$2; SPLIT=$3; METHODS=$4
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
R=/srv/lesrooters/fable/runs
PROBS=$R/$NAME/oof_probs.npz; [ "$SPLIT" = test ] && PROBS=$R/$NAME/test_mean_probs.npz
mkdir -p $R/scan
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
PRIOR=$(ls $R/*/rect_$SPLIT.json 2>/dev/null | grep -v "/scan/" | tr '\n' ' ' || true)
$K/.venv/bin/python -u $C/rect_refine.py --kit $K --split $SPLIT --probs $PROBS --frame-json $R/frame/frame_decode.json \
  --out $R/scan/rect_${SPLIT}_${METHODS//,/_}.json --device cuda:$GPU --methods $METHODS --global-scan --prior $PRIOR \
  > $R/scan/scan_${SPLIT}_${METHODS//,/_}.log 2>&1
