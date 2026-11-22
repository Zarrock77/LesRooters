#!/bin/bash
# usage: finish_post.sh <run name> <gpu>   -- redo only the test rectangle search and the final assembly.
set -euo pipefail
NAME=$1; GPU=${2:-0}
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
B=/srv/lesrooters/fable/runs/$NAME
PY=$K/.venv/bin/python
FRAME=/srv/lesrooters/fable/runs/frame/frame_decode.json
REF=/srv/lesrooters/fable/reference/image_scores.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
PRIOR_TEST=$(ls /srv/lesrooters/fable/runs/*/rect_test.json /srv/lesrooters/fable/runs/scan/rect_test_*.json 2>/dev/null | grep -v "/$NAME/" | tr '\n' ' ' || true)
$PY -u $C/rect_refine.py --kit $K --split test --probs $B/test_mean_probs.npz --frame-json $FRAME --out $B/rect_test.json --device cuda:$GPU --prior $PRIOR_TEST > $B/rect_test.log 2>&1
$PY $C/assemble.py --folds $B --train-root $K/release/train --reference-scores $REF --rect-train $B/rect_train.json \
  --rect-test $B/rect_test.json --replay $B/replay.json --candidate ${NAME}_rect --report $B/report_rect.json --submission $B/submission_rect > $B/assemble_rect.log 2>&1
echo POST_DONE > $B/POST_DONE
