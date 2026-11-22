#!/bin/bash
# usage: run_sam_post.sh <run with rect_train.json / rect_test.json and replay.json>
# Nested segment selection on top of the rectangle-snapped pipeline -> report_sam.json + submission_sam.
set -euo pipefail
NAME=$1
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
R=/srv/lesrooters/fable/runs
B=$R/$NAME
PY=$K/.venv/bin/python
FRAME=$R/frame/frame_decode.json
REF=/srv/lesrooters/fable/reference/image_scores.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
# A calibrated run (<combo>_cal, <combo>_cal2) holds shifted maps of the same fold models: its replay record is the base's.
SRC=${NAME%_cal2}; SRC=${SRC%_cal}
[ -f $B/replay.json ] || cp $R/$SRC/replay.json $B/replay.json
cd $C
$PY sam_select.py --folds $B --train-root $K/release/train --rect-train $B/rect_train.json --rect-test $B/rect_test.json \
  --frame-json $FRAME --sam $R/sam/train_0.npz $R/sam/train_1.npz $R/sam/first/train_0.npz $R/sam/test_0.npz $R/sam/test_1.npz $R/sam/test_2.npz \
  --out $B/sam_scores.json --test-out $B/sam_test.npz > $B/sam_select.log 2>&1
cd $K
$PY $C/assemble.py --folds $B --train-root $K/release/train --reference-scores $REF --rect-train $B/rect_train.json \
  --rect-test $B/rect_test.json --replay $B/replay.json --sam-scores $B/sam_scores.json --sam-test $B/sam_test.npz \
  --candidate ${NAME}_sam --report $B/report_sam.json --submission $B/submission_sam > $B/assemble_sam.log 2>&1
echo SAM_POST_DONE > $B/SAM_POST_DONE
