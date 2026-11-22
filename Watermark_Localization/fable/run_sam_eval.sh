#!/bin/bash
# usage: run_sam_eval.sh <run> <regimes> <log name> [extra runs...]   -- nested segment-selection evaluation only (no test masks)
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
R=/srv/lesrooters/fable/runs
cd /srv/lesrooters/fable/code
RUN=$1; REG=$2; TAG=$3; shift 3
EXTRA=""
[ $# -gt 0 ] && EXTRA="--extra $(for r in "$@"; do printf '%s ' $R/$r; done)"
set -- $RUN $REG $TAG
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 $K/.venv/bin/python sam_select.py --regimes $2 --folds $R/$1 --train-root $K/release/train \
  --rect-train $R/$1/rect_train.json --frame-json $R/frame/frame_decode.json \
  --sam $R/sam/train_0.npz $R/sam/train_1.npz $R/sam/first/train_0.npz $R/sam/full_train_0.npz $R/sam/full_train_1.npz $R/sam/full_train_2.npz \
  --out $R/$1/sam_scores_$3.json $EXTRA ${SAM_FLAGS:-} > $R/$1/sam_eval_$3.log 2>&1
echo done >> $R/$1/sam_eval_$3.log
