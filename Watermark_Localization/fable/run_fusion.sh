#!/bin/bash
# usage: run_fusion.sh <folds run name> <gpu index> [suffix] [extra fusion_cv.py arguments, e.g. --occ DIR --ig DIR]
# Cross-fitted fusion with decoder-occlusion maps, then the same post-processing on the fused probabilities.
set -euo pipefail
NAME=$1; GPU=${2:-0}; SUF=${3:-fus}; shift 3 || true
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
R=/srv/lesrooters/fable/runs
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
[ -f $R/$NAME/oof_probs.npz ] || $K/.venv/bin/python $C/merge_folds.py $R/$NAME
mkdir -p $R/${NAME}_${SUF}
CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $C/fusion_cv.py --folds $R/$NAME "$@" --frame-json $R/frame/frame_decode.json \
  --train-root $K/release/train --reference /srv/lesrooters/fable/reference --out $R/${NAME}_${SUF} > $R/${NAME}_${SUF}/fusion.log 2>&1
bash $C/run_post.sh ${NAME}_${SUF} $GPU $NAME
