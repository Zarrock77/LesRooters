#!/bin/bash
# usage: run_ft.sh <finetune name> <init run name> <gpu index> <fold> [<fold> ...] [-- finetune_cv.py arguments]
set -euo pipefail
NAME=$1; INIT=$2; GPU=$3; shift 3
FOLDS=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do FOLDS+=("$1"); shift; done
[ $# -gt 0 ] && shift
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
B=/srv/lesrooters/fable/runs/$NAME
mkdir -p $B
# Freeze the starting checkpoint once so every fold starts from the same weights.
[ -f $B/init.pt ] || cp /srv/lesrooters/fable/runs/$INIT/ckpt_last.pt $B/init.pt
cd $K
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/srv/lesrooters/fable/hf CUDA_VISIBLE_DEVICES=$GPU
for FOLD in "${FOLDS[@]}"; do
  R=$B/fold_$FOLD
  mkdir -p $R
  $K/.venv/bin/python -u $C/finetune_cv.py --kit $K --layers /scratch/fable/syn --reference /srv/lesrooters/fable/reference \
    --init $B/init.pt --frame-json /srv/lesrooters/fable/runs/frame/frame_decode.json --fold $FOLD --out $R "$@" > $R/ft.log 2>&1
done
