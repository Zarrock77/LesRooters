#!/bin/bash
# usage: run_train.sh <run name> <gpu index> [train_det.py arguments...]
set -euo pipefail
NAME=$1; GPU=$2; shift 2
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
R=/srv/lesrooters/fable/runs/$NAME
mkdir -p $R
cd $K
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 HF_HOME=/srv/lesrooters/fable/hf CUDA_VISIBLE_DEVICES=$GPU
exec $K/.venv/bin/python -u $C/train_det.py --kit $K --layers /scratch/fable/syn --out $R "$@" > $R/train.log 2>&1
