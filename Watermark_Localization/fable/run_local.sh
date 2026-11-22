#!/bin/bash
# usage: run_local.sh <run name> <gpu> <method> [train_local.py arguments]
set -euo pipefail
NAME=$1; GPU=$2; METHOD=$3; shift 3
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
R=/srv/lesrooters/fable/runs/$NAME
mkdir -p $R
cd $K
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=$GPU
exec $K/.venv/bin/python -u /srv/lesrooters/fable/code/train_local.py --kit $K --method $METHOD \
  --layers /scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl --out $R "$@" > $R/train.log 2>&1
