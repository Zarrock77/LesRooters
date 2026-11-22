#!/bin/bash
# usage: run_tta_probe.sh <gpu> <run> <method> [<run> <method> ...]   -- one after the other
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
R=/srv/lesrooters/fable/runs
GPU=$1; shift
cd $K
while [ $# -gt 1 ]; do
  CUDA_VISIBLE_DEVICES=$GPU PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 .venv/bin/python -u /srv/lesrooters/fable/code/tta_probe.py $K $R/$1 $R/frame/frame_decode.json $2 > $R/tta_$1_$2.log 2>&1
  shift 2
done
