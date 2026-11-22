#!/bin/bash
# Shift-averaged siblings of the runs that are already complete, so the combination does not have to wait for them.
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
for run in "$@"; do
  [ -f $R/${run}_tta/REPREDICTED ] && continue
  mkdir -p $R/${run}_tta
  CUDA_VISIBLE_DEVICES=1 $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/$run --tta-out $R/${run}_tta --frame-json $R/frame/frame_decode.json > $R/${run}_tta/repredict.log 2>&1
done
