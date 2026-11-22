#!/bin/bash
# usage: tta_when_done.sh <run> <gpu>
# Waits for the run's five folds, then writes its shift-averaged sibling <run>_tta (repredict.py --tta-out).
RUN=$1; GPU=${2:-0}
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
until [ -f $R/$RUN/ALL_DONE ]; do sleep 5; done
[ -f $R/${RUN}_tta/REPREDICTED ] && exit 0
mkdir -p $R/${RUN}_tta
CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/$RUN --tta-out $R/${RUN}_tta --frame-json $R/frame/frame_decode.json > $R/${RUN}_tta/repredict.log 2>&1
