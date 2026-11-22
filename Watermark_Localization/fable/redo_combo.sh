#!/bin/bash
# usage: redo_combo.sh <combo name> <gpu> <runs to repredict, comma separated> pixelseal=<cands> trustmark=<cands> stegastamp=<cands>
# Re-store every component run's predictions through the deterministic inference path, then combine and post-process.
set -uo pipefail
COMBO=$1; GPU=$2; RUNS=$3; shift 3
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
FRAME=$R/frame/frame_decode.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
for run in ${RUNS//,/ }; do
  if [ ! -f $R/$run/REPREDICTED ]; then
    if [[ $run == *_tta ]]; then       # shift-averaged sibling of the run without the suffix
      mkdir -p $R/$run
      CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/${run%_tta} --tta-out $R/$run --frame-json $FRAME > $R/$run/repredict.log 2>&1 &
    else
      CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/$run --frame-json $FRAME > $R/$run/repredict.log 2>&1 &
    fi
    while [ $(jobs -rp | wc -l) -ge ${REPREDICT_JOBS:-3} ]; do sleep 2; done      # a few at a time: GPU memory
  fi
done
wait
for run in ${RUNS//,/ }; do [ -f $R/$run/REPREDICTED ] || { echo "REPREDICT FAILED $run"; exit 1; }; done
rm -f $R/$COMBO/replay.json $R/$COMBO/replay.log $R/$COMBO/rect_*.json $R/$COMBO/PLAIN_DONE $R/$COMBO/POST_DONE
$K/.venv/bin/python $F/code/combine_methods.py $R $COMBO "$@" > $R/${COMBO}_combine.log 2>&1
COMPONENTS=$(cat $R/$COMBO/components.txt) bash $F/code/run_post.sh $COMBO $GPU
