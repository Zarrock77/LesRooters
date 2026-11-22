#!/bin/bash
# usage: fast_combo.sh <combo> <gpu> pixelseal=<candidates> trustmark=<candidates> stegastamp=<candidates>
# Per-method combination, replay of the chosen fold models, nested decision-cut calibration, then rectangles and
# segment selection ONCE, on the calibrated maps: <combo>_cal2 is the candidate. redo_combo.sh + chain_cal.sh do the
# same but also post-process the uncalibrated combination first, which costs one rectangle search (about ten
# minutes) for a version that is no longer submitted. Every run named in the candidates must already hold its
# stored predictions (shift-averaged siblings written by tta_when_done.sh).
set -euo pipefail
COMBO=$1; GPU=${2:-0}; shift 2
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
PY=$K/.venv/bin/python
FRAME=$R/frame/frame_decode.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
rm -rf $R/$COMBO $R/${COMBO}_cal2
$PY $F/code/combine_methods.py $R $COMBO "$@" > $R/${COMBO}_combine.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $PY -u $F/code/replay_check.py --kit $K --folds $R/$COMBO --frame-json $FRAME --components $(cat $R/$COMBO/components.txt) > $R/$COMBO/replay.log 2>&1 &
$PY -u $F/code/calibrate_thr.py $R $COMBO ${COMBO}_cal2 $FRAME $F/reference 0.02,0.10 > $R/${COMBO}_cal2.log 2>&1
wait
[ -f $R/$COMBO/replay.json ] || { echo "replay failed for $COMBO"; exit 1; }
bash $F/code/run_post.sh ${COMBO}_cal2 $GPU $COMBO
bash $F/code/run_sam_post.sh ${COMBO}_cal2
