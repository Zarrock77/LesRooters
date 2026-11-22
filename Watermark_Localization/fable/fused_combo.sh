#!/bin/bash
# usage: fused_combo.sh <combo> <gpu> <fused run> pixelseal=<candidates> trustmark=<candidates> stegastamp=<candidates>
# Like fast_combo.sh, with the tile-decoder fusion offered for PixelSeal.
#   1. <combo>0 = per-method combination of fold-model runs only, replayed (these are the models behind the maps).
#   2. <combo>  = <combo>0, except that PixelSeal comes from <fused run> if the nested regime-wise ranking prefers it.
#      <fused run> is the cross-fitted fusion (fusion_cv.py) of the SAME PixelSeal maps with the tile-decoder presence
#      maps; a fusion head is not a fold model, so it is not replayed: the replay record is the one of <combo>0 and the
#      PixelSeal maps went through one more cross-fitted stage (stated in the notes of the submission).
#   3. nested decision cut, rectangles and segment selection once: <combo>_cal2.
set -euo pipefail
COMBO=$1; GPU=${2:-0}; FUSED=$3; shift 3
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
PY=$K/.venv/bin/python
FRAME=$R/frame/frame_decode.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
rm -rf $R/${COMBO}0 $R/$COMBO $R/${COMBO}_cal2
$PY $F/code/combine_methods.py $R ${COMBO}0 "$@" > $R/${COMBO}0_combine.log 2>&1
CUDA_VISIBLE_DEVICES=$GPU $PY -u $F/code/replay_check.py --kit $K --folds $R/${COMBO}0 --frame-json $FRAME --components $(cat $R/${COMBO}0/components.txt) > $R/${COMBO}0/replay.log 2>&1 &
# <fused run> = auto: fuse the PixelSeal maps just chosen for <combo>0 (needed when they are not those of an existing fusion).
if [ "$FUSED" = auto ]; then bash $F/code/run_tilefus.sh ${COMBO}0 $((1 - GPU)); FUSED=${COMBO}0_tilefus; fi
$PY $F/code/combine_methods.py $R $COMBO "pixelseal=$FUSED+${COMBO}0" "trustmark=${COMBO}0" "stegastamp=${COMBO}0" > $R/${COMBO}_combine.log 2>&1
$PY -u $F/code/calibrate_thr.py $R $COMBO ${COMBO}_cal2 $FRAME $F/reference 0.02,0.10 > $R/${COMBO}_cal2.log 2>&1
wait
[ -f $R/${COMBO}0/replay.json ] || { echo "replay failed for ${COMBO}0"; exit 1; }
cp $R/${COMBO}0/replay.json $R/$COMBO/replay.json
bash $F/code/run_post.sh ${COMBO}_cal2 $GPU $COMBO
bash $F/code/run_sam_post.sh ${COMBO}_cal2
