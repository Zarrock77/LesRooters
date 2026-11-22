#!/bin/bash
# usage: chain_cal.sh <base combination> <gpu> [size edges, e.g. 0.02,0.10]
# Nested decision cut per (method, regime) and, with size edges, per predicted size of the minority region, on top
# of a finished combination; then the usual post-processing (rectangles, segment selection). The result
# <base>_cal (or <base>_cal2 with sizes) is a candidate: compare it out-of-fold with the base (compare_runs.py).
BASE=$1; GPU=${2:-0}; EDGES=${3:-}
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
OUT=${BASE}_cal; [ -n "$EDGES" ] && OUT=${BASE}_cal2
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
until [ -f $R/$BASE/SAM_POST_DONE ]; do sleep 5; done
rm -rf $R/$OUT
$K/.venv/bin/python -u $F/code/calibrate_thr.py $R $BASE $OUT $R/frame/frame_decode.json $F/reference $EDGES > $R/$OUT.log 2>&1
# Results searched before rect_refine.py stored map hashes get their side files, so that unchanged maps are reused.
$K/.venv/bin/python $F/code/rect_hashes.py $R/$BASE > $R/${OUT}_hashes.log 2>&1 || true
bash $F/code/run_post.sh $OUT $GPU $BASE
bash $F/code/run_sam_post.sh $OUT
