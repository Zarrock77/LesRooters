#!/bin/bash
# usage: run_tilefus.sh <base combination> <gpu>
# Cross-fitted fusion of a combination's maps with the PixelSeal tile-decoder presence maps, WITHOUT post-processing:
# <base>_tilefus is only offered as a PixelSeal candidate to combine_methods.py, which ranks it out-of-fold with the
# nested regime-wise cut against the unfused maps.
BASE=$1; GPU=${2:-0}
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
[ -f $R/$BASE/oof_probs.npz ] || $K/.venv/bin/python $F/code/merge_folds.py $R/$BASE
rm -rf $R/${BASE}_tilefus; mkdir -p $R/${BASE}_tilefus
CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/fusion_cv.py --folds $R/$BASE --occ $R/tilemaps --frame-json $R/frame/frame_decode.json \
  --train-root $K/release/train --reference $F/reference --out $R/${BASE}_tilefus > $R/${BASE}_tilefus/fusion.log 2>&1
echo done > $R/${BASE}_tilefus/FUSION_DONE
