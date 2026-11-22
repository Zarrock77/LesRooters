#!/bin/bash
# Stage 26: third round of kit-recipe pre-training for PixelSeal + TrustMark: v16_pt = v15_pt + 3000 steps.
# The TrustMark fine-tune from v14_tm (v12_tm + 3000 steps) scores 0.752 / 0.758 on the first two folds against
# 0.733 / 0.752 from v12_tm, so more steps on these compositions still pay.
# usage: stage26.sh <gpu> <marker file to wait for, or ->
F=/srv/lesrooters/fable
R=$F/runs
GPU=${1:-0}; MARK=${2:--}
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
until grep -q TRAIN_DONE $R/v15_pt/train.log 2>/dev/null; do sleep 5; done
[ "$MARK" = "-" ] || until [ -f $MARK ]; do sleep 5; done
[ -e $R/v16_pt ] && { echo "v16_pt exists"; exit 1; }
bash $F/code/run_train.sh v16_pt $GPU --flag --wm-branch --methods pixelseal,trustmark --objects 0.7 --pure 0.85 --fields 0.5 --seed 38 --iters 3000 --eval-every 500 --lr 2e-4 --init $R/v15_pt/ckpt_last.pt --layers $ALL
