#!/bin/bash
# Stage 27: one more round of kit-recipe pre-training (3000 steps, blob-field holes) continuing a finished one.
# usage: stage27.sh <gpu> <new run> <previous run> <methods> <lr> <seed>
# Methods containing pixelseal or trustmark use the extractor branches; StegaStamp alone does not.
F=/srv/lesrooters/fable
R=$F/runs
GPU=$1; NEW=$2; PREV=$3; METHODS=$4; LR=$5; SEED=$6
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
WM="--wm-branch"; [ "$METHODS" = "stegastamp" ] && WM=""
until grep -q TRAIN_DONE $R/$PREV/train.log 2>/dev/null; do sleep 5; done
[ -e $R/$NEW ] && { echo "$NEW exists"; exit 1; }
bash $F/code/run_train.sh $NEW $GPU --flag $WM --methods $METHODS --objects 0.7 --pure 0.85 --fields 0.5 --seed $SEED --iters 3000 --eval-every 500 --lr $LR --init $R/$PREV/ckpt_last.pt --layers $ALL
