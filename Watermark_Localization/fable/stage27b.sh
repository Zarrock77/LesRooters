#!/bin/bash
# Stage 27b: stage27.sh with the share of blob-field holes as an argument.
# TrustMark: the round with blob fields (v15_tm, fine-tune ft_tm15 0.748 on four folds) is below the round without
# (v14_tm, ft_tm14 0.756), while StegaStamp gained a lot with them (v14_st). So TrustMark pre-trainings use 0.
# usage: stage27b.sh <gpu> <new run> <previous run> <methods> <lr> <seed> <fields share>
F=/srv/lesrooters/fable
R=$F/runs
GPU=$1; NEW=$2; PREV=$3; METHODS=$4; LR=$5; SEED=$6; FIELDS=$7
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
WM="--wm-branch"; [ "$METHODS" = "stegastamp" ] && WM=""
until grep -q TRAIN_DONE $R/$PREV/train.log 2>/dev/null; do sleep 5; done
[ -e $R/$NEW ] && { echo "$NEW exists"; exit 1; }
bash $F/code/run_train.sh $NEW $GPU --flag $WM --methods $METHODS --objects 0.7 --pure 0.85 --fields $FIELDS --seed $SEED --iters 3000 --eval-every 500 --lr $LR --init $R/$PREV/ckpt_last.pt --layers $ALL
