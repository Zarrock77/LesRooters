#!/bin/bash
# Stage 23: longer kit-recipe pre-training for StegaStamp as well (v14_st = v12_st + 3000 steps, blob-field holes).
F=/srv/lesrooters/fable
R=$F/runs
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
[ -e $R/v14_st ] && { echo "v14_st exists"; exit 1; }
bash $F/code/run_train.sh v14_st ${1:-1} --flag --methods stegastamp --objects 0.7 --pure 0.85 --fields 0.5 --seed 36 --iters 3000 --eval-every 500 --lr 2e-4 --init $R/v12_st/ckpt_last.pt --layers $ALL
