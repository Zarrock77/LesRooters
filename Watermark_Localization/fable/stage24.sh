#!/bin/bash
# Stage 24: keep pre-training TrustMark on kit-recipe compositions while the fine-tunes of v14_tm run:
# v15_tm = v14_tm + 3000 steps, with blob-field holes. For a later round, if the gain keeps growing with steps.
F=/srv/lesrooters/fable
R=$F/runs
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
until grep -q TRAIN_DONE $R/v14_tm/train.log 2>/dev/null; do sleep 5; done
[ -e $R/v15_tm ] && { echo "v15_tm exists"; exit 1; }
bash $F/code/run_train.sh v15_tm ${1:-1} --flag --wm-branch --methods trustmark --objects 0.7 --pure 0.85 --fields 0.5 --seed 37 --iters 3000 --eval-every 500 --lr 1.5e-4 --init $R/v14_tm/ckpt_last.pt --layers $ALL
