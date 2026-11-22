#!/bin/bash
# Stage 10: pre-training on compositions built as the starter kit describes the dataset (the unwatermarked part is
# always the same clean picture, mostly hard edges, nothing foreign pasted): --pure 0.85.
#   v12_pt (PixelSeal+TrustMark, from v9_obj)   v12_tm (TrustMark only, from v10_tm)   v12_st (StegaStamp only, from v10_st)
F=/srv/lesrooters/fable
R=$F/runs
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
GPU=${1:-1}
bash $F/code/run_train.sh v12_pt $GPU --flag --wm-branch --methods pixelseal,trustmark --objects 0.7 --pure 0.85 --seed 31 --iters 2500 --eval-every 500 --lr 2e-4 --init $R/v9_obj/ckpt_last.pt --layers $ALL &
bash $F/code/run_train.sh v12_tm $GPU --flag --wm-branch --methods trustmark --objects 0.7 --pure 0.85 --seed 32 --iters 2500 --eval-every 500 --lr 1.5e-4 --init $R/v10_tm/ckpt_last.pt --layers $ALL &
bash $F/code/run_train.sh v12_st $GPU --flag --methods stegastamp --objects 0.7 --pure 0.85 --seed 33 --iters 2500 --eval-every 500 --lr 2e-4 --init $R/v10_st/ckpt_last.pt --layers $ALL &
wait
