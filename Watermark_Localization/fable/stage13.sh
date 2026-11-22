#!/bin/bash
# Stage 13: PixelSeal detector with an unpooled full-resolution branch (v13_ps), continued from v9_obj.
# PixelSeal transfers from synthetic to official pictures almost unchanged (0.64 against 0.62), so what limits it is
# the detector itself on the synthetic task: a branch that sees the top octave is tested there first.
# The branch needs about 40 GB at batch 32 (a first start next to two fine-tune streams ran out of memory), so the
# script waits until the chosen GPU has that much free.
F=/srv/lesrooters/fable
R=$F/runs
GPU=${1:-1}
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
# Start only when the fine-tune sharing this GPU is finished: waiting on free memory alone started it between two
# folds and both jobs ran out of memory.
until [ -f $R/ft_v/ALL_DONE ]; do sleep 5; done
bash $F/code/run_train.sh v13_ps $GPU --flag --wm-branch --hr-branch --methods pixelseal --objects 0.7 --pure 0.85 --seed 41 --batch 24 --iters 2500 --eval-every 500 --lr 2e-4 --init $R/v9_obj/ckpt_last.pt --layers $ALL
