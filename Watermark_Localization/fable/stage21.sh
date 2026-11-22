#!/bin/bash
# Stage 21: PixelSeal + TrustMark pre-training on kit-recipe compositions whose generated holes are, half of the
# time, blob fields (--fields 0.5): looked at in 512 px, many official hole masks of full-frame pictures are several
# scattered smooth blobs, while the generator mostly drew one compact region. Half of the remaining out-of-fold error
# is in the holes of full-frame PixelSeal and TrustMark pictures. v15_pt replaces v14_pt (stopped at step 600).
F=/srv/lesrooters/fable
R=$F/runs
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
bash $F/code/run_train.sh v15_pt ${1:-1} --flag --wm-branch --methods pixelseal,trustmark --objects 0.7 --pure 0.85 --fields 0.5 --seed 35 --iters 3000 --eval-every 500 --lr 2e-4 --init $R/v12_pt/ckpt_last.pt --layers $ALL
