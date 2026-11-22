#!/bin/bash
# Stage 19: longer pre-training on kit-recipe compositions (--pure 0.85).
# Evidence: the TrustMark fine-tune from v12_tm (v10_tm + 1000 such steps) scores 0.750 out-of-fold, the same
# fine-tune from v10_tm itself 0.737 (ft_u against ft_y); PixelSeal from v12_pt (v9_obj + 1000 steps) 0.649 on the
# first folds against 0.639 from v9_obj. Both pre-trainings had been stopped at 1000 steps because the score of the
# synthetic-only model on official pictures did not move; the fine-tuned score does.
#   v14_tm: v12_tm + 3000 steps (TrustMark only)      v14_pt: v12_pt + 3000 steps (PixelSeal + TrustMark)
# The full-resolution branch test (v13_ps, stage13.sh) is dropped: the backbone's 4x4 stride-4 stem is linear and
# loses nothing, and the PixelSeal residual is computed at 256 px, so it has no energy in the top octave.
F=/srv/lesrooters/fable
R=$F/runs
GPU=${1:-1}
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
until [ -f $R/ft_v/ALL_DONE ]; do sleep 5; done
bash $F/code/run_train.sh v14_tm $GPU --flag --wm-branch --methods trustmark --objects 0.7 --pure 0.85 --seed 34 --iters 3000 --eval-every 500 --lr 1.5e-4 --init $R/v12_tm/ckpt_last.pt --layers $ALL &
bash $F/code/run_train.sh v14_pt $GPU --flag --wm-branch --methods pixelseal,trustmark --objects 0.7 --pure 0.85 --seed 35 --iters 3000 --eval-every 500 --lr 2e-4 --init $R/v12_pt/ckpt_last.pt --layers $ALL &
wait
