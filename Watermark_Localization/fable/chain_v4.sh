#!/bin/bash
# When the AI-looking banks are ready: stop v1, start v4 (decoder branches, warm start) and fine-tune v1's folds.
set -uo pipefail
F=/srv/lesrooters/fable
R=$F/runs
mkdir -p $R/v4_wm
until [ -f /scratch/fable/syn_sdxl/REGEN_DONE ] && [ -f /scratch/fable/syn_vae/BANK_DONE ] && [ -f /scratch/fable/syn_sdxl/BANK_DONE ]; do sleep 2; done
pkill -f 'out /srv/lesrooters/fable/runs/v1_flag'
sleep 3
cp $R/v1_flag/ckpt_last.pt $R/v4_wm/init.pt
bash $F/code/run_train.sh v4_wm 1 --flag --wm-branch --iters 12000 --eval-every 1000 --init $R/v4_wm/init.pt \
  --layers /scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl &
bash $F/code/run_ft.sh ft_b v1_flag 0 0 1 2 3 4
bash $F/code/run_post.sh ft_b 0
wait
