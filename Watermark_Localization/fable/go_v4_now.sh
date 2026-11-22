#!/bin/bash
# Start v4 (decoder branches) immediately on GPU 0 from v1's current weights, and fine-tune v1's folds on GPU 1.
set -uo pipefail
F=/srv/lesrooters/fable
R=$F/runs
pkill -f 'code/chain_v4.sh'
pkill -f 'gen_ai_sources.py --mode regen'
pkill -f 'out /srv/lesrooters/fable/runs/v1_flag'
sleep 4
mkdir -p $R/v4_wm
cp $R/v1_flag/ckpt_last.pt $R/v4_wm/init.pt
(bash $F/code/run_train.sh v4_wm 0 --flag --wm-branch --iters 12000 --eval-every 1000 --init $R/v4_wm/init.pt \
  --layers /scratch/fable/syn,/scratch/fable/syn_vae > $R/v4_wm/launch.log 2>&1 &)
bash $F/code/run_ft.sh ft_b v1_flag 1 0 1 2 3 4
bash $F/code/run_post.sh ft_b 1
