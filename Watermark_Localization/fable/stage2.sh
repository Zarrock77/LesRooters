#!/bin/bash
# Stage 2: stop the plateaued v4 training, then (GPU 1) generalist fine-tune with real-image recombination and a
# StegaStamp-only continuation of v1; (GPU 0) specialist fine-tune with recombination from the last v4 checkpoint.
set -uo pipefail
F=/srv/lesrooters/fable
R=$F/runs
pkill -f 'out /srv/lesrooters/fable/runs/v4_wm'
sleep 3
mkdir -p $R/ft_d $R/ft_e $R/v6_stega
[ -f $R/ft_d/init.pt ] || cp $R/ft_b/init.pt $R/ft_d/init.pt
[ -f $R/ft_e/init.pt ] || cp $R/v4_wm/ckpt_last.pt $R/ft_e/init.pt
[ -f $R/v6_stega/init.pt ] || cp $R/ft_b/init.pt $R/v6_stega/init.pt
(bash $F/code/run_ft.sh ft_d v1_flag 1 0 1 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_d v1_flag 1 2 3 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_d v1_flag 1 4 -- --cutmix 0.5 &
 wait; echo FT_D_DONE > $R/ft_d/ALL_DONE) &
(bash $F/code/run_ft.sh ft_e v4_wm 0 0 1 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_e v4_wm 0 2 3 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_e v4_wm 0 4 -- --cutmix 0.5 &
 wait; echo FT_E_DONE > $R/ft_e/ALL_DONE) &
bash $F/code/run_train.sh v6_stega 1 --flag --methods stegastamp --iters 8000 --eval-every 1000 --init $R/v6_stega/init.pt \
  --layers /scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl &
wait
