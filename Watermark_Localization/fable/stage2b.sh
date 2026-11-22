#!/bin/bash
# Restart of stage 2 with bounded CPU threads (the first attempt starved the machine: load 190 on 48 cores).
set -uo pipefail
F=/srv/lesrooters/fable
R=$F/runs
pkill -f 'code/stage2.s[h]'; pkill -f 'code/restart_v6.s[h]'; pkill -f 'code/pause_bg.s[h]'
pkill -f 'run_ft.sh ft_[de] '; sleep 1
pkill -f 'finetune_cv.py.*runs/ft_[de]/'; sleep 4
rm -rf $R/ft_d/fold_* $R/ft_e/fold_* $R/ft_d/ALL_DONE $R/ft_e/ALL_DONE
(bash $F/code/run_ft.sh ft_d v1_flag 1 0 1 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_d v1_flag 1 2 3 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_d v1_flag 1 4 -- --cutmix 0.5 &
 wait; echo FT_D_DONE > $R/ft_d/ALL_DONE) &
(bash $F/code/run_ft.sh ft_e v4_wm 0 0 1 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_e v4_wm 0 2 3 -- --cutmix 0.5 &
 bash $F/code/run_ft.sh ft_e v4_wm 0 4 -- --cutmix 0.5 &
 wait; echo FT_E_DONE > $R/ft_e/ALL_DONE) &
wait
for p in $(pgrep -f 'rect_refine.py.*--methods pixelseal --global-sca[n]'); do kill -CONT $p; done
bash $F/code/run_train.sh v6_stega 1 --flag --methods stegastamp --iters 8000 --eval-every 1000 --init $R/v6_stega/init.pt \
  --layers /scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
