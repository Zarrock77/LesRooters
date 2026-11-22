#!/bin/bash
# Stage 3: single-method decoder-branch specialists (new seeds, all three source banks) for later averaging.
F=/srv/lesrooters/fable
R=$F/runs
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
mkdir -p $R/v8_ps $R/v8_tm
cp $R/ft_b/init.pt $R/v8_ps/init.pt
cp $R/ft_b/init.pt $R/v8_tm/init.pt
(until [ -f $R/ft_d/ALL_DONE ]; do sleep 3; done
 bash $F/code/run_train.sh v8_tm 1 --flag --wm-branch --methods trustmark --seed 7 --iters 6000 --eval-every 1000 --init $R/v8_tm/init.pt --layers $ALL) &
(until [ -f $R/combo_d/PLAIN_DONE ]; do sleep 3; done
 bash $F/code/run_train.sh v8_ps 0 --flag --wm-branch --methods pixelseal --seed 8 --iters 6000 --eval-every 1000 --init $R/v8_ps/init.pt --layers $ALL) &
wait
