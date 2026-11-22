#!/bin/bash
# v4: decoder-branch detector focused on the two weak-signal methods (StegaStamp is already solved by v1).
F=/srv/lesrooters/fable
R=$F/runs
pkill -f 'out /srv/lesrooters/fable/runs/v4_wm'
sleep 3
bash $F/code/run_train.sh v4_wm 0 --flag --wm-branch --methods pixelseal,trustmark --iters 12000 --eval-every 1000 \
  --init $R/v4_wm/init.pt --layers /scratch/fable/syn,/scratch/fable/syn_vae
