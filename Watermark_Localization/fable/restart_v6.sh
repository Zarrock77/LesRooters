#!/bin/bash
# Free the GPU memory held by the paused StegaStamp continuation; restart it when the stage-2 fine-tunes are done.
F=/srv/lesrooters/fable
R=$F/runs
for p in $(pgrep -f 'train_det.py.*runs/v6_steg[a]'); do kill -CONT $p; kill $p; done
until [ -f $R/ft_d/ALL_DONE ] && [ -f $R/ft_e/ALL_DONE ]; do sleep 3; done
bash $F/code/run_train.sh v6_stega 1 --flag --methods stegastamp --iters 8000 --eval-every 1000 --init $R/v6_stega/init.pt \
  --layers /scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
