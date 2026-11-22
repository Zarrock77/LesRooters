#!/bin/bash
# Replace the shallow local-decoder run (stuck at chance) by the full-extractor variant.
pkill -f 'train_local.py.*runs/loc_p[s]'
sleep 2
bash /srv/lesrooters/fable/code/run_local.sh locf_ps 1 pixelseal --arch full --iters 4000 --eval-every 250 --lr 1e-4
