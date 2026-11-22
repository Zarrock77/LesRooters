#!/bin/bash
# usage: run_bank.sh <bank dir containing clean.npy> [rects per source]
# Embeds the three methods on both GPUs with several shards each, then writes BANK_DONE.
set -euo pipefail
D=$1; RECTS=${2:-4}
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
PY=$K/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=3
cd $K
mkdir -p $D/logs
for M in pixelseal trustmark stegastamp; do $PY $C/gen_layers.py --kit $K --data $D --method $M --init > $D/logs/init_$M.log 2>&1; done
launch() { # method shards
  for S in $(seq 0 $(($2 - 1))); do
    $PY -u $C/gen_layers.py --kit $K --data $D --method $1 --rects $RECTS --shard $S --shards $2 --device cuda:$((S % 2)) > $D/logs/gen_$1_$S.log 2>&1 &
  done
}
launch trustmark 8
launch pixelseal 3
launch stegastamp 3
wait
ls $D/trustmark/DONE_7 $D/pixelseal/DONE_2 $D/stegastamp/DONE_2 > /dev/null
echo BANK_DONE > $D/BANK_DONE
