#!/bin/bash
# Segment the official images where the message does not decode in the full frame: train first (needed for the
# out-of-fold evaluation), then test. Two shards each.
C=/srv/lesrooters/fable/code
O=/srv/lesrooters/fable/runs/sam
PPS=${1:-24}
bash $C/run_sam_shard.sh 0 train 0 2 $PPS &
bash $C/run_sam_shard.sh 0 train 1 2 $PPS &
wait
echo done > $O/TRAIN_DONE
bash $C/run_sam_shard.sh 0 test 0 2 $PPS &
bash $C/run_sam_shard.sh 1 test 1 2 $PPS &
wait
echo done > $O/SAM_ALL_DONE
