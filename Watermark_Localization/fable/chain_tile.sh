#!/bin/bash
# usage: chain_tile.sh <base combo> <gpu>
# Cross-fitted fusion of the base combination with the PixelSeal tile-decoder presence maps, then the usual
# post-processing and segment selection. The fused run is only a candidate: it is compared out-of-fold with the base.
BASE=$1; GPU=${2:-1}
F=/srv/lesrooters/fable
R=$F/runs
until [ -f $R/$BASE/SAM_POST_DONE ] && [ -f $R/tilemaps/DONE ]; do sleep 5; done
bash $F/code/run_fusion.sh $BASE $GPU tilefus --occ $R/tilemaps
cp $R/$BASE/replay.json $R/${BASE}_tilefus/replay.json
bash $F/code/run_sam_post.sh ${BASE}_tilefus
