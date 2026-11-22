#!/bin/bash
# usage: chain_fus.sh <folds-like run> <gpu>  -- after its post-processing, try the cross-fitted occlusion fusion on top.
NAME=$1; GPU=${2:-0}
R=/srv/lesrooters/fable/runs
until [ -f $R/$NAME/POST_DONE ]; do sleep 3; done
bash /srv/lesrooters/fable/code/run_fusion.sh $NAME $GPU occfus --occ $R/occ
