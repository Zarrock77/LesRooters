#!/bin/bash
# usage: chain_combo2.sh <combo name> <gpu> <wait file 1> <wait file 2> pixelseal=<cands> trustmark=<cands> stegastamp=<cands>
# Wait for the listed fine-tunes, take per method the best out-of-fold candidate (single runs or averages), post-process.
set -uo pipefail
COMBO=$1; GPU=$2; W1=$3; W2=$4; shift 4
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
until [ -f $W1 ] && [ -f $W2 ]; do sleep 3; done
cd $K
$K/.venv/bin/python $F/code/combine_methods.py $R $COMBO "$@" > $R/${COMBO}_combine.log 2>&1
COMPONENTS=$(cat $R/$COMBO/components.txt) bash $F/code/run_post.sh $COMBO $GPU
