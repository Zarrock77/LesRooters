#!/bin/bash
# usage: chain_combo.sh <specialist ft name> <v4 run> <generalist ft name> <combo name> <gpu A> [gpu B] [wait-for pattern in the v4 log]
# Fine-tune the decoder-branch specialist on the five frozen folds (two streams), then take each method from the
# best run out-of-fold, snap rectangles, replay, and write the submission artifacts.
set -uo pipefail
FT=$1; V=$2; GEN=$3; COMBO=$4; GPU=${5:-1}; GPU2=${6:-$GPU}; WAITFOR=${7:-}
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
mkdir -p $R/$FT
if [ -n "$WAITFOR" ]; then until grep -q "$WAITFOR" $R/$V/train.log; do sleep 2; done; sleep 3; fi
[ -f $R/$FT/init.pt ] || cp $R/$V/ckpt_last.pt $R/$FT/init.pt
bash $F/code/run_ft.sh $FT $V $GPU 0 1 2 &
bash $F/code/run_ft.sh $FT $V $GPU2 3 4 &
wait
until [ -f $R/$GEN/fold_4/best.pt ]; do sleep 3; done
cd $K
$K/.venv/bin/python $F/code/combine_methods.py $R $COMBO pixelseal=$FT+$GEN trustmark=$FT+$GEN stegastamp=$GEN > $R/${COMBO}_combine.log 2>&1
COMPONENTS=$(cat $R/$COMBO/components.txt) bash $F/code/run_post.sh $COMBO $GPU
