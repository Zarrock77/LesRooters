#!/bin/bash
# Stage 22: fold fine-tune from a finished pre-training, two streams on one GPU.
# usage: stage22.sh <gpu> <run name> <pre-training run> <marker file to wait for, or -> [finetune_cv.py arguments]
# The pre-training must have printed TRAIN_DONE; the optional marker (another run's ALL_DONE) frees the GPU first.
F=/srv/lesrooters/fable
R=$F/runs
GPU=$1; NAME=$2; INIT=$3; MARK=$4; shift 4
until grep -q TRAIN_DONE $R/$INIT/train.log 2>/dev/null; do sleep 5; done
[ "$MARK" = "-" ] || until [ -f $MARK ]; do sleep 5; done
[ -e $R/$NAME ] && { echo "run name $NAME already exists"; exit 1; }
mkdir -p $R/$NAME
cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- --cutmix 0.5 --objects 0.7 --pure 0.85 "$@" &
bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- --cutmix 0.5 --objects 0.7 --pure 0.85 "$@" &
wait
[ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
