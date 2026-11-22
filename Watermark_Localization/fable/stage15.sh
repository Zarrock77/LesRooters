#!/bin/bash
# Stage 15: independent second runs of the kit-recipe fine-tunes, from other pre-trainings, for averaging.
#   ft_z  (TrustMark, from v8_tm, fit + tune 600 steps, seed 2)   GPU 0, after ft_y
#   ft_t2 (StegaStamp only, from v9_gen, seed 1)                   GPU 1, after ft_v
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7 --pure 0.85"
ft() {                # ft <name> <init run> <gpu> [finetune_cv.py arguments]
  local NAME=$1 INIT=$2 GPU=$3; shift 3
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- $ARGS "$@" &
  bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- $ARGS "$@" &
  wait
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}
( until [ -f $R/ft_y/ALL_DONE ]; do sleep 5; done; ft ft_z v8_tm 0 --all-inner --steps 600 --seed 2 ) &
# ft_t2 as a single stream: it shares GPU 1 with the full-resolution pre-training v13_ps (about 30-40 GB).
( until [ -f $R/ft_v/ALL_DONE ]; do sleep 5; done; mkdir -p $R/ft_t2; [ -f $R/ft_t2/init.pt ] || cp $R/v9_gen/ckpt_last.pt $R/ft_t2/init.pt
  bash $F/code/run_ft.sh ft_t2 v9_gen 1 0 1 2 3 4 -- $ARGS --methods stegastamp --seed 1
  [ $(ls $R/ft_t2/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_t2/ALL_DONE ) &
wait
