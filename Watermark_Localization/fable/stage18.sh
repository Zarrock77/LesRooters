#!/bin/bash
# Stage 18: official mask shapes in the synthetic half of the fine-tune batch (--shapes: holes and object regions
# shaped like the masks of the fold's own training pictures, never the outer fold), on GPU 0:
#   ft_sh = ft_u's recipe + --shapes 0.6 (TrustMark, from v12_tm, fit + tune 600 steps, same seed), two streams
#   ft_t2 = StegaStamp only from v9_gen, seed 1, kit-recipe compositions, one stream
# A first start of this stage was stopped while waiting, but one of its sub-shells survived and later started the
# same work under already used run names (ft_m), filling the GPU: everything was stopped and restarted from here.
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7 --pure 0.85"
GPU=${1:-0}
ft() {                # ft <name> <init run> <gpu> [finetune_cv.py arguments]: two streams
  local NAME=$1 INIT=$2 G=$3; shift 3
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  bash $F/code/run_ft.sh $NAME $INIT $G 0 2 4 -- $ARGS "$@" &
  bash $F/code/run_ft.sh $NAME $INIT $G 1 3 -- $ARGS "$@" &
  wait
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}
ft ft_sh v12_tm $GPU --all-inner --steps 600 --shapes 0.6 &
( mkdir -p $R/ft_t2; [ -f $R/ft_t2/init.pt ] || cp $R/v9_gen/ckpt_last.pt $R/ft_t2/init.pt
  bash $F/code/run_ft.sh ft_t2 v9_gen $GPU 0 1 2 3 4 -- $ARGS --methods stegastamp --seed 1
  [ $(ls $R/ft_t2/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_t2/ALL_DONE ) &
wait
