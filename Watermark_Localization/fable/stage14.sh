#!/bin/bash
# Stage 14: the kit-recipe compositions (--pure) in the fine-tune gave TrustMark +0.02 (ft_u 0.750 against ft_h2 0.730)
# and StegaStamp +0.009 (ft_t 0.914 against ft_k 0.905). Same recipe for PixelSeal, and a second TrustMark run.
#   ft_v (PixelSeal + TrustMark, from v12_pt)                                GPU 1
#   ft_y (TrustMark, from v10_tm, fit + tune 600 steps, independent seed)    GPU 0
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
ft ft_v v12_pt 1 &
ft ft_y v10_tm 0 --all-inner --steps 600 --seed 1 &
wait
