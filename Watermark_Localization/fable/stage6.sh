#!/bin/bash
# Stage 6 (replaces the queue of stage 5; the two running pre-trainings are left alone).
#   ft_m : same recipe as ft_g plus unlabelled official pictures labelled by combo_d's fold models (A/B against ft_g)
#   ft_k : fold fine-tunes of the StegaStamp specialist v10_st, when it ends
#   ft_j : fold fine-tunes of the PixelSeal specialist v10_ps, when it ends
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7"

finetune() {          # finetune <ft name> <init run> <gpu> [extra finetune_cv.py arguments]
  local NAME=$1 INIT=$2 GPU=$3; shift 3
  until ! pgrep -f "train_det.py.*runs/$INIT " > /dev/null; do sleep 5; done
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- $ARGS "$@" &
  bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- $ARGS "$@" &
  wait
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}

(until [ -f $R/ft_h/ALL_DONE ]; do sleep 5; done
 finetune ft_m v9_obj 1 --teacher $R/combo_d --pseudo 8) &
finetune ft_k v10_st 1 &
finetune ft_j v10_ps 0 &
wait
