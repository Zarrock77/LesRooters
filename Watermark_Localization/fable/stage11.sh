#!/bin/bash
# Stage 11: fold fine-tunes of the pre-trainings made with kit-recipe compositions (--pure), as each one ends.
#   ft_t (StegaStamp only, from v12_st)                         GPU 1
#   ft_u (TrustMark, from v12_tm, fit + tune for 600 steps)     GPU 1
#   ft_v (PixelSeal + TrustMark, from v12_pt)                   GPU 0
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7 --pure 0.85"

ft() {                # ft <name> <init run> <gpu> [finetune_cv.py arguments]
  local NAME=$1 INIT=$2 GPU=$3; shift 3
  until ! pgrep -f "train_det.py.*runs/$INIT " > /dev/null; do sleep 5; done
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  # one stream per run: the pre-trainings still hold most of the GPU memory when the first fine-tunes start
  bash $F/code/run_ft.sh $NAME $INIT $GPU 0 1 2 3 4 -- $ARGS "$@"
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}

ft ft_t v12_st 1 --methods stegastamp &
ft ft_u v12_tm 1 --all-inner --steps 600 &
# ft_v (PixelSeal + TrustMark from v12_pt) dropped: v12_pt brought nothing for PixelSeal after 1000 steps and was stopped
wait
