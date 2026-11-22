#!/bin/bash
# Stage 9: fine-tunes trained on fit + tune (26 % more official pictures) for a fixed number of steps = median step
# selected on tune by the corresponding earlier runs. No selection, the outer fold is scored once.
#   ft_g2 (PixelSeal+TrustMark, from v9_obj, 600 steps; ft_g selected 700/800/600/500/600)   GPU 1
#   ft_h2 (TrustMark, from v8_tm, 600 steps; ft_h selected 800/700/600/500/300)             GPU 0
#   ft_s2 (StegaStamp only, from v9_gen, 500 steps; ft_s selected 400/700/./500/.)          GPU 0, after ft_h2
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7 --all-inner"

ft() {                # ft <name> <init run> <gpu> [finetune_cv.py arguments]
  local NAME=$1 INIT=$2 GPU=$3; shift 3
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- $ARGS "$@" &
  bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- $ARGS "$@" &
  wait
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}

# the single-stream runs of stage 8 may still hold memory: wait for them
(until [ -f $R/ft_l/ALL_DONE ]; do sleep 5; done; ft ft_g2 v9_obj 1 --steps 600) &
(until [ -f $R/ft_q/ALL_DONE ] && [ -f $R/ft_r/ALL_DONE ]; do sleep 5; done
 ft ft_h2 v8_tm 0 --steps 600
 ft ft_s2 v9_gen 0 --methods stegastamp --steps 500) &
wait
