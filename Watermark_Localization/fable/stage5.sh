#!/bin/bash
# Stage 5: single-method specialists trained with object-shaped synthetic edits, then their fold fine-tunes.
#   v10_st (StegaStamp, from v9_gen)  -> ft_k   GPU 1, starts now
#   v10_ps (PixelSeal, from v9_obj)   -> ft_j   GPU 0, after ft_g
#   v10_tm (TrustMark, from v8_tm)    -> ft_l   GPU 1, after ft_h
F=/srv/lesrooters/fable
R=$F/runs
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
ARGS="--cutmix 0.5 --objects 0.7"

finetune() {          # finetune <ft name> <init run> <gpu>
  local NAME=$1 INIT=$2 GPU=$3
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- $ARGS &
  bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- $ARGS &
  wait
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}

(bash $F/code/run_train.sh v10_st 1 --flag --methods stegastamp --objects 0.7 --seed 12 --iters 4000 --eval-every 1000 --lr 2e-4 --init $R/v9_gen/ckpt_last.pt --layers $ALL
 finetune ft_k v10_st 1) &
(until [ -f $R/ft_g/ALL_DONE ]; do sleep 5; done
 bash $F/code/run_train.sh v10_ps 0 --flag --wm-branch --methods pixelseal --objects 0.7 --seed 11 --iters 4000 --eval-every 1000 --lr 2e-4 --init $R/v9_obj/ckpt_last.pt --layers $ALL
 finetune ft_j v10_ps 0) &
(until [ -f $R/ft_h/ALL_DONE ]; do sleep 5; done
 bash $F/code/run_train.sh v10_tm 1 --flag --wm-branch --methods trustmark --objects 0.7 --seed 13 --iters 3000 --eval-every 1000 --lr 1.5e-4 --init $R/v8_tm/ckpt_last.pt --layers $ALL
 finetune ft_l v10_tm 1) &
wait
