#!/bin/bash
# Stage 6b (memory-aware replacement of stage 6; the PixelSeal-only pre-training brought nothing and was stopped).
#   ft_m : same recipe as ft_g plus unlabelled official pictures labelled by combo_d's fold models, folds 0-2 first
#   ft_k : fold fine-tunes of the StegaStamp specialist v10_st, when it ends
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7"
(until [ -f $R/ft_h/ALL_DONE ]; do sleep 5; done
 mkdir -p $R/ft_m
 [ -f $R/ft_m/init.pt ] || cp $R/v9_obj/ckpt_last.pt $R/ft_m/init.pt
 bash $F/code/run_ft.sh ft_m v9_obj 1 0 1 2 3 4 -- $ARGS --teacher $R/combo_d --pseudo 8
 [ $(ls $R/ft_m/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_m/ALL_DONE) &
(until ! pgrep -f "train_det.py.*runs/v10_st " > /dev/null; do sleep 5; done
 mkdir -p $R/ft_k
 [ -f $R/ft_k/init.pt ] || cp $R/v10_st/ckpt_last.pt $R/ft_k/init.pt
 bash $F/code/run_ft.sh ft_k v10_st 1 0 2 4 -- $ARGS &
 bash $F/code/run_ft.sh ft_k v10_st 1 1 3 -- $ARGS &
 wait
 [ $(ls $R/ft_k/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_k/ALL_DONE) &
wait
