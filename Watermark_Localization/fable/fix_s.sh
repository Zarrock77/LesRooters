#!/bin/bash
# One stream of the StegaStamp-only fine-tune ran out of GPU memory: redo its folds once the pre-training has left the GPU.
F=/srv/lesrooters/fable
R=$F/runs
until ! pgrep -f "train_det.py.*runs/v11_obj " > /dev/null; do sleep 5; done
# ... and its sibling stream (folds 1 and 3) has finished: four jobs would not fit on the GPU
until [ -f $R/ft_s/fold_1/best.pt ] && [ -f $R/ft_s/fold_3/best.pt ]; do sleep 5; done
for k in 0 2 4; do [ -f $R/ft_s/fold_$k/best.pt ] || rm -rf $R/ft_s/fold_$k; done
FOLDS=$(for k in 0 2 4; do [ -f $R/ft_s/fold_$k/best.pt ] || printf '%s ' $k; done)
[ -n "$FOLDS" ] && bash $F/code/run_ft.sh ft_s v9_gen 0 $FOLDS -- --cutmix 0.5 --objects 0.7 --methods stegastamp
until [ $(ls $R/ft_s/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ]; do sleep 5; done
echo done > $R/ft_s/ALL_DONE
