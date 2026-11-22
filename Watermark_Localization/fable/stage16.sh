#!/bin/bash
# Stage 16: ft_v2 = the kit-recipe PixelSeal + TrustMark fine-tune again, longer (1400 steps instead of 800) and with
# another seed. In ft_v the tune score was still rising at the last step in the first folds (0.6695 and 0.6893 at
# step 800), so the checkpoint choice on tune may land later. Same init (v12_pt), selection on tune only.
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7 --pure 0.85"
GPU=${1:-0}
until [ -f $R/ft_z/ALL_DONE ]; do sleep 5; done
mkdir -p $R/ft_v2
[ -f $R/ft_v2/init.pt ] || cp $R/v12_pt/ckpt_last.pt $R/ft_v2/init.pt
bash $F/code/run_ft.sh ft_v2 v12_pt $GPU 0 2 4 -- $ARGS --steps 1400 --seed 1 &
bash $F/code/run_ft.sh ft_v2 v12_pt $GPU 1 3 -- $ARGS --steps 1400 --seed 1 &
wait
[ $(ls $R/ft_v2/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_v2/ALL_DONE
