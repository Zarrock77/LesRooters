#!/bin/bash
# Stage 12: TrustMark fine-tune with almost only official pictures in the batch (28 real + 4 synthetic instead of
# 16 + 16). The synthetic TrustMark watermark is written by our encoder, whose residual does not match the official
# one (edge and template probes), so half a batch of it may teach the wrong appearance. Same init as ft_h (v8_tm),
# same selection on tune, outer fold scored once: direct comparison with ft_h (0.7213 out-of-fold).
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7 --real 28"
mkdir -p $R/ft_w
[ -f $R/ft_w/init.pt ] || cp $R/v8_tm/ckpt_last.pt $R/ft_w/init.pt
bash $F/code/run_ft.sh ft_w v8_tm 0 0 2 4 -- $ARGS &
bash $F/code/run_ft.sh ft_w v8_tm 1 1 3 -- $ARGS &
wait
[ $(ls $R/ft_w/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_w/ALL_DONE
