#!/bin/bash
# Stage 20: official mask shapes for PixelSeal. ft_sp = ft_v's recipe (PixelSeal + TrustMark, from v12_pt, 800 steps,
# same seed, checkpoint chosen on tune) + --shapes 0.6: direct comparison with ft_v (PixelSeal 0.6452 out-of-fold).
F=/srv/lesrooters/fable
R=$F/runs
GPU=${1:-0}
until [ -f $R/ft_sh/ALL_DONE ]; do sleep 5; done
mkdir -p $R/ft_sp
[ -f $R/ft_sp/init.pt ] || cp $R/v12_pt/ckpt_last.pt $R/ft_sp/init.pt
bash $F/code/run_ft.sh ft_sp v12_pt $GPU 0 2 4 -- --cutmix 0.5 --objects 0.7 --pure 0.85 --shapes 0.6 &
bash $F/code/run_ft.sh ft_sp v12_pt $GPU 1 3 -- --cutmix 0.5 --objects 0.7 --pure 0.85 --shapes 0.6 &
wait
[ $(ls $R/ft_sp/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_sp/ALL_DONE
