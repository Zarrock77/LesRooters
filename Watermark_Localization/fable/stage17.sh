#!/bin/bash
# Stage 17: ft_r8 = TrustMark fit + tune fine-tune with 8 official + 24 kit-recipe synthetic pictures per batch
# (the opposite direction of ft_w, which used 28 official and was worse). One stream on GPU 1 after ft_t2.
F=/srv/lesrooters/fable
R=$F/runs
until [ -f $R/ft_t2/ALL_DONE ]; do sleep 5; done
mkdir -p $R/ft_r8
[ -f $R/ft_r8/init.pt ] || cp $R/v12_tm/ckpt_last.pt $R/ft_r8/init.pt
bash $F/code/run_ft.sh ft_r8 v12_tm ${1:-1} 0 1 2 3 4 -- --cutmix 0.5 --objects 0.7 --pure 0.85 --real 8 --all-inner --steps 600 --seed 3
[ $(ls $R/ft_r8/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/ft_r8/ALL_DONE
