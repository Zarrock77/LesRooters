#!/bin/bash
# Stage 11b: finish the fine-tunes of stage 11 with two streams per run, one on each GPU (GPU 0 became free).
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7 --pure 0.85"
for k in 1 2 3 4; do for d in ft_t ft_u; do [ -f $R/$d/fold_$k/best.pt ] || rm -rf $R/$d/fold_$k; done; done
bash $F/code/run_ft.sh ft_t v12_st 0 1 3 -- $ARGS --methods stegastamp &
bash $F/code/run_ft.sh ft_t v12_st 1 2 4 -- $ARGS --methods stegastamp &
bash $F/code/run_ft.sh ft_u v12_tm 0 1 3 -- $ARGS --all-inner --steps 600 &
bash $F/code/run_ft.sh ft_u v12_tm 1 2 4 -- $ARGS --all-inner --steps 600 &
wait
for d in ft_t ft_u; do [ $(ls $R/$d/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$d/ALL_DONE; done
