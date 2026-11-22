#!/bin/bash
# ft_v fold 3 ran out of memory when v13_ps started between two folds; run it again and close the run.
F=/srv/lesrooters/fable
R=$F/runs
rm -rf $R/ft_v/fold_3
bash $F/code/run_ft.sh ft_v v12_pt 1 3 -- --cutmix 0.5 --objects 0.7 --pure 0.85
until [ $(ls $R/ft_v/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ]; do sleep 5; done
echo done > $R/ft_v/ALL_DONE
