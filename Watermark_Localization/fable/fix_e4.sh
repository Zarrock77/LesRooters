#!/bin/bash
# One specialist fold died of GPU memory exhaustion. Redo it once its sibling streams are finished, then build combo_d.
F=/srv/lesrooters/fable
R=$F/runs
pkill -f 'code/chain_combo2.s[h] combo_d'
until [ -f $R/ft_e/ALL_DONE ]; do sleep 3; done
rm -rf $R/ft_e/fold_4
bash $F/code/run_ft.sh ft_e v4_wm 0 4 -- --cutmix 0.5
bash $F/code/chain_combo2.sh combo_d 0 $R/ft_d/ALL_DONE $R/ft_e/fold_4/best.pt 'pixelseal=ft_c+ft_e+ft_c*ft_e' 'trustmark=ft_c*ft_e+ft_c*ft_e*ft_d+ft_c' 'stegastamp=ft_b*ft_d+ft_d+ft_b'
