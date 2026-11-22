#!/bin/bash
# combo_k: third TrustMark round (ft_tm15, from v15_tm) on top of combo_j1's candidates, and the tile-decoder fusion of
# the same PixelSeal maps (combo_j1_tilefus) offered for PixelSeal. Result: combo_k_cal2 (fused_combo.sh).
F=/srv/lesrooters/fable
R=$F/runs
ready() { for r in "$@"; do [ -f $R/${r}_tta/REPREDICTED ] || return 1; done; }
until ready ft_tm15 ft_tm14 && [ -f $R/combo_j1_tilefus/FUSION_DONE ]; do sleep 5; done
bash $F/code/fused_combo.sh combo_k 0 combo_j1_tilefus \
  'pixelseal=ft_g_tta*ft_p_tta*ft_v_tta' \
  'trustmark=ft_tm15_tta+ft_tm15_tta*ft_tm14_tta+ft_tm15_tta*ft_tm14_tta*ft_u_tta+ft_tm14_tta*ft_u_tta+ft_tm14_tta' \
  'stegastamp=ft_k_tta*ft_t_tta'
