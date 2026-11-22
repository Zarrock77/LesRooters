#!/bin/bash
# combo_k2: as soon as the StegaStamp fine-tune of the longer pre-training (ft_st14) and the PixelSeal one (ft_ps15)
# have their predictions. No tile fusion (it loses the nested ranking once the cut is calibrated). fast_combo.sh:
# combination, replay, decision cut, rectangles and segment selection once. Result: combo_k2_cal2.
F=/srv/lesrooters/fable
R=$F/runs
ready() { for r in "$@"; do [ -f $R/${r}_tta/REPREDICTED ] || return 1; done; }
until ready ft_tm15 ft_st14 ft_ps15 && [ -f $R/combo_k_cal2/SAM_POST_DONE ]; do sleep 5; done
bash $F/code/fast_combo.sh combo_k2 0 \
  'pixelseal=ft_g_tta*ft_p_tta*ft_v_tta+ft_g_tta*ft_p_tta*ft_v_tta*ft_ps15_tta+ft_g_tta*ft_p_tta*ft_ps15_tta+ft_v_tta*ft_ps15_tta' \
  'trustmark=ft_tm15_tta*ft_tm14_tta*ft_u_tta+ft_tm15_tta*ft_tm14_tta+ft_tm14_tta*ft_u_tta+ft_tm14_tta' \
  'stegastamp=ft_st14_tta+ft_t_tta*ft_st14_tta+ft_k_tta*ft_t_tta*ft_st14_tta+ft_k_tta*ft_t_tta'
