#!/bin/bash
# combo_j through fast_combo.sh (one rectangle search, on the calibrated maps): the PixelSeal fine-tune of the longer
# pre-training (ft_ps15) added to combo_j1's candidates. It waits only for ft_ps15: the second TrustMark seed and
# the StegaStamp fine-tune share a GPU with a pre-training and finish later, they enter combo_k. Result: combo_j_cal2.
F=/srv/lesrooters/fable
R=$F/runs
ready() { for r in "$@"; do [ -f $R/${r}_tta/REPREDICTED ] || return 1; done; }
until ready ft_tm14 ft_sh ft_ps15 && [ -f $R/combo_j1_cal2/SAM_POST_DONE ]; do sleep 5; done
bash $F/code/fast_combo.sh combo_j 0 \
  'pixelseal=ft_ps15_tta+ft_ps15_tta*ft_v_tta+ft_g_tta*ft_p_tta*ft_ps15_tta+ft_g_tta*ft_p_tta*ft_v_tta*ft_ps15_tta+ft_g_tta*ft_p_tta+ft_g_tta*ft_p_tta*ft_v_tta' \
  'trustmark=ft_tm14_tta+ft_tm14_tta*ft_u_tta+ft_tm14_tta*ft_u_tta*ft_sh_tta+ft_u_tta*ft_sh_tta+ft_u_tta' \
  'stegastamp=ft_k_tta*ft_t_tta'
