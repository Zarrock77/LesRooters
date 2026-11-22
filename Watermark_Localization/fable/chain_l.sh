#!/bin/bash
# combo_l: combo_k2 plus the second TrustMark lineage (ft_tm14c, from v14b_tm = v8_tm + 3000 kit-recipe steps) for a
# more diverse TrustMark average. Result: combo_l_cal2 (fast_combo.sh).
F=/srv/lesrooters/fable
R=$F/runs
ready() { for r in "$@"; do [ -f $R/${r}_tta/REPREDICTED ] || return 1; done; }
until ready ft_tm15 ft_tm14c ft_st14 ft_ps15 && [ -f $R/combo_k2_cal2/SAM_POST_DONE ]; do sleep 5; done
bash $F/code/fast_combo.sh combo_l 0 \
  'pixelseal=ft_g_tta*ft_p_tta*ft_v_tta+ft_g_tta*ft_p_tta*ft_v_tta*ft_ps15_tta+ft_g_tta*ft_p_tta*ft_ps15_tta+ft_v_tta*ft_ps15_tta' \
  'trustmark=ft_tm14_tta*ft_tm14c_tta+ft_tm14_tta*ft_tm15_tta*ft_tm14c_tta+ft_tm14_tta*ft_tm14c_tta*ft_u_tta+ft_tm15_tta*ft_tm14_tta*ft_u_tta+ft_tm14_tta*ft_u_tta' \
  'stegastamp=ft_st14_tta+ft_t_tta*ft_st14_tta+ft_k_tta*ft_t_tta*ft_st14_tta+ft_k_tta*ft_t_tta'
