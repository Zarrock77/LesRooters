#!/bin/bash
# combo_m: next round. PixelSeal with ft_ps16 (from v16_pt), StegaStamp with the third pre-training round (ft_st15)
# and the fine-tune variant with official hole shapes (ft_st14s); TrustMark as in combo_l. Result: combo_m_cal2.
F=/srv/lesrooters/fable
R=$F/runs
ready() { for r in "$@"; do [ -f $R/${r}_tta/REPREDICTED ] || return 1; done; }
until ready ft_ps16 ft_st15 ft_st14s ft_tm14c ft_st14 ft_ps15 && [ -f $R/combo_l_cal2/SAM_POST_DONE ]; do sleep 5; done
bash $F/code/fast_combo.sh combo_m 0 \
  'pixelseal=ft_g_tta*ft_p_tta*ft_v_tta*ft_ps15_tta*ft_ps16_tta+ft_g_tta*ft_p_tta*ft_v_tta*ft_ps16_tta+ft_g_tta*ft_p_tta*ft_v_tta*ft_ps15_tta+ft_g_tta*ft_p_tta*ft_v_tta+ft_v_tta*ft_ps15_tta*ft_ps16_tta' \
  'trustmark=ft_tm14_tta*ft_tm14c_tta+ft_tm14_tta*ft_tm15_tta*ft_tm14c_tta+ft_tm14_tta*ft_tm14c_tta*ft_u_tta+ft_tm15_tta*ft_tm14_tta*ft_u_tta+ft_tm14_tta*ft_u_tta' \
  'stegastamp=ft_st15_tta+ft_st14_tta*ft_st15_tta+ft_st14_tta*ft_st15_tta*ft_st14s_tta+ft_st14_tta*ft_st14s_tta+ft_st14_tta+ft_t_tta*ft_st14_tta'
