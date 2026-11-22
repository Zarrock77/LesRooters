#!/bin/bash
# usage: chain_j.sh <j1|j>
# combo_j1: TrustMark candidates include the fine-tune from the longer kit-recipe pre-training (ft_tm14).
# combo_j : the same plus a second TrustMark seed (ft_tm14b) and the PixelSeal (ft_ps15) and StegaStamp (ft_st14,
#           ft_t2) fine-tunes of the longer pre-trainings.
# Candidates are ranked per method with the nested regime-wise cut (combine_methods.py), so PixelSeal is offered
# both the older average ft_g*ft_p (ranks cells better once its cut is calibrated) and the kit-recipe fine-tunes.
# Each combination is followed by segment selection and by the nested decision-cut calibration (<combo>_cal2),
# which is the version to compare and submit.
WHICH=${1:-j1}
F=/srv/lesrooters/fable
R=$F/runs
ready() { for r in "$@"; do [ -f $R/${r}_tta/REPREDICTED ] || return 1; done; }
if [ $WHICH = j1 ]; then
  until ready ft_tm14 ft_sh; do sleep 5; done
  bash $F/code/redo_combo.sh combo_j1 0 ft_g_tta,ft_p_tta,ft_v_tta,ft_u_tta,ft_sh_tta,ft_tm14_tta,ft_k_tta,ft_t_tta \
    'pixelseal=ft_g_tta*ft_p_tta+ft_g_tta*ft_p_tta*ft_v_tta+ft_g_tta*ft_v_tta+ft_v_tta' \
    'trustmark=ft_tm14_tta+ft_tm14_tta*ft_u_tta+ft_tm14_tta*ft_u_tta*ft_sh_tta+ft_u_tta*ft_sh_tta+ft_u_tta' 'stegastamp=ft_k_tta*ft_t_tta'
  bash $F/code/run_sam_post.sh combo_j1
  bash $F/code/chain_cal.sh combo_j1 0 0.02,0.10
else
  until ready ft_tm14 ft_tm14b ft_sh ft_ps15 ft_st14 ft_t2 && [ -f $R/combo_j1_cal2/SAM_POST_DONE ]; do sleep 5; done
  bash $F/code/redo_combo.sh combo_j 0 ft_g_tta,ft_p_tta,ft_v_tta,ft_ps15_tta,ft_u_tta,ft_sh_tta,ft_tm14_tta,ft_tm14b_tta,ft_k_tta,ft_t_tta,ft_t2_tta,ft_st14_tta \
    'pixelseal=ft_ps15_tta+ft_ps15_tta*ft_v_tta+ft_g_tta*ft_p_tta*ft_ps15_tta+ft_g_tta*ft_p_tta*ft_v_tta*ft_ps15_tta+ft_g_tta*ft_p_tta+ft_g_tta*ft_p_tta*ft_v_tta' \
    'trustmark=ft_tm14_tta*ft_tm14b_tta+ft_tm14_tta+ft_tm14_tta*ft_tm14b_tta*ft_u_tta+ft_tm14_tta*ft_u_tta+ft_u_tta*ft_sh_tta+ft_u_tta' \
    'stegastamp=ft_st14_tta+ft_t_tta*ft_st14_tta+ft_k_tta*ft_t_tta*ft_st14_tta+ft_t_tta*ft_st14_tta*ft_t2_tta+ft_k_tta*ft_t_tta'
  bash $F/code/run_sam_post.sh combo_j
  bash $F/code/chain_cal.sh combo_j 0 0.02,0.10
fi
