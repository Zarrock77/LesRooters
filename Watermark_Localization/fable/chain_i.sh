#!/bin/bash
# combo_i: kit-recipe fine-tunes for PixelSeal (ft_v) and a second TrustMark run (ft_y) added to combo_h's candidates.
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
until [ -f $R/ft_v/ALL_DONE ] && [ -f $R/ft_y/ALL_DONE ] && [ -f $R/combo_h/SAM_POST_DONE ]; do sleep 5; done
tta() {
  local GPU=$1; shift
  for run in "$@"; do
    [ -f $R/${run}_tta/REPREDICTED ] && continue
    mkdir -p $R/${run}_tta
    CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/$run --tta-out $R/${run}_tta --frame-json $R/frame/frame_decode.json > $R/${run}_tta/repredict.log 2>&1
  done
}
tta 0 ft_v &
tta 1 ft_y &
wait
bash $F/code/redo_combo.sh combo_i 0 ft_g_tta,ft_p_tta,ft_v_tta,ft_h2_tta,ft_l_tta,ft_u_tta,ft_y_tta,ft_k_tta,ft_t_tta,ft_s2_tta \
  'pixelseal=ft_v_tta+ft_g_tta*ft_v_tta+ft_g_tta*ft_p_tta*ft_v_tta+ft_g_tta*ft_p_tta' \
  'trustmark=ft_u_tta*ft_y_tta+ft_u_tta*ft_y_tta*ft_v_tta+ft_u_tta*ft_y_tta*ft_h2_tta+ft_u_tta*ft_h2_tta*ft_l_tta+ft_u_tta' \
  'stegastamp=ft_k_tta*ft_t_tta+ft_k_tta*ft_t_tta*ft_s2_tta+ft_t_tta+ft_k_tta'
bash $F/code/run_sam_post.sh combo_i
