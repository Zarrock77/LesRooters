#!/bin/bash
# combo_h: the fine-tunes of the kit-recipe pre-trainings added as candidates (ft_u TrustMark, ft_t StegaStamp) plus
# the fit+tune StegaStamp fine-tune ft_s2. Shift-averaged siblings first, one per GPU at a time.
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
until [ -f $R/ft_t/ALL_DONE ] && [ -f $R/ft_u/ALL_DONE ]; do sleep 5; done
tta() {
  local GPU=$1; shift
  for run in "$@"; do
    [ -f $R/${run}_tta/REPREDICTED ] && continue
    mkdir -p $R/${run}_tta
    CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/$run --tta-out $R/${run}_tta --frame-json $R/frame/frame_decode.json > $R/${run}_tta/repredict.log 2>&1
  done
}
tta 0 ft_u ft_s2 &
tta 0 ft_t &
wait
bash $F/code/redo_combo.sh combo_h 0 ft_g_tta,ft_p_tta,ft_h2_tta,ft_l_tta,ft_u_tta,ft_k_tta,ft_t_tta,ft_s2_tta \
  'pixelseal=ft_g_tta*ft_p_tta' \
  'trustmark=ft_u_tta*ft_h2_tta*ft_l_tta+ft_u_tta*ft_h2_tta+ft_u_tta+ft_h2_tta*ft_l_tta' \
  'stegastamp=ft_k_tta*ft_t_tta+ft_k_tta*ft_t_tta*ft_s2_tta+ft_t_tta+ft_k_tta'
bash $F/code/run_sam_post.sh combo_h
