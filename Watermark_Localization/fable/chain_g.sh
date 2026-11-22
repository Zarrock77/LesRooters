#!/bin/bash
# combo_g: TrustMark candidates extended with the fit+tune specialist (ft_h2, 0.730 out-of-fold alone) and the
# object-aware specialist (ft_l, 0.726); PixelSeal and StegaStamp as in combo_f. Shift-averaged siblings first.
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
GPU=${1:-0}
for run in ft_h2 ft_l; do
  [ -f $R/${run}_tta/REPREDICTED ] && continue
  mkdir -p $R/${run}_tta
  CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/$run --tta-out $R/${run}_tta --frame-json $R/frame/frame_decode.json > $R/${run}_tta/repredict.log 2>&1 &
done
wait
bash $F/code/redo_combo.sh combo_g $GPU ft_g_tta,ft_p_tta,ft_h_tta,ft_h2_tta,ft_l_tta,ft_k_tta \
  'pixelseal=ft_g_tta*ft_p_tta' \
  'trustmark=ft_h2_tta*ft_l_tta+ft_g_tta*ft_h2_tta*ft_l_tta+ft_h_tta*ft_h2_tta*ft_l_tta+ft_h2_tta+ft_g_tta*ft_h_tta' \
  'stegastamp=ft_k_tta'
bash $F/code/run_sam_post.sh combo_g
