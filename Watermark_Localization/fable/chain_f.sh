#!/bin/bash
# combo_f: the StegaStamp single-method fine-tunes (ft_k, ft_s) and the PixelSeal one (ft_p) added, every component
# predicted with shift-averaging (out-of-fold gain: PixelSeal +0.0085, TrustMark +0.0029, StegaStamp +0.0035), then
# segment selection. The fit+tune runs of stage 9 start only afterwards: the post-processing needs GPU memory.
F=/srv/lesrooters/fable
R=$F/runs
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
tta() {   # tta <gpu> <run> ...   one after the other on that GPU
  local GPU=$1; shift
  for run in "$@"; do
    [ -f $R/${run}_tta/REPREDICTED ] && continue
    mkdir -p $R/${run}_tta
    CUDA_VISIBLE_DEVICES=$GPU $K/.venv/bin/python -u $F/code/repredict.py --kit $K --run $R/$run --tta-out $R/${run}_tta --frame-json $R/frame/frame_decode.json > $R/${run}_tta/repredict.log 2>&1
  done
}
# finished runs first, while the last single-method folds are still training
tta 1 ft_g ft_k &
tta 1 ft_h &
until [ -f $R/ft_p/ALL_DONE ] && [ -f $R/ft_s/ALL_DONE ]; do sleep 5; done
tta 1 ft_p &
tta 0 ft_s &
wait
bash $F/code/redo_combo.sh combo_f 1 ft_g_tta,ft_p_tta,ft_h_tta,ft_k_tta,ft_s_tta \
  'pixelseal=ft_g_tta+ft_g_tta*ft_p_tta+ft_p_tta' \
  'trustmark=ft_g_tta*ft_h_tta+ft_h_tta' \
  'stegastamp=ft_k_tta*ft_s_tta+ft_s_tta+ft_k_tta'
bash $F/code/run_sam_post.sh combo_f
setsid nohup bash $F/code/stage9.sh > $R/stage9.log 2>&1 < /dev/null &
