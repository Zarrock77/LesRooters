#!/bin/bash
# Re-render fresh embeddings (generator environment), then print signatures for the three methods (kit environment).
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
R=/srv/lesrooters/fable/runs
C=/srv/lesrooters/fable/code
cd $K
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=${1:-0}
[ -f $R/stability/rerender_stegastamp.npz ] || HF_HOME=/scratch/fable/hf /srv/lesrooters/fable/venv_gen/bin/python -u $C/rerender.py $R/stability > $R/stability/rerender.log 2>&1
for m in trustmark pixelseal stegastamp; do
  .venv/bin/python -u $C/sig_probe.py $K $R/stability $R/frame/frame_decode.json $m > $R/stability/sig_$m.log 2>&1
done
echo SIG_ALL_DONE > $R/stability/SIG_ALL_DONE
