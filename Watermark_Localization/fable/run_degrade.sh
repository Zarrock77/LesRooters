#!/bin/bash
# VAE round trips (generator env) then the degradation probe on those round trips (kit env).
F=/srv/lesrooters/fable
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
D=$F/runs/stability
GPU=${1:-1}
export HF_HOME=/scratch/fable/hf PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
rm -f $D/extra_*.npz
CUDA_VISIBLE_DEVICES=$GPU $F/venv_gen/bin/python -u $F/code/vae_roundtrip.py $D > $D/roundtrip.log 2>&1
cd $K
$K/.venv/bin/python -u $F/code/degrade_probe.py $K $D cuda:$GPU extra > $D/degrade_flux.log 2>&1
