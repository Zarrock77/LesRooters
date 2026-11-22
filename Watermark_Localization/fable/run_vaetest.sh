#!/bin/bash
# Does a VAE round trip of synthetic compositions hurt the detectors the way the official images do?
F=/srv/lesrooters/fable
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
D=/scratch/fable/synval
GPU=${1:-1}
export HF_HOME=/scratch/fable/hf PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$GPU
cd $K
$K/.venv/bin/python $F/code/dump_synval.py /scratch/fable/syn $D 384
$F/venv_gen/bin/python $F/code/vae_apply.py $D/images.npy $D/images_fluxvae.npy
cp $F/runs/v4_wm/ckpt_last.pt $D/v4.pt
for C in $F/runs/diag/ckpt.pt $D/v4.pt; do
  $K/.venv/bin/python $F/code/eval_npz.py $K $D/labels.npz $C $D/images.npy $D/images_fluxvae.npy
done
echo VAETEST_DONE
