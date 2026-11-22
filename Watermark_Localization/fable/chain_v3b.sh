#!/bin/bash
# Regenerated ("edited-looking") versions of every source, then v3 training warm-started from v1.
set -uo pipefail
F=/srv/lesrooters/fable
L=$F/runs/chain_v3
G=$F/venv_gen/bin/python
GPU=${1:-1}
mkdir -p $L $F/runs/v3_flag
export HF_HOME=/scratch/fable/hf PYTHONDONTWRITEBYTECODE=1
until [ -f /scratch/fable/syn_sdxl/SOURCES_DONE ]; do sleep 2; done
for B in syn_sdxl syn syn_vae; do
  CUDA_VISIBLE_DEVICES=$GPU $G -u $F/code/gen_ai_sources.py --mode regen --src /scratch/fable/$B/clean.npy --out /scratch/fable/$B --count 5000 > $L/regen_$B.log 2>&1
done
until [ -f /scratch/fable/syn_vae/BANK_DONE ] && [ -f /scratch/fable/syn_sdxl/BANK_DONE ]; do sleep 2; done
cp $F/runs/v1_flag/ckpt_last.pt $F/runs/v3_flag/init.pt
bash $F/code/run_train.sh v3_flag $GPU --flag --iters 10000 --eval-every 1000 --init $F/runs/v3_flag/init.pt \
  --layers /scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
