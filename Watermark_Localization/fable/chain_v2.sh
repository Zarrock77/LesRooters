#!/bin/bash
# AI-looking source banks -> layers -> v2 training, without manual steps in between.
set -uo pipefail
F=/srv/lesrooters/fable
L=$F/runs/chain_v2
G=$F/venv_gen/bin/python
mkdir -p $L
export HF_HOME=/scratch/fable/hf PYTHONDONTWRITEBYTECODE=1
until grep -q VENV_OK $F/runs/venv_gen.log 2>/dev/null; do sleep 2; done
until grep -q ANN_OK $F/runs/ann.log 2>/dev/null; do sleep 2; done
CUDA_VISIBLE_DEVICES=0 $G -u $F/code/gen_ai_sources.py --mode vae --coco /scratch/fable/syn/clean.npy --out /scratch/fable/syn_vae --count 4002 > $L/vae.log 2>&1
(bash $F/code/run_bank.sh /scratch/fable/syn_vae 4 > $L/bank_vae.log 2>&1) &
CUDA_VISIBLE_DEVICES=0 $G -u $F/code/gen_ai_sources.py --mode sdxl --captions /scratch/fable/coco/annotations/captions_val2017.json --out /scratch/fable/syn_sdxl --count 4000 > $L/sdxl.log 2>&1
bash $F/code/run_bank.sh /scratch/fable/syn_sdxl 4 > $L/bank_sdxl.log 2>&1
wait
if [ -f /scratch/fable/syn_vae/BANK_DONE ] && [ -f /scratch/fable/syn_sdxl/BANK_DONE ]; then
  bash $F/code/run_train.sh v2_flag 0 --flag --iters 14000 --eval-every 1000 --layers /scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
fi
