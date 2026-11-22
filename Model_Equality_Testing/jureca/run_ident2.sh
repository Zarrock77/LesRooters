#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=output/%j.out
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache HF_HUB_OFFLINE=1
M=HuggingFaceTB/SmolLM2-360M-Instruct; R="answers/ref_v2_short.jsonl answers/ref_v1.jsonl"
( for g in 0.25 0.5; do for d in cuda cpu; do echo "== gamma $g $d"; CUDA_VISIBLE_DEVICES=3 .venv/bin/python wm_detect.py $M $d $g lefthash $R; done; done > output/wm_ref.log 2>&1 ) &
bash jureca/run_score.sh jureca/tasks_ident2.txt
wait
