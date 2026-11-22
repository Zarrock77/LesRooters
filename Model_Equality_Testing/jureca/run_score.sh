#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --output=output/%j.out
# One node, 4 GPUs: lines of TASKS (MODEL OUT.npz in.jsonl ...) are dealt round-robin to the GPUs, run in sequence per GPU.
# Submit from model_equality/jj with:  bash team_sbatch.sh jureca/run_score.sh tasks.txt
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache HF_HUB_OFFLINE=1
TASKS=$1
GPUS=(${GPUS:-0 1 2 3})  # GPUS="1 2 3" to run as an --overlap step beside another job
for i in "${!GPUS[@]}"; do
  g=${GPUS[$i]}
  ( awk -v i=$i -v n=${#GPUS[@]} 'NF && (NR-1)%n==i' "$TASKS" | while read -r line; do
      out=$(echo $line | cut -d" " -f2); case "$out" in *fp8*) export FP8=1;; *) unset FP8;; esac
      CUDA_VISIBLE_DEVICES=$g .venv/bin/python lm_score.py $line > "output/$(basename $out .npz).log" 2>&1
    done ) &
done
wait
