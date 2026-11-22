#!/bin/bash
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --time=00:45:00
#SBATCH --output=output/%j.out
# T2: temperature-aware learned-LoRA probes on ep-* (train on the most deviant endpoint of each family) + rescoring.
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache HF_HUB_OFFLINE=1
L=answers/ep_v6_long.jsonl
probe() { CUDA_VISIBLE_DEVICES=$1 .venv/bin/python lora_probe.py $2 $3 jureca/temps_$4.json scores/probeT_$3.json $L > output/probeT_$3.log 2>&1; }
( probe 0 Qwen/Qwen2.5-3B-Instruct ep-14 q3b; probe 0 Qwen/Qwen2.5-3B-Instruct ep-24 q3b ) &
( probe 1 Qwen/Qwen2.5-1.5B-Instruct ep-07 q15b; probe 1 Qwen/Qwen2.5-1.5B-Instruct ep-28 q15b ) &
( probe 2 unsloth/Llama-3.2-1B-Instruct ep-27 l1b; probe 2 unsloth/Llama-3.2-1B-Instruct ep-05 l1b ) &
GPUS="3" bash jureca/run_score.sh jureca/tasks_ep_all.txt &
wait
