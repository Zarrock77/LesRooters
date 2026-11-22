#!/bin/bash
# T2 simulator for one base: generate sim answers, then score them in bf16 and with the fp8 replica.
# Usage (one work.txt line, GPU set by the worker): bash jureca/simrun.sh Qwen/Qwen2.5-1.5B-Instruct q15b
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache HF_HUB_OFFLINE=1
.venv/bin/python sim.py $1 answers/sim_$2.jsonl > output/sim_$2.log 2>&1 || exit 1
.venv/bin/python lm_score.py $1 scores/sim_$2_bf16.npz answers/sim_$2.jsonl > output/simscore_$2_bf16.log 2>&1
FP8=1 .venv/bin/python lm_score.py $1 scores/sim_$2_fp8.npz answers/sim_$2.jsonl > output/simscore_$2_fp8.log 2>&1
