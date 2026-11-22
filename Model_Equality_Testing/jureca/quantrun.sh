#!/bin/bash
# Score all ep-* answers under bitsandbytes replicas (int8, NF4, FP4) of one base, same inputs as scores/epa_<tag>_bf16.npz.
# Usage (one queue line): bash jureca/quantrun.sh Qwen/Qwen2.5-1.5B-Instruct q15b
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache HF_HUB_OFFLINE=1
E="answers/ep_v2_short.jsonl answers/ep_ident.jsonl answers/ep_v5_probe.jsonl answers/ep_v6_long.jsonl"
for q in bnb8 nf4 fp4; do QUANT=$q .venv/bin/python lm_score.py $1 scores/epa_$2_$q.npz $E > output/quant_$2_$q.log 2>&1; done
