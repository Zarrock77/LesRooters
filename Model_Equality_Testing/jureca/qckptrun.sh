#!/bin/bash
# Score all ep-* answers under exact replicas of Qwen's official GPTQ-Int8 / GPTQ-Int4 / AWQ checkpoints (dequantized).
# Usage (one queue line): bash jureca/qckptrun.sh 1.5B q15b
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache HF_HUB_OFFLINE=1
E="answers/ep_v2_short.jsonl answers/ep_ident.jsonl answers/ep_v5_probe.jsonl answers/ep_v6_long.jsonl"
H=/p/project1/training2666/lesrooters/hf/hub
for q in ${QS:-GPTQ-Int8 GPTQ-Int4 AWQ}; do
  d=$(ls -d $H/models--Qwen--Qwen2.5-$1-Instruct-$q/snapshots/*)
  QCKPT=$d .venv/bin/python lm_score.py Qwen/Qwen2.5-$1-Instruct scores/epa_$2_$q.npz $E > output/qckpt_$2_$q.log 2>&1
done
