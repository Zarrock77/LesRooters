#!/bin/bash
#SBATCH --job-name=rooters-wm-v1
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# Before the first run, from this code directory on the login node:
# TASK_ROOT=/p/scratch/training2666/gras1/LesRooters/Hackathon_Watermark_Localisation_Task
# "$TASK_ROOT/.venv/bin/python" prepare_trustmark.py
# mkdir -p output
# sbatch run_baseline.sh
# Results and feature caches are written to $TASK_ROOT/output/baseline_v1.
# Resubmitting reuses cached features and checks that the frozen split matches.

set -euo pipefail
TASK_ROOT=${TASK_ROOT:-/p/scratch/training2666/gras1/LesRooters/Hackathon_Watermark_Localisation_Task}
CODE_ROOT=${CODE_ROOT:-$SLURM_SUBMIT_DIR}
export PYTHONUNBUFFERED=1
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
srun "$TASK_ROOT/.venv/bin/python" "$CODE_ROOT/baseline.py" \
  --kit-root "$TASK_ROOT" \
  --data-root "$TASK_ROOT/release" \
  --output-root "$TASK_ROOT/output/baseline_v1" "$@"
