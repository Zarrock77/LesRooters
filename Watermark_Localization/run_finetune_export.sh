#!/bin/bash
#SBATCH --job-name=rooters-wm-fine-export
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
set -euo pipefail
TASK_ROOT=/p/scratch/training2666/gras1/LesRooters/Hackathon_Watermark_Localisation_Task
export PYTHONUNBUFFERED=1
export HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
set +e
srun "$TASK_ROOT/.venv/bin/python" "$SLURM_SUBMIT_DIR/assess_finetune.py" --kit-root "$TASK_ROOT"
ASSESSMENT=$?
set -e
if [ "$ASSESSMENT" -eq 3 ]; then
  echo 'NO_ROBUST_GAIN: preserve the previous ensemble'; exit 0
fi
if [ "$ASSESSMENT" -ne 0 ]; then exit "$ASSESSMENT"; fi
srun "$TASK_ROOT/.venv/bin/python" "$SLURM_SUBMIT_DIR/export_cv_ensemble.py" \
  --kit-root "$TASK_ROOT" --guidance-root "$TASK_ROOT/output/localization_v2" \
  --comparison-root "$TASK_ROOT/output/cv_finetuned_v1" \
  --output-root "$TASK_ROOT/output/submission_cv_finetuned_v1" --version cv_finetuned_v1 \
  --allow-numerical-replay
