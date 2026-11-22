#!/bin/bash
#SBATCH --job-name=rooters-wm-export2
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:04:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
set -euo pipefail
TASK_ROOT=/p/scratch/training2666/gras1/LesRooters/Hackathon_Watermark_Localisation_Task
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
VERSION=${1:-localization_v2}
ARCHITECTURE=${2:-residual}
srun "$TASK_ROOT/.venv/bin/python" "$SLURM_SUBMIT_DIR/export_segmenter.py" \
  --kit-root "$TASK_ROOT" --model-root "$TASK_ROOT/output/$VERSION" \
  --guidance-root "$TASK_ROOT/output/localization_v2" \
  --output-root "$TASK_ROOT/output/submission_$VERSION" --version "$VERSION" --architecture "$ARCHITECTURE"
