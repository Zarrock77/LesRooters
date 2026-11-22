#!/bin/bash
#SBATCH --job-name=rooters-wm-adobe-export
#SBATCH --account=training2666
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:07:00
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err
set -euo pipefail
TASK_ROOT=/p/scratch/training2666/gras1/LesRooters/Hackathon_Watermark_Localisation_Task
export PYTHONUNBUFFERED=1 HF_HOME=/p/scratch/training2666/gras1/LesRooters/.cache
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
srun "$TASK_ROOT/.venv/bin/python" "$SLURM_SUBMIT_DIR/finish_adobe.py" --kit-root "$TASK_ROOT"
