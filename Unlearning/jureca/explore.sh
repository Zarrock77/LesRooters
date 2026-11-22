#!/bin/bash
#SBATCH --account=training2666
#SBATCH --gres=gpu:1
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:30:00
#SBATCH --output=output/%j.out
srun ../.venv/bin/python explore.py --data .
