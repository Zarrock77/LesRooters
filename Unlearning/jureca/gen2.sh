#!/bin/bash
# More variations per image (4 more at strength 0.5/0.6, two seeds): a per-image estimate instead of one sample.
# Run it through the team worker (work.txt). Do NOT submit it as a second job: the cispahack reservation is MAGNETIC,
# so a job without --reservation still takes a reservation node (happened on 2026-11-21, cancelled).
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:30:00
#SBATCH --output=output/slot_10.out
G=/p/project1/training2666/delegue1/t3gen
export HF_HOME=$G/hf HUGGINGFACE_HUB_CACHE=$G/hf/hub HF_HUB_OFFLINE=1 PYTHONPATH=$G/pylib
for i in 0 1 2 3; do
  (CUDA_VISIBLE_DEVICES=$i ../.venv/bin/python gen.py $i 4 $G/w1_$i.npz 0.5,0.6 1 && CUDA_VISIBLE_DEVICES=$i ../.venv/bin/python gen.py $i 4 $G/w2_$i.npz 0.5,0.6 2) > $G/gen2_$i.log 2>&1 &
done
wait
