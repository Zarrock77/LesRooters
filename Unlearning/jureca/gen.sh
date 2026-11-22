#!/bin/bash
# ONE node: 4 shards of gen.py (one per GPU), then merged by the first consumer. Output on /p/project1 (scratch inode quota is full).
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=00:40:00
#SBATCH --output=output/node.out
G=/p/project1/training2666/delegue1/t3gen
export HF_HOME=$G/hf HUGGINGFACE_HUB_CACHE=$G/hf/hub HF_HUB_OFFLINE=1 PYTHONPATH=$G/pylib
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i ../.venv/bin/python gen.py $i 4 $G/var_$i.npz > output/slot_$i.out 2>&1 &
done
wait
# optional: then run each line of $1 (unlearn.py args), 4 at a time, like run_node.sh
[ -n "$1" ] || exit 0
i=0
while read -r line; do
  [ -z "$line" ] && continue
  CUDA_VISIBLE_DEVICES=$((i % 4)) ../.venv/bin/python unlearn.py --data . $line > output/slot_$((i + 4)).out 2>&1 &
  i=$((i + 1))
  [ $((i % 4)) -eq 0 ] && wait
done < "$1"
wait
