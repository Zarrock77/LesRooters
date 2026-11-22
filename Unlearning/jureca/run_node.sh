#!/bin/bash
# ONE node only (organizer rule): runs each line of $1 (unlearn.py args) on the 4 GPUs, $2 at a time (default 4 = one per GPU).
# Fixed log names (output/node.out, output/slot_<i>.out): the project inode quota is often full, existing files can still be overwritten.
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=01:30:00
#SBATCH --output=output/node.out
par=${2:-4}
i=0
while read -r line; do
  [ -z "$line" ] && continue
  CUDA_VISIBLE_DEVICES=$((i % 4)) ../.venv/bin/python unlearn.py --data . $line > output/slot_$i.out 2>&1 &
  i=$((i + 1))
  [ $((i % par)) -eq 0 ] && wait
done < "$1"
wait
