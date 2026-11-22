#!/bin/bash
# Queue a command on the team node:  bash add.sh "cd /p/.../unlearning/jj && ../.venv/bin/python x.py ..."
# It runs on the next free GPU (cd explicitly: lines start in queue/). Submits the node if the team has no job.
Q=/p/scratch/training2666/gras1/LesRooters/queue
[ -n "$1" ] || { sed -n 2,3p "$0"; exit 1; }
( flock 9; printf '%s\n' "$*" >> "$Q/work.txt" ) 9>"$Q/.lock"
echo "queued as line $(wc -l < "$Q/work.txt")"
busy=$(squeue -r --reservation=cispahack -h -o "%i|%u|%T|%j|%Z" | grep /p/scratch/training2666/gras1/LesRooters)
if [ -z "$busy" ]; then
  ( flock 8; [ -z "$(squeue -u gras1,delegue1,korkmaz1,cazalsdefabel1 -h)" ] && cd "$Q" && sbatch worker.sh ) 8>"$Q/.submit.lock"
else
  echo "team job: $busy"
fi
