#!/bin/bash
# Team node shared by T2 and T3 (organizer rule: 1 job per team). Runs each line appended to work.txt on a free GPU.
# re-queue costs 30+ min: this job stays up and runs each line appended to work.txt as a shell command on a free GPU.
# Special lines: "wait" = let running commands finish first, "exit" = stop. Stops by itself after 45 min without work.
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=06:00:00
#SBATCH --output=output/worker_%j.out
n=0
last=$(date +%s)
declare -a pid
busy() { for k in 0 1 2 3; do [ -n "${pid[$k]}" ] && kill -0 "${pid[$k]}" 2>/dev/null && return 0; done; return 1; }
while true; do
  if busy; then last=$(date +%s); fi
  if [ "$n" -ge "$(wc -l < work.txt)" ]; then
    [ $(($(date +%s) - last)) -gt 1800 ] && { echo "$(date +%T) idle 30 min, releasing the node"; break; }
    sleep 3; continue
  fi
  line=$(sed -n "$((n + 1))p" work.txt)
  [ "$line" = "exit" ] && break
  if [ "$line" = "wait" ]; then wait; n=$((n + 1)); continue; fi
  g=-1
  for k in 0 1 2 3; do
    if [ -z "${pid[$k]}" ] || ! kill -0 "${pid[$k]}" 2>/dev/null; then g=$k; break; fi
  done
  if [ "$g" -lt 0 ]; then sleep 2; continue; fi
  n=$((n + 1))
  last=$(date +%s)
  echo "$(date +%T) gpu $g: $line"
  CUDA_VISIBLE_DEVICES=$g bash -c "$line" &
  pid[$g]=$!
done
wait
