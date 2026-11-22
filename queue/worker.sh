#!/bin/bash
# LesRooters team node (organizer rule: ONE job for the whole team). Runs the lines of work.txt in order, each on the
# next free GPU (4 at a time). Special lines: "wait" = let running lines finish first, "exit" = stop the node.
# Resumes where the previous node stopped (.next), logs each line to logs/line_N.out, records "N rc=.. time gpu" in
# done.txt, and releases the node after 30 min without work. Submit with add.sh (or: cd queue && sbatch worker.sh).
#SBATCH --account=training2666
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --partition=dc-gpu
#SBATCH --reservation=cispahack
#SBATCH --time=08:00:00
#SBATCH --output=/p/project1/training2666/lesrooters/queue_logs/worker_%j.out
Q=/p/scratch/training2666/gras1/LesRooters/queue
cd "$Q" || exit 1
L=/p/project1/training2666/lesrooters/queue_logs  # logs on project1: scratch hits the project-wide inode cap
mkdir -p $L
touch work.txt done.txt
next=$(cat .next 2>/dev/null || echo 0)
last=$(date +%s)
declare -a pid
busy() { for k in 0 1 2 3; do [ -n "${pid[$k]}" ] && kill -0 "${pid[$k]}" 2>/dev/null && return 0; done; return 1; }
while true; do
  if busy; then last=$(date +%s); fi
  if [ "$next" -ge "$(wc -l < work.txt)" ]; then
    [ $(($(date +%s) - last)) -gt 1800 ] && { echo "$(date +%T) idle 30 min, releasing the node"; break; }
    sleep 5; continue
  fi
  line=$(sed -n "$((next + 1))p" work.txt)
  if [ "$line" = "exit" ]; then next=$((next + 1)); echo $next > .next; break; fi
  if [ "$line" = "wait" ] || [ -z "$line" ]; then wait; next=$((next + 1)); echo $next > .next; continue; fi
  g=-1
  for k in 0 1 2 3; do
    if [ -z "${pid[$k]}" ] || ! kill -0 "${pid[$k]}" 2>/dev/null; then g=$k; break; fi
  done
  if [ "$g" -lt 0 ]; then sleep 3; continue; fi
  next=$((next + 1)); echo $next > .next; n=$next
  last=$(date +%s)
  echo "$(date +%T) line $n on gpu $g: $line"
  ( CUDA_VISIBLE_DEVICES=$g bash -c "$line" > "$L/line_$n.out" 2>&1; echo "$n rc=$? $(date +%T) gpu$g" >> done.txt ) &
  pid[$g]=$!
done
wait
