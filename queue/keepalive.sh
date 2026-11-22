#!/bin/bash
# Keeps the team node from releasing itself: a work.txt line "bash keepalive.sh <its own line number>" that sleeps while it is the
# last line (worker.sh sees a live process, so its 30-min idle release never fires). When a real line is appended after it, it exits
# (the worker reuses the slot at once) and re-appends itself at the tail, so it only holds a GPU slot while nothing else is waiting.
# shortcut: the 8 h Slurm limit and an explicit "exit" line still stop the node on purpose.
Q=${Q:-/p/scratch/training2666/gras1/LesRooters/queue}
N=$1
[ -n "$N" ] || { sed -n 2,3p "$0"; exit 1; }
while [ "$(wc -l < "$Q/work.txt")" -le "$N" ]; do sleep "${POLL:-20}"; done
( flock 9; printf 'bash %s/keepalive.sh %s\n' "$Q" "$(( $(wc -l < "$Q/work.txt") + 1 ))" >> "$Q/work.txt" ) 9>"$Q/.lock"
