#!/bin/bash
# Team node and queue at a glance.
Q=/p/scratch/training2666/gras1/LesRooters/queue
echo "== team jobs";   squeue -u gras1,delegue1,korkmaz1,cazalsdefabel1 -h -o "%i %u %T %j %M/%l %R"
n=$(cat "$Q/.next" 2>/dev/null || echo 0); t=$(wc -l < "$Q/work.txt")
echo "== queue: $n of $t lines started"
echo "== waiting";     sed -n "$((n + 1)),\$p" "$Q/work.txt" | cut -c1-150 | nl -v $((n + 1))
echo "== last done";   tail -5 "$Q/done.txt"
