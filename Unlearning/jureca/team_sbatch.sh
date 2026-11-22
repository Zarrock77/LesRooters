#!/bin/bash
# sbatch, but only if team LesRooters has no job running or queued in the cispahack reservation (organizer rule: 1 reservation job per team, all tasks).
# Usage: bash team_sbatch.sh run_node.sh cfg.txt
#        bash team_sbatch.sh --general run_node.sh cfg.txt   # general JURECA queue, no reservation: does not count toward the limit (organizers, 2026-11-21)
T=/p/scratch/training2666/gras1/LesRooters
case "$PWD" in "$T"/*) ;; *) echo "REFUSED: submit from inside $T" >&2; exit 1 ;; esac
if [ "$1" = "--general" ]; then
  shift
  script=$1; shift
  tmp=$(mktemp ./.general_XXXXXX.sh)
  # sbatch copies the script at submit time, so the temp copy can go right after.
  sed '/^#SBATCH --reservation/d' "$script" > "$tmp"
  out=$(sbatch --parsable "$tmp" "$@"); rc=$?
  rm -f "$tmp"
  [ $rc -ne 0 ] && exit $rc
  id=${out%%;*}
  # 2026-11-21: job 15903629 got Reservation=cispahack even with the line removed, so it counted as a second team node.
  if [ -n "$(squeue -h -j "$id" -o %v | grep -v '(null)')" ]; then
    scancel "$id"; echo "REFUSED: job $id was still placed in a reservation, cancelled it" >&2; exit 1
  fi
  echo "Submitted batch job $id (general queue)"
  exit 0
fi
busy=$(squeue -r --reservation=cispahack -h -o "%i|%u|%T|%j|%Z" | grep "$T")
if [ -n "$busy" ]; then
  echo "REFUSED: team already holds its node:" >&2
  echo "$busy" >&2
  echo "Chain instead: sbatch --dependency=afterany:<jobid> $*  (only if the team agrees)" >&2
  echo "Or use the general queue: bash team_sbatch.sh --general $*" >&2
  exit 1
fi
sbatch "$@"
