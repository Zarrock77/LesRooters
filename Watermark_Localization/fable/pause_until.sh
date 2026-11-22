#!/bin/bash
# usage: pause_until.sh <pgrep pattern> <file to wait for>
# Pause matching processes (SIGSTOP) until the file exists, then resume them.
PIDS=$(pgrep -f "$1" | tr '\n' ' ')
[ -n "$PIDS" ] || { echo "no process for $1"; exit 0; }
kill -STOP $PIDS
echo "paused $PIDS"
until [ -f "$2" ]; do sleep 2; done
kill -CONT $PIDS
echo "resumed $PIDS"
