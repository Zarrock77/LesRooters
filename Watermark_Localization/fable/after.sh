#!/bin/bash
# usage: after.sh <marker file> <command...>
# Runs a command once a marker file exists (another run's ALL_DONE): keeps a GPU from being handed to two jobs at once.
MARK=$1; shift
until [ -f $MARK ]; do sleep 5; done
exec "$@"
