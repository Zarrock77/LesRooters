#!/bin/bash
# Build the synthetic layer bank on the GPU machine: clean sources, then the three embedders in parallel.
set -euo pipefail
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
D=/scratch/fable/syn
L=/srv/lesrooters/fable/runs/data
PY=$K/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
mkdir -p $D $L
cd $K
if [ ! -f $D/clean.npy ]; then
  $PY -u $C/prepare_sources.py $D > $L/sources.log 2>&1
fi
$PY -u $C/gen_layers.py --kit $K --data $D --method pixelseal --device cuda:0 > $L/gen_pixelseal.log 2>&1 &
$PY -u $C/gen_layers.py --kit $K --data $D --method trustmark --device cuda:0 > $L/gen_trustmark.log 2>&1 &
$PY -u $C/gen_layers.py --kit $K --data $D --method stegastamp --device cuda:1 > $L/gen_stegastamp.log 2>&1 &
wait
echo DATA_DONE > $L/DONE
