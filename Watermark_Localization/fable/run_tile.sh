#!/bin/bash
# usage: run_tile.sh <gpu>  -- tile decoder (fixed 1000 steps of the 2400-step schedule), then presence maps for all PixelSeal pictures
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
R=/srv/lesrooters/fable/runs
C=/srv/lesrooters/fable/code
export CUDA_VISIBLE_DEVICES=${1:-0} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
mkdir -p $R/tiledec3_ps
.venv/bin/python -u $C/tile_decoder.py $K /scratch/fable/syn/clean.npy $R/frame/frame_decode.json $R/tiledec3_ps 2400 0.5 1000 > $R/tiledec3_ps/log.txt 2>&1
.venv/bin/python -u $C/tile_maps.py $K $R/tiledec3_ps/tile_decoder.pt $R/tilemaps 32 > $R/tiledec3_ps/maps.log 2>&1
echo done > $R/tilemaps/DONE
