#!/bin/bash
# usage: run_sam_shard.sh <gpu> <split> <shard> <shards> [points per side]
F=/srv/lesrooters/fable
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
O=$F/runs/sam
export PYTHONPATH=/scratch/fable/sam/segment-anything PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1
cd $O
exec $F/venv_gen/bin/python -u $F/code/sam_masks.py $K $2 /scratch/fable/sam/sam_vit_h_4b8939.pth $O/$2_$3.npz $3 $4 ${5:-32} $F/runs/frame/frame_decode.json > $O/$2_$3.log 2>&1
