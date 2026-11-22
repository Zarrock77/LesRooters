#!/bin/bash
# usage: run_sam_full.sh <gpu> <split> <shard> <shards>   -- SAM proposals for pictures WITH a full-frame watermark
F=/srv/lesrooters/fable
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
O=$F/runs/sam
export PYTHONPATH=/scratch/fable/sam/segment-anything PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=$1
cd $O
exec $F/venv_gen/bin/python -u $F/code/sam_masks.py $K $2 /scratch/fable/sam/sam_vit_h_4b8939.pth $O/full_$2_$3.npz $3 $4 24 $F/runs/frame/frame_decode.json full > $O/full_$2_$3.log 2>&1
