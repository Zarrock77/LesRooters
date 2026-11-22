#!/bin/bash
# Install Segment Anything in the generator environment, fetch the ViT-H checkpoint, segment train and test images.
set -uo pipefail
F=/srv/lesrooters/fable
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
O=$F/runs/sam
G=$F/venv_gen/bin/python
mkdir -p $O /scratch/fable/sam
cd $O
[ -d /scratch/fable/sam/segment-anything ] || git clone -q --depth 1 https://github.com/facebookresearch/segment-anything /scratch/fable/sam/segment-anything
export PYTHONPATH=/scratch/fable/sam/segment-anything
CK=/scratch/fable/sam/sam_vit_h_4b8939.pth
[ -f $CK ] || (curl -fL --retry 3 -o $CK.part https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth && mv $CK.part $CK)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
CUDA_VISIBLE_DEVICES=0 $G -u $F/code/sam_masks.py $K train $CK $O/train.npz > $O/train.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 $G -u $F/code/sam_masks.py $K test $CK $O/test_0.npz 0 2 > $O/test_0.log 2>&1 &
CUDA_VISIBLE_DEVICES=0 $G -u $F/code/sam_masks.py $K test $CK $O/test_1.npz 1 2 > $O/test_1.log 2>&1 &
wait
echo SAM_ALL_DONE > $O/SAM_ALL_DONE
