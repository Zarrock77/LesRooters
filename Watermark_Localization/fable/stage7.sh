#!/bin/bash
# Stage 7: second seeds for averaging, and SAM proposals for full-frame test pictures.
#   v11_obj (PixelSeal+TrustMark, object-aware, new seed, from v4_wm) -> ft_n   GPU 0
#   v10_tm  (TrustMark specialist continued with object-aware data)   -> ft_l   GPU 1
F=/srv/lesrooters/fable
R=$F/runs
ALL=/scratch/fable/syn,/scratch/fable/syn_vae,/scratch/fable/syn_sdxl
ARGS="--cutmix 0.5 --objects 0.7"

finetune() {          # finetune <ft name> <init run> <gpu>
  local NAME=$1 INIT=$2 GPU=$3
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- $ARGS &
  bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- $ARGS &
  wait
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}

for s in 0 1 2; do bash $F/code/run_sam_full.sh 0 test $s 3 & done
(bash $F/code/run_train.sh v11_obj 0 --flag --wm-branch --methods pixelseal,trustmark --objects 0.7 --seed 21 --iters 3000 --eval-every 1000 --lr 2e-4 --init $R/v4_wm/ckpt_last.pt --layers $ALL
 finetune ft_n v11_obj 0) &
(bash $F/code/run_train.sh v10_tm 1 --flag --wm-branch --methods trustmark --objects 0.7 --seed 13 --iters 3000 --eval-every 1000 --lr 1.5e-4 --init $R/v8_tm/ckpt_last.pt --layers $ALL
 until [ -f $R/ft_k/ALL_DONE ]; do sleep 5; done
 finetune ft_l v10_tm 1) &
wait
