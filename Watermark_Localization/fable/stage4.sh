#!/bin/bash
# Stage 4: fold fine-tunes of the new pre-trainings, as soon as each one ends.
#   v8_tm  (TrustMark specialist, new seed, three banks)        -> ft_h on GPU 1
#   v9_obj (PixelSeal+TrustMark, object-shaped synthetic edits) -> ft_g on GPU 0
#   v9_gen (generalist, object-shaped synthetic edits)          -> ft_i on GPU 1
# Two streams per run: three decoder-branch fine-tunes on one GPU ran out of memory before.
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7"

finetune() {          # finetune <ft name> <init run> <gpu>
  local NAME=$1 INIT=$2 GPU=$3
  until ! pgrep -f "train_det.py.*runs/$INIT " > /dev/null; do sleep 5; done
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- $ARGS &
  bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- $ARGS &
  wait
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}

finetune ft_h v8_tm 1 &
finetune ft_g v9_obj 0 &
finetune ft_i v9_gen 1 &
wait
