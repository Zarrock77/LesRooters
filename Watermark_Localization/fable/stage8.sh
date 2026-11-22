#!/bin/bash
# Stage 8: single-method fold fine-tunes. The StegaStamp-only fine-tune ft_k scored 0.905 out-of-fold against 0.879
# for the three-method fine-tune ft_d, and the TrustMark-only ft_h beat the two-method ones: do the same everywhere.
#   GPU 1: ft_p (PixelSeal only, from v9_obj), then ft_l (TrustMark only, from v10_tm when it ends)
#   GPU 0: ft_q (TrustMark only, from v9_obj), ft_s (StegaStamp only, from v9_gen), then ft_r (PixelSeal only, from v11_obj)
F=/srv/lesrooters/fable
R=$F/runs
ARGS="--cutmix 0.5 --objects 0.7"

ft() {                # ft <name> <init run> <gpu> <streams: 1|2> [finetune_cv.py arguments]
  local NAME=$1 INIT=$2 GPU=$3 STREAMS=$4; shift 4
  until ! pgrep -f "train_det.py.*runs/$INIT " > /dev/null; do sleep 5; done
  mkdir -p $R/$NAME
  [ -f $R/$NAME/init.pt ] || cp $R/$INIT/ckpt_last.pt $R/$NAME/init.pt
  if [ $STREAMS -eq 2 ]; then
    bash $F/code/run_ft.sh $NAME $INIT $GPU 0 2 4 -- $ARGS "$@" &
    bash $F/code/run_ft.sh $NAME $INIT $GPU 1 3 -- $ARGS "$@" &
    wait
  else
    bash $F/code/run_ft.sh $NAME $INIT $GPU 0 1 2 3 4 -- $ARGS "$@"
  fi
  [ $(ls $R/$NAME/fold_*/best.pt 2>/dev/null | wc -l) -eq 5 ] && echo done > $R/$NAME/ALL_DONE
}

ft ft_p v9_obj 1 2 --methods pixelseal &
ft ft_l v10_tm 1 1 &
ft ft_q v9_obj 0 1 --methods trustmark &
ft ft_s v9_gen 0 2 --methods stegastamp &
ft ft_r v11_obj 0 1 --methods pixelseal &
wait
