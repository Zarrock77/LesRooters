#!/bin/bash
# usage: run_post.sh <folds run name> <gpu index>
# OOF merge -> decoder rectangle snapping (train OOF + test) -> fold replay -> honest OOF reports -> submission artifacts.
set -euo pipefail
NAME=$1; GPU=${2:-0}
K=/srv/lesrooters/Hackathon_Watermark_Localisation_Task
C=/srv/lesrooters/fable/code
B=/srv/lesrooters/fable/runs/$NAME
PY=$K/.venv/bin/python
FRAME=/srv/lesrooters/fable/runs/frame/frame_decode.json
REF=/srv/lesrooters/fable/reference/image_scores.json
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
cd $K
$PY $C/merge_folds.py $B
# Let rectangle searches already running finish first, so their confident results are reused instead of recomputed.
for i in $(seq 1 400); do pgrep -f "rect_refine.py.*--out /srv/lesrooters/fable/runs/ft_" > /dev/null || break; sleep 3; done
# Rectangles already found with confidence in earlier rounds are reused (they depend on the image, not on the model).
PRIOR_TRAIN=$(ls /srv/lesrooters/fable/runs/*/rect_train.json /srv/lesrooters/fable/runs/scan/rect_train_*.json 2>/dev/null | grep -v "/$NAME/" | tr '\n' ' ' || true)
PRIOR_TEST=$(ls /srv/lesrooters/fable/runs/*/rect_test.json /srv/lesrooters/fable/runs/scan/rect_test_*.json 2>/dev/null | grep -v "/$NAME/" | tr '\n' ' ' || true)
$PY -u $C/rect_refine.py --kit $K --split train --probs $B/oof_probs.npz --frame-json $FRAME --out $B/rect_train.json --device cuda:$GPU --prior $PRIOR_TRAIN > $B/rect_train.log 2>&1 &
$PY -u $C/rect_refine.py --kit $K --split test --probs $B/test_mean_probs.npz --frame-json $FRAME --out $B/rect_test.json --device cuda:$((1 - GPU)) --prior $PRIOR_TEST > $B/rect_test.log 2>&1 &
REPLAY=""
BASE=${3:-$NAME}                       # fold models behind these probabilities (for the replay check)
BB=/srv/lesrooters/fable/runs/$BASE
if [ ! -f $BB/replay.json ] && [ -f $BB/fold_0/best.pt ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -u $C/replay_check.py --kit $K --folds $BB --frame-json $FRAME > $BB/replay.log 2>&1
fi
[ -f $BB/replay.json ] && REPLAY="--replay $BB/replay.json"
if [ -n "${COMPONENTS:-}" ]; then      # per-method combination: replay each method with the fold models it comes from
  CUDA_VISIBLE_DEVICES=$GPU $PY -u $C/replay_check.py --kit $K --folds $B --frame-json $FRAME --components $COMPONENTS > $B/replay.log 2>&1
  REPLAY="--replay $B/replay.json"
fi
$PY $C/assemble.py --folds $B --train-root $K/release/train --reference-scores $REF $REPLAY --candidate ${NAME}_plain \
  --report $B/report_plain.json --submission $B/submission_plain > $B/assemble_plain.log 2>&1
echo PLAIN_DONE > $B/PLAIN_DONE
wait
$PY $C/assemble.py --folds $B --train-root $K/release/train --reference-scores $REF --rect-train $B/rect_train.json \
  --rect-test $B/rect_test.json $REPLAY --candidate ${NAME}_rect --report $B/report_rect.json --submission $B/submission_rect > $B/assemble_rect.log 2>&1
echo POST_DONE > $B/POST_DONE
