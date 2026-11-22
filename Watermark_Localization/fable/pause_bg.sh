#!/bin/bash
# Pause the lower-priority background jobs (StegaStamp continuation, PixelSeal test search) until the stage-2
# fold fine-tunes are finished, then resume them. Explicit PIDs only.
R=/srv/lesrooters/fable/runs
P1=$(pgrep -f 'train_det.py.*runs/v6_steg[a]' | tr '\n' ' ')
P2=$(pgrep -f 'rect_refine.py.*--methods pixelseal --global-sca[n]' | tr '\n' ' ')
echo "pausing train: $P1 scan: $P2"
[ -n "$P1$P2" ] && kill -STOP $P1 $P2
until [ -f $R/ft_d/ALL_DONE ] && [ -f $R/ft_e/ALL_DONE ]; do sleep 3; done
kill -CONT $P1 $P2
echo resumed
