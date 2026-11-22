#!/bin/bash
# Whole-image decoder rectangle searches, one after the other on one GPU. Results are reused by every later post-processing.
GPU=${1:-1}
C=/srv/lesrooters/fable/code
bash $C/run_scan.sh ft_a $GPU test trustmark
bash $C/run_scan.sh ft_a $GPU train pixelseal
bash $C/run_scan.sh ft_a $GPU test pixelseal
echo SCAN_CHAIN_DONE > /srv/lesrooters/fable/runs/scan/CHAIN_DONE
