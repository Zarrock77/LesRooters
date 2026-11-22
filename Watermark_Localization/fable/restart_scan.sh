#!/bin/bash
pkill -f 'rect_refine.py.*global-scan'
sleep 2
bash /srv/lesrooters/fable/code/run_scan.sh ft_a 1 train trustmark
