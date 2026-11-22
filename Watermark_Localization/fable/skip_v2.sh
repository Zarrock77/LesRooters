#!/bin/bash
# Let chain_v2 build the SDXL bank but not launch the v2 training (superseded by v3).
until pgrep -f 'run_bank.sh /scratch/fable/syn_sdxl' > /dev/null; do sleep 1; done
pkill -f 'code/chain_v2.sh'
echo skipped_v2
