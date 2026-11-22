#!/bin/bash
# usage: chain_sam_post.sh <run>  -- when the run's rectangles and all SAM test shards are ready, assemble the segment-selected candidate.
R=/srv/lesrooters/fable/runs
until [ -f $R/$1/POST_DONE ] && grep -q SAM_DONE $R/sam/test_0.log && grep -q SAM_DONE $R/sam/test_1.log && grep -q SAM_DONE $R/sam/test_2.log; do sleep 4; done
bash /srv/lesrooters/fable/code/run_sam_post.sh $1
