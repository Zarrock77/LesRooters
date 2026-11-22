#!/bin/bash
# Duplicate rectangle searches on the fused probabilities slow the critical jobs; drop them (redone later from cache).
pkill -f 'rect_refine.py.*ft_a_occfus'
echo killed
