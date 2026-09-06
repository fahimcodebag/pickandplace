#!/bin/bash
# Distil the grasp actor to several widths, from two initialisations.
# The pruned-vs-random comparison is the point: does structured pruning give a
# better starting point than random init, or is it just a smaller net trained
# twice?
PY=/home/fahim/Thesis_fahim/venv/bin/python
CK=checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best
for cfg in "48 24" "32 16" "24 12" "16 8"; do
  set -- $cfg
  for init in pruned random; do
    $PY distill_actor.py --ckpt $CK --keep1 $1 --keep2 $2 --init $init \
        --states 60000 --epochs 60 --seed 0 \
        --out Results/distill/d_${1}x${2}_$init
  done
done
