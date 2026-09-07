#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
O=Results/cereal
SRC=checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best
BASE="--algo td3_ln --n-envs 5 --episodes 55000 --spawn-level 2.0 \
--updates-per-step 2 --batch-size 1024 --buffer-size 2000000 --warmup 10000 \
--lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
--target-success 1.01 --builtin-reward --require-lift --best-window 200 \
--best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant \
--actor-fc1 64 --actor-fc2 32 --fc1 512 --fc2 256 --object-type cereal"
# Warm arms at horizon 70: the slowest measured cereal success takes 40 steps,
# so 200 spent ~3x the simulation on failure tails. Paired with the truncation
# fix -- a short horizon makes timeouts common, and storing them as terminal
# would bias the critic.
for sd in 0 1 2 3; do
  for arm in base align; do
    FLAG=""; [ "$arm" = "align" ] && FLAG="--align-grip"
    nohup $PY "$TR" $BASE --grasp-horizon 70 --warm-start-actor-only \
      --warm-start-from $SRC --seed $sd --tag cereal_${arm}warm $FLAG \
      >> $O/train_${arm}warm_s${sd}.log 2>&1 &
    sleep 10
  done
done
wait
