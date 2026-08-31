#!/usr/bin/env bash
# Large host-side critics with the SMALL deployable actor.
#
# Motivation: widening the actor to 128/64 cost -5.93 INT8 points, and the
# likely cause was an asymmetry I introduced -- actor/critic capacity went from
# 0.96 to 2.64 while v7 (which does grasp well) sits at 1.00. TD3's actor
# ascends the CRITIC's Q estimate, so a bigger actor mainly gets better at
# exploiting a fixed critic's errors. This inverts the fix: leave the actor at
# the deployed 64/32 (8088 B INT8, unchanged) and give the critic 28x capacity.
# Critics never leave the host (Sec 7), so this is free at deployment.
#
# BOTH arms initialise critics FRESH so that critic SIZE is the only variable;
# a warm-started control would confound size with initialisation. Both
# warm-start the actor from the cont64 lineage (currently the best: 83.58% INT8).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
O=Results/bigcritic; mkdir -p $O
common () {
  echo --algo td3_ln --n-envs 5 --episodes 6000 --spawn-level 2.0 \
    --updates-per-step 2 --batch-size 512 --buffer-size 200000 --warmup 10000 \
    --lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
    --target-success 0.95 --builtin-reward --require-lift --best-window 200 \
    --best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant \
    --actor-fc1 64 --actor-fc2 32 --warm-start-actor-only
}
for sd in ${@:-0 1 2 3 4 5}; do
  SRC=checkpoints/td3_grasp_rand_td3_ln_cont64_s${sd}/best
  [ -f "$SRC/actor_td3" ] || { echo "MISSING $SRC"; continue; }
  setsid nohup $PY "$TR" $(common) --seed $sd --tag bigc512 \
    --fc1 512 --fc2 256 --warm-start-from $SRC \
    > $O/train_bigc512_s${sd}.log 2>&1 < /dev/null &
  setsid nohup $PY "$TR" $(common) --seed $sd --tag smallc64 \
    --fc1 64 --fc2 32 --warm-start-from $SRC \
    > $O/train_smallc64_s${sd}.log 2>&1 < /dev/null &
done
disown -a
echo "launched bigc512 + smallc64 for seeds: ${@:-0 1 2 3 4 5}"
