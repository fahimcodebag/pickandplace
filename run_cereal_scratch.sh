#!/bin/bash
# CEREAL, RANDOM SPAWN, FROM SCRATCH -- is "random-spawn place training fails"
# about random spawn at all?
#
# Every place run that collapsed in this investigation was CEREAL, or
# WARM-STARTED, or both:
#     cereal default / cerealsmall / cerealbig / pairfix  -- cereal, warm-started
#     breadbig                                            -- bread, warm-started cross-object
#     thesis_context 9.13 attempts                        -- bread, warm-started from fixed spawn
# The only from-scratch random-spawn run is the bread curR arm of the curriculum
# pair, and it is healthy -- indistinguishable from its fixed-spawn twin.
#
# This arm is curR with exactly one change: the object (and its grasp policy).
# Same recipe (200k / 512 / critic 64x32, LayerNorm), curriculum on, no warm
# start, 4000 episodes, snapshots every 250, seeds 0-2.
#   collapses like the warm-started cereal runs -> the failure is CEREAL
#   healthy like curR                           -> the failure was WARM-STARTING
#
# Naming: deliberately NOT td3_place_cur*.  finish_curriculum_pair.sh waits on
# processes matching td3_place_cur, so that prefix would stall the bread report
# for hours; eval_place_snapshots.sh globs td3_place_cur[FR]_s* and hard-codes
# bread, so this arm needs its own evaluator anyway.
set -u
cd "$(dirname "$0")/Decomposed state training"
PY=/home/fahim/Thesis_fahim/venv/bin/python
for SD in 0 1 2; do
  nohup $PY -u train_place.py --random-spawn --layer-norm --n-envs 8 \
    --episodes 4000 --snapshot-every 250 --object-type cereal --seed $SD \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_cerealscratch_s$SD \
    > ../logs/train_place_cerealscratch_s$SD.log 2>&1 &
  sleep 4
done
echo "launched 3 cereal from-scratch random-spawn runs"
