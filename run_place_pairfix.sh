#!/usr/bin/env bash
# Does warm-starting the CRITICS (and skipping the random warmup) stop place
# training from destroying its own initialisation?
#
# Config is the DEFAULT recipe (200k / batch 512 / critic 64x32) so the critic
# widths match the source checkpoint (td3_place_cereal_s2, itself trained at
# 64x32).  --critic-fc1/2 must NOT be set here or the critic load fails on a
# shape mismatch.
#
# Source: td3_place_cereal_s2/best, which evaluates at 85.33% end-to-end.
# The test is simply whether the run ends at or above 85.33%.  Every previous
# actor-only warm start ended below its start:
#     73.33 -> 67.77,  85.33 -> 39.33,  85.33 -> 35.22
set -u
cd "$(dirname "$0")/Decomposed state training"
PY=/home/fahim/Thesis_fahim/venv/bin/python
for SD in 0 1 2; do
  nohup $PY -u train_place.py --random-spawn --layer-norm \
    --n-envs 8 --episodes 8000 --object-type cereal --seed $SD \
    --warm-start-from ../checkpoints/td3_place_cereal_s2/best \
    --warm-start-critics --skip-warmup \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_pairfix_s$SD \
    > ../logs/train_place_pairfix_s$SD.log 2>&1 &
  sleep 4
done
echo "launched 3 pairfix runs"
