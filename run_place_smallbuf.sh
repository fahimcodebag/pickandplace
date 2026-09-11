#!/usr/bin/env bash
# (B) Cereal, 200k buffer + batch 1024 + critic 512/256.
#
# Isolates the buffer.  Three arms now differ by one axis at a time:
#   default   200k / 512  / 64x32    -> 67.77%
#   big       2M   / 1024 / 512x256  -> 35.22%
#   this      200k / 1024 / 512x256  -> ?
# this vs big  = the effect of buffer size alone (the volume-dependent lever)
# this vs def  = batch + critic together (the volume-independent levers)
#
# Warm start from cereal s2, SAME OBJECT -- unlike the bread big arm, which was
# cross-object and is therefore confounded.
# 8000 episodes, matching the arms it is compared against.
set -u
cd "$(dirname "$0")/Decomposed state training"
PY=/home/fahim/Thesis_fahim/venv/bin/python
for SD in 0 1 2; do
  nohup $PY -u train_place.py --random-spawn --layer-norm \
    --batch-size 1024 --buffer-size 200000 --critic-fc1 512 --critic-fc2 256 \
    --n-envs 8 --episodes 8000 --object-type cereal --seed $SD \
    --warm-start-from ../checkpoints/td3_place_cereal_s2/best \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_cerealsmall_s$SD \
    > ../logs/train_place_cerealsmall_s$SD.log 2>&1 &
  sleep 4
done
echo "(B) launched 3 cereal 200k/1024/512x256 runs"
