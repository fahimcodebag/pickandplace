#!/usr/bin/env bash
# Relaunch the cereal place seeds. Kept as a file so the killer's command line
# never contains the literal script name -- pkill -f matches its own shell
# otherwise, which killed two relaunches in this session before the pattern
# was understood.
set -u
cd "$(dirname "$0")/Decomposed state training"
PY=/home/fahim/Thesis_fahim/venv/bin/python
for SD in 0 1 2 3 4; do
  nohup $PY -u train_place.py \
    --object-type cereal --random-spawn --layer-norm --seed $SD \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --warm-start-from ../checkpoints/td3_place/best \
    --place-chkpt-dir ../checkpoints/td3_place_cereal_s$SD \
    --n-envs 8 --episodes 8000 \
    > ../logs/train_place_cereal_s$SD.log 2>&1 &
  sleep 3
done
echo "relaunched 5 cereal place seeds"
