#!/usr/bin/env bash
# (A) Extend the six 2M-buffer place runs from 8000 to ~55,000 episodes.
#
# WHY. The 2M arm scored cereal 35.22% / bread 24.39%, losing badly to the
# 200k default (67.77%) and to no training at all (73.33% transfer, 92.50%
# bread original).  The likely reason is that 2M NEVER CYCLED: the place stage
# makes ~38 transitions/episode, so 8000 episodes = ~304k transitions = 15% of
# capacity.  Its only effect was to disable the FIFO forgetting that
# place_env_wrapper.py:196-206 documents as this stage's defence against
# "hold -> small negative" transitions flooding the buffer.
# 2,000,000 / 38 = ~52,600 episodes to fill, so 55k is the first budget at
# which the 2M buffer behaves like a buffer rather than an accumulator.
#
# --warm-start-from is DELIBERATELY OMITTED: it takes precedence over resume
# and would discard the 8000 trained episodes (train_place.py, "Resume from
# checkpoint" block).  Buffer/critic flags must be repeated or load_models()
# hits a shape mismatch.
#
# NOTE the bread arm is confounded: it was warm-started from the CEREAL s2
# actor, so it is "cross-object init + big recipe", not a clean bread arm.
set -u
cd "$(dirname "$0")/Decomposed state training"
PY=/home/fahim/Thesis_fahim/venv/bin/python
BIG="--random-spawn --layer-norm --batch-size 1024 --buffer-size 2000000 \
--critic-fc1 512 --critic-fc2 256 --n-envs 8 --episodes 47000"
for SD in 0 1 2; do
  nohup $PY -u train_place.py $BIG --object-type cereal --seed $SD \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_cerealbig_s$SD \
    >> ../logs/train_place_cerealbig_s$SD.log 2>&1 &
  sleep 4
done
for SD in 0 1 2; do
  nohup $PY -u train_place.py $BIG --object-type bread --seed $SD \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_bi_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_breadbig_s$SD \
    >> ../logs/train_place_breadbig_s$SD.log 2>&1 &
  sleep 4
done
echo "(A) extended 6 runs by 47000 episodes"
