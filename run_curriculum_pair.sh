#!/usr/bin/env bash
# WHY DOES FIXED-SPAWN PLACE TRAINING WORK AND RANDOM-SPAWN FAIL?
#
# Every random-spawn place run this project has trained -- pairfix, cereal
# default, cerealsmall, cerealbig (~26k episodes), breadbig, and both 9.13
# attempts -- kept the reverse curriculum at frac 0.20-0.40 (max ever 0.60).
# The original fixed-spawn run ratcheted to frac 0.96 at 88% (thesis_context
# Hurdle 5 / Stage-2 outcome).
#
# Hypothesis: curriculum advancement is judged on NOISY training success
# (_success_given, noise 0.1, >=75% over 40 episodes).  A fixed handoff can
# clear that bar; a randomised one plateaus near 50%, so the curriculum never
# moves, the policy only ever practises the final ~20% of transport, and it
# is then evaluated on the full distance.  best_metric = frac50 x place50 is
# capped near 0.2 at that difficulty, which also freezes best/ on the init.
#
# DESIGN: one variable.  Both arms from scratch (the original succeeded from
# scratch), default recipe (200k / 512 / 64x32 -- the bread place model's own),
# curriculum on, same grasp policy (bi_s0 grasps fixed spawn at 100%), same
# seeds.  ONLY --random-spawn differs.  --layer-norm is kept identical in
# both so it cannot confound the pair.
# Snapshots every 250 episodes feed analyze_place_snapshots.py, which
# correlates each snapshot's TRAINING metrics (place50, frac50, metric) with
# its END-TO-END success through fsm_sim.  4000 episodes: the original reached
# frac 0.96 in ~1370.
set -u
cd "$(dirname "$0")/Decomposed state training"
PY=/home/fahim/Thesis_fahim/venv/bin/python
COMMON="--layer-norm --n-envs 8 --episodes 4000 --snapshot-every 250 \
--object-type bread \
--grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_bi_s0/best"
for SD in 0 1 2; do
  nohup $PY -u train_place.py $COMMON --seed $SD \
    --place-chkpt-dir ../checkpoints/td3_place_curF_s$SD \
    > ../logs/train_place_curF_s$SD.log 2>&1 &
  sleep 4
  nohup $PY -u train_place.py $COMMON --random-spawn --seed $SD \
    --place-chkpt-dir ../checkpoints/td3_place_curR_s$SD \
    > ../logs/train_place_curR_s$SD.log 2>&1 &
  sleep 4
done
echo "launched 3 fixed-spawn (curF) + 3 random-spawn (curR) place runs"
