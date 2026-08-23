#!/usr/bin/env bash
# Retrain the grasp stage against a LIFT-CERTIFIED success criterion.
#
# Why: the old criterion (5 consecutive _check_grasp steps, no lift) certifies
# momentary contact. Measured at a fixed pose, the Phase B policy scored 97.5%
# on it and 0.0% once a scripted lift was required, while the ORIGINAL
# fixed-spawn model scored 100.0% / 47.5%. The newer policies score higher on
# the metric and are worse at the thing the metric is supposed to stand for --
# they learned to satisfy the contact check with a grip that cannot carry the
# object. End-to-end that showed up as 57/100 episodes failing at handoff and
# 13% place success.
#
# --require-lift makes stage 1 apply exactly the bar stage 2 applies at handoff
# (8-step hold + 20-step lift, >=3cm rise, still grasped), with partial credit
# for the rise achieved so the signal stays dense while the grip firms up.
#
# Two warm starts, because it is genuinely unclear which transfers better:
#   phaseB : can reach and orient for rotated objects, grips wrong (0% certified)
#   orig   : grips well (47.5% certified) but only at a fixed pose
set -euo pipefail
cd "$(dirname "$0")/Decomposed state training/Random spawn model"

# Small MLP: pin the thread pools or three jobs each claim ~10 cores and thrash.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

ENVS=${ENVS:-5}
EPISODES=${EPISODES:-6000}
LEVEL=${LEVEL:-2.0}
RESET=${RESET:-25000}
LOGDIR=../../logs/liftcert
mkdir -p "$LOGDIR"

launch () {  # launch <seed> <warmstart-tag> <warmstart-path>
  local seed=$1 wtag=$2 wpath=$3
  echo "launching seed $seed warm=$wtag"
  nohup python3 -u train_rand.py --algo td3_ln --n-envs "$ENVS" \
      --episodes "$EPISODES" --seed "$seed" --tag "liftcert_$wtag" \
      --spawn-level "$LEVEL" --critic-reset-every "$RESET" \
      --require-lift --warm-start-from "$wpath" \
      > "$LOGDIR/${wtag}_s${seed}.log" 2>&1 &
  echo "  pid $!"
  sleep 3
}

for seed in 0 1 2; do
  launch "$seed" phaseB checkpoints/td3_grasp_rand_td3_ln_phaseBr_s2/best
  launch "$seed" orig   checkpoints/td3_grasp
done

echo
echo "six lift-certified runs launched at level $LEVEL"
