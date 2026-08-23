#!/usr/bin/env bash
# Phase B: add z-rotation to the random spawn (--spawn-level 2.0).
#
# Level semantics (grasp_spawn_wrapper.py): 0.1-1.0 grows the position box with
# rotation pinned at 0; 1.0-2.0 keeps the full box and opens z-rotation, with
# 2.0 = full rotation. Phase A trained and evaluated entirely at 1.0, so
# rotation is genuinely new capability, not more of the same.
#
# Warm-started from Phase A's winner (td3_ln + critic resets), which is the
# same decision Sec 9.3 made when it reseeded Phase A from the 0.355 artifact.
# Resets stay on: they were worth +20.3 points at level 1.0 and the plasticity
# pressure only increases with a wider data distribution.
#
# Controls at level 2.0 (no resets) run alongside so the reset claim is tested
# again at the harder level rather than assumed to carry over.
set -euo pipefail
cd "$(dirname "$0")/Decomposed state training/Random spawn model"

ENVS=${ENVS:-5}
EPISODES=${EPISODES:-6000}     # rotation is harder; give it more than Phase A
LEVEL=${LEVEL:-2.0}
RESET=${RESET:-25000}
WARM=${WARM:-checkpoints/td3_grasp_rand_td3_ln_resetA_s0/best}
LOGDIR=../../logs/phaseB
mkdir -p "$LOGDIR"

launch () {  # launch <seed> <control|reset>
  local seed=$1 arm=$2 extra=() tag=phaseB
  if [ "$arm" = reset ]; then tag=phaseBr; extra=(--critic-reset-every "$RESET"); fi
  echo "launching td3_ln seed $seed $arm (level $LEVEL)"
  nohup python3 -u train_rand.py --algo td3_ln --n-envs "$ENVS" \
      --episodes "$EPISODES" --seed "$seed" --tag "$tag" \
      --spawn-level "$LEVEL" --warm-start-from "$WARM" "${extra[@]}" \
      > "$LOGDIR/td3_ln_s${seed}_${arm}.log" 2>&1 &
  echo "  pid $!"
  sleep 3
}

for seed in 0 1 2; do
  launch "$seed" reset
  launch "$seed" control
done

echo
echo "six runs launched at spawn level $LEVEL, warm-started from $WARM"
echo "monitor: tail -f $LOGDIR/*.log"
