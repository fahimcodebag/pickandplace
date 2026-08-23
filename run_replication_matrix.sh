#!/usr/bin/env bash
# Replication matrix for the critic-reset (plasticity-loss) result.
#
# The seed-0 pair showed resets beating their seed-matched controls: sac hit the
# 85% stopping target at episode 1625 (control: never, peak 82% over 4000) and
# evaluates at 80.0% deterministic vs the control's 69.0%. A paired claim needs
# more than one seed, and BOTH arms must exist at each seed -- the phaseA
# controls are seed 0 only, so every new seed launches a control alongside.
#
# Sized from measured load: ~1.9 cores per run (one learner ~68%, five envs
# ~24% each), ~0.6 GB VRAM, ~7 GB RAM. Nine concurrent runs ~= 17/32 cores,
# 5.4/24 GB VRAM, 63/128 GB RAM.
set -euo pipefail
cd "$(dirname "$0")/Decomposed state training/Random spawn model"
LOGDIR=../../logs/replication
mkdir -p "$LOGDIR"

launch () {  # launch <algo> <seed> <control|reset>
  local algo=$1 seed=$2 arm=$3 extra=() tag=phaseA
  if [ "$arm" = reset ]; then tag=resetA; extra=(--critic-reset-every 25000); fi
  echo "launching $algo seed $seed $arm"
  nohup python3 -u train_rand.py --algo "$algo" --n-envs 5 --episodes 4000 \
      --seed "$seed" --tag "$tag" "${extra[@]}" \
      > "$LOGDIR/${algo}_s${seed}_${arm}.log" 2>&1 &
  echo "  pid $!"
  sleep 3          # stagger so nine MuJoCo compiles do not collide
}

# sac seed 1 is already in flight; everything else in the 2x2x2 matrix:
launch td3_ln 1 control
launch td3_ln 1 reset
launch sac    2 control
launch sac    2 reset
launch td3_ln 2 control
launch td3_ln 2 reset

echo "six launched; with the three in flight that is nine concurrent runs"
