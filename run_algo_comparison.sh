#!/usr/bin/env bash
# Launch the §9 random-spawn algorithm comparison concurrently.
#
# Four variants, identical environment / reward / spawn / logging / best-metric,
# one variable each. On the 5950X (16C/32T) these fit side by side: 4 runs x
# 5 envs = 20 simulation processes, leaving cores for the learners.
#
# The historical update-to-data ratio (0.5 gradient steps per env-step) is
# preserved by train_rand.py's default (--updates-per-step = n_envs//2), so a
# difference between runs is an ALGORITHM difference, not a schedule artifact.
set -euo pipefail
cd "$(dirname "$0")/Decomposed state training/Random spawn model"

ENVS=${ENVS:-5}
EPISODES=${EPISODES:-4000}
SEED=${SEED:-0}
TAG=${TAG:-phaseA}
LOGDIR=${LOGDIR:-../../logs/comparison_${TAG}_s${SEED}}
mkdir -p "$LOGDIR"

launch () {  # launch <algo> [extra args...]
  local algo=$1; shift
  echo "launching $algo -> $LOGDIR/$algo.log"
  nohup python3 -u train_rand.py --algo "$algo" --n-envs "$ENVS" \
      --episodes "$EPISODES" --seed "$SEED" --tag "$TAG" "$@" \
      > "$LOGDIR/$algo.log" 2>&1 &
  echo "  pid $!"
}

launch td3
launch td3_ln
launch sac
launch ppo

cat <<EOF

All four launched. Monitor with:
  tail -f $LOGDIR/*.log
  tensorboard --logdir ../../logs
Compare when done:
  column -s, -t < ../../logs/grasp_rand_<algo>_${TAG}_s${SEED}/episodes.csv | less -S
EOF
wait
