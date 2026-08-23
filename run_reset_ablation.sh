#!/usr/bin/env bash
# Primacy-bias / plasticity-loss ablation for the §9 random-spawn grasp.
#
# Phase A established that EVERY algorithm decays after peaking (SAC -20,
# td3_ln -28, ppo -31, td3 -38). PPO decaying without a replay buffer ruled
# out stale replay composition. The surviving explanation is plasticity loss:
# as the policy improves its own data narrows onto near-success trajectories,
# and the critic over-specializes to that slice.
#
# This runs the remedy (Nikishin et al.: periodic partial resets of the critic
# output layer, trunk and buffer retained) against the two Phase A runs that
# already have a no-reset control at the same seed. ONE variable changes:
# --critic-reset-every. Everything else -- env, reward, spawn level, warm
# start, UTD, batch, lr, seed -- is identical to the phaseA manifests.
#
# Read the result as: does the long-run trend stop decaying, and does each
# post-reset recovery return ABOVE the pre-reset level?
#   plasticity loss    -> dip at each reset, recovery to a higher plateau
#   task contradiction -> resets change nothing; decay continues regardless
set -euo pipefail
cd "$(dirname "$0")/Decomposed state training/Random spawn model"

ENVS=${ENVS:-5}
EPISODES=${EPISODES:-4000}
SEED=${SEED:-0}
TAG=${TAG:-resetA}
# SAC logged 191042 grad steps over 4000 episodes (~48/ep); 25k -> ~7 resets.
RESET=${RESET:-25000}
LOGDIR=${LOGDIR:-../../logs/comparison_${TAG}_s${SEED}}
mkdir -p "$LOGDIR"

launch () {  # launch <algo> [extra args...]
  local algo=$1; shift
  echo "launching $algo (reset every $RESET grad steps) -> $LOGDIR/$algo.log"
  nohup python3 -u train_rand.py --algo "$algo" --n-envs "$ENVS" \
      --episodes "$EPISODES" --seed "$SEED" --tag "$TAG" \
      --critic-reset-every "$RESET" "$@" \
      > "$LOGDIR/$algo.log" 2>&1 &
  echo "  pid $!"
}

launch sac
launch td3_ln

cat <<BANNER

Two reset arms launched. Controls already exist at the same seed:
  logs/grasp_rand_sac_phaseA_s0      (critic_reset_every: 0)
  logs/grasp_rand_td3_ln_phaseA_s0   (critic_reset_every: 0)

Monitor:
  tail -f $LOGDIR/*.log
BANNER
wait
