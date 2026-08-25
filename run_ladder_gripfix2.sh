#!/usr/bin/env bash
# End-to-end ladder for the selection-fixed (w200) gripfix grasp policies.
#
# Four grasp policies x two eval seeds x 200 episodes = 400 episodes each.
# The baseline (liftcert_phaseB_s2) is RE-measured under the same protocol:
# its published 82% came from 100 episodes, and 100-episode evals have run
# ~10 points optimistic in this project. Comparing a 400-episode number to a
# 100-episode one is the exact trap that produced the earlier 72-vs-82 split.
#
# Transport stage is the ORIGINAL fixed-spawn policy in every arm -- retraining
# it was a negative result (see Results/transport_retrain_negative.txt).
set -u
cd "$(dirname "$0")"
OUT=Results/ladder_gripfix2
EPISODES=200
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

declare -A GRASP=(
  [gripfix_s0]=checkpoints/td3_grasp_rand_td3_ln_gripfix_s0/best
  [gripfix_s1]=checkpoints/td3_grasp_rand_td3_ln_gripfix_s1/best
  [gripfix_s2]=checkpoints/td3_grasp_rand_td3_ln_gripfix_s2/best
  [baseline_liftcert_s2]=checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best
)

for name in "${!GRASP[@]}"; do
  for eseed in 7 123; do
    log="$OUT/${name}_eval${eseed}.log"
    python "Decomposed state training/test_place.py" \
      --episodes "$EPISODES" \
      --seed "$eseed" \
      --random-spawn \
      --best \
      --grasp-chkpt-dir "${GRASP[$name]}" \
      > "$log" 2>&1 &
    echo "launched $name eval${eseed} -> $log (pid $!)"
  done
done
wait
echo "ALL LADDER RUNS COMPLETE"
