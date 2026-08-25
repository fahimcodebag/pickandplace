#!/usr/bin/env bash
# Reward v2 arm (A+C+D). Config byte-identical to the gripfix runs with
# --reward-v2 added, so the reward is the only difference from the incumbent.
# Prices verified against 400 recorded episodes BEFORE training; measured
# totals matched the offline prediction to 0.1 on all five outcome classes.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1 2; do
  python "Decomposed state training/Random spawn model/train_rand.py" \
    --algo td3_ln --seed "$s" --n-envs 5 --episodes 6000 \
    --spawn-level 2.0 --updates-per-step 2 --batch-size 512 \
    --buffer-size 200000 --warmup 10000 --critic-reset-every 25000 \
    --target-success 0.85 --best-window 200 --best-margin 0.01 \
    --warm-start-from checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best \
    --require-lift --reward-v2 --tag rv2 \
    > "logs/rv2_s${s}.out" 2>&1 &
  echo "launched seed $s (pid $!)"
done
wait
echo "REWARD V2 TRAINING COMPLETE"
