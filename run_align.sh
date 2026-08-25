#!/usr/bin/env bash
# Wrist-alignment arm. Config IDENTICAL to the gripfix runs (see
# logs/grasp_rand_td3_ln_gripfix_s0/manifest.json) with --align-grip added, so
# the only difference from the current best policy is the alignment term.
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
    --require-lift --align-grip --tag align \
    > "logs/align_s${s}.out" 2>&1 &
  echo "launched seed $s (pid $!)"
done
wait
echo "ALIGN TRAINING COMPLETE"
