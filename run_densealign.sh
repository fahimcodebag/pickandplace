#!/usr/bin/env bash
# Dense-alignment arm. Same config as rv2 plus --dense-align.
# PRIMARY METRIC IS corr(Q(s,pi(s)), alignment), not success rate: the reward
# already correlates +0.85 with alignment while the critic correlates -0.004,
# and the actor's gradient is dQ/da. If the critic still cannot see alignment
# after this, the reward path is exhausted.
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
    --require-lift --reward-v2 --dense-align --tag da \
    > "logs/da_s${s}.out" 2>&1 &
  echo "launched seed $s (pid $!)"
done
wait
echo "DENSE ALIGN TRAINING COMPLETE"
