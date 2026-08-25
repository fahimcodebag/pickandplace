#!/usr/bin/env bash
# Built-in-reward arm. Same config as the gripfix/rv2 runs, but the per-step
# reward is robosuite's own staged shaped reward -- what the monolithic v7
# policy (98.7% lift-certified vs this stage's 79.2%) actually trained on.
# The structural difference is a DENSE LIFT TERM: reward rises continuously
# with object height once grasped. This stage had none.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for s in 0 1 2; do
  python "Decomposed state training/Random spawn model/train_rand.py" \
    --algo td3_ln --seed "$s" --n-envs 5 --episodes 6000 \
    --spawn-level 2.0 --updates-per-step 2 --batch-size 512 \
    --buffer-size 200000 --warmup 10000 --critic-reset-every 25000 \
    --target-success 0.95 --best-window 200 --best-margin 0.01 \
    --warm-start-from checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best \
    --require-lift --builtin-reward --tag bi \
    > "logs/bi_s${s}.out" 2>&1 &
  echo "launched seed $s (pid $!)"
done
wait
echo "BUILTIN REWARD TRAINING COMPLETE"
