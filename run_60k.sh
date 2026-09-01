#!/usr/bin/env bash
# The convergence diagnostic. Every result in this campaign was measured on a
# run that stopped when a NOISY 100-episode window touched 95% grasp success --
# never on convergence. cont64 then gained +4.17 INT8 from nothing but more
# episodes, so headroom demonstrably remained. This finds where the curve
# actually flattens.
#
# --target-success 1.01 disables the early stop (the comparison is
# mean(success_hist[-100:]) >= target, and a mean of 0/1 flags cannot exceed 1).
#
# TWO QUESTIONS, TWO ARMS:
#   long60k_sN  RESUME from the validated smallc64_s5 (its own 64/32 critics
#               load, so no fresh-critic collapse). Answers the one that
#               matters for the thesis: is 93.08% INT8 near the ceiling, or
#               were the ablations all measured mid-climb?
#   scratch60k  FROM SCRATCH, directly comparable to v7's 56k-episode curve.
#               Answers whether decomposed grasp converges from zero at all --
#               no random-spawn policy in this project ever has, and the
#               aborted 2x2 died at 12k. High risk, one seed, bonus data.
#
# THERMAL: 4 runs, ~7 of 32 cores (~22%).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
SRC=checkpoints/td3_grasp_rand_td3_ln_smallc64_s5/best
O=Results/long60k; mkdir -p $O
common () {
  echo --algo td3_ln --n-envs 5 --episodes 60000 --spawn-level 2.0 \
    --updates-per-step 2 --batch-size 512 --buffer-size 200000 --warmup 10000 \
    --lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
    --target-success 1.01 --builtin-reward --require-lift --best-window 200 \
    --best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant
}
for sd in 0 1 2; do
  setsid nohup $PY "$TR" $(common) --seed $sd --tag long60k \
    --warm-start-from $SRC \
    > $O/train_long60k_s${sd}.log 2>&1 < /dev/null &
done
setsid nohup $PY "$TR" $(common) --seed 0 --tag scratch60k --warm-start-from "" \
  > $O/train_scratch60k_s0.log 2>&1 < /dev/null &
disown -a
echo "launched 3x long60k (resume) + 1x scratch60k"
