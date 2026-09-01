#!/usr/bin/env bash
# Does replay-buffer size explain the seed variance?
#
# A buffer larger than the data collected is a NO-OP. At ~40 steps/episode a
# 25k-episode run collects ~1.0M transitions, so:
#     200k -> retains  20%  (current setting, aggressive forgetting)
#     500k -> retains  50%
#       1M -> retains 100%  (no forgetting at all)
# Anything above 1M is identical to 1M at this run length, which is why the
# sweep stops there rather than chasing v7's 4M.
#
# CALIBRATION: v7 retained 4M/28M = 14.3%; the current setting retains
# 200k/702k = 28.5%. This project already forgets LESS than v7 did in
# proportional terms, so the forgetting hypothesis is weaker than it first
# looked. 1M still eliminates forgetting entirely, which is the clean test.
#
# All arms resume from the validated smallc64_s5, early stopping disabled,
# 25k episodes -- past the point where the 60k diagnostic showed the curve flat.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
SRC=checkpoints/td3_grasp_rand_td3_ln_smallc64_s5/best
O=Results/buffer; mkdir -p $O
for sd in 0 1 2; do
  for b in 200000 500000 1000000; do
    tag="buf$((b/1000))k"
    setsid nohup $PY "$TR" --algo td3_ln --n-envs 5 --episodes 25000 --seed $sd \
      --tag $tag --spawn-level 2.0 --updates-per-step 2 --batch-size 512 \
      --buffer-size $b --warmup 10000 --lr-actor 3e-4 --lr-critic 3e-4 \
      --critic-reset-every 25000 --target-success 1.01 --builtin-reward \
      --require-lift --best-window 200 --best-margin 0.01 --probe-every 25 \
      --actor-wclip 8 --actor-fakequant --warm-start-from $SRC \
      > $O/train_${tag}_s${sd}.log 2>&1 < /dev/null &
  done
done
disown -a
echo "launched 9 runs: buffer 200k/500k/1M x seeds 0,1,2"
