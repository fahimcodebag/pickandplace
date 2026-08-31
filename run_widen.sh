#!/usr/bin/env bash
# Capacity via Net2WiderNet: 64/32 -> 128/64, transferred from the trained QAT
# policies so training RESUMES rather than restarts. A wider actor cannot
# inherit the 5-stage warm-start curriculum by shape, and the aborted 2x2
# showed cold starts fail at every width, so transfer is the only clean route.
#
# TWO ARMS, because widening also grants 6000 extra episodes:
#   wide128_sN  widen qat8_sN to 128/64, then train 6000
#   cont64_sN   continue qat8_sN at 64/32, then train 6000   <- isolates WIDTH
# Without cont64 an improvement could just be the extra training.
#
# Critics stay 64/32: they never deploy, Q(s,a) does not scale with actor
# width, and Net2Wider through the td3_ln LayerNorm would not preserve the
# function anyway.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
O=Results/widen; mkdir -p $O
SEEDS="${@:-0 1 2 3 4 5}"
common () {
  echo --algo td3_ln --n-envs 5 --episodes 6000 --spawn-level 2.0 \
    --updates-per-step 2 --batch-size 512 --buffer-size 200000 --warmup 10000 \
    --lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
    --target-success 0.95 --builtin-reward --require-lift --best-window 200 \
    --best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant
}
for sd in $SEEDS; do
  SRC=checkpoints/td3_grasp_rand_td3_ln_qat8_s${sd}/best
  [ -f "$SRC/actor_td3" ] || { echo "MISSING $SRC"; continue; }
  W=checkpoints/wide128_from_qat8_s${sd}
  $PY net2wider.py --src $SRC --dst $W --fc1 128 --fc2 64 \
      --split-noise 0.5 --seed $sd > $O/widen_s${sd}.log 2>&1
  grep -o "function preservation.*" $O/widen_s${sd}.log | sed "s/^/  s$sd /"
  setsid nohup $PY "$TR" $(common) --seed $sd --tag wide128 \
    --actor-fc1 128 --actor-fc2 64 --warm-start-from $W \
    > $O/train_wide128_s${sd}.log 2>&1 < /dev/null &
  setsid nohup $PY "$TR" $(common) --seed $sd --tag cont64 \
    --warm-start-from $SRC \
    > $O/train_cont64_s${sd}.log 2>&1 < /dev/null &
done
disown -a
echo "launched wide128 + cont64 for seeds: $SEEDS"
