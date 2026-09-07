#!/usr/bin/env bash
# Restart every cereal run after a power cut. One command, detached, resumes
# from checkpoints + replay buffer. The lab loses power regularly, so this is
# the recovery path -- do NOT launch training with a bare `setsid ... &` from a
# tool call: if the caller is killed on a timeout the children die with it.
#
# Horizon 70 on BOTH arms now. Measured steps-to-success:
#   warm policy (61% grasp): median 26  p95 32  MAX 40
#   cold policy (26% grasp): median 30  p95 42  MAX 45
# Horizon 70 keeps 100% of successes in both, with 55% margin over the slowest.
# I first kept the cold runs at 200 assuming an unskilled policy needs longer
# lucky episodes; measuring refuted that.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
O=Results/cereal; mkdir -p $O
SRC=checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best
BASE="--algo td3_ln --n-envs 5 --episodes 55000 --spawn-level 2.0 \
--updates-per-step 2 --batch-size 1024 --buffer-size 2000000 --warmup 10000 \
--lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
--target-success 1.01 --builtin-reward --require-lift --best-window 200 \
--best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant \
--actor-fc1 64 --actor-fc2 32 --fc1 512 --fc2 256 --object-type cereal \
--grasp-horizon 70"
# warm arms: 4 seeds x {base, align}
for sd in 0 1 2 3; do
  for arm in base align; do
    FLAG=""; [ "$arm" = "align" ] && FLAG="--align-grip"
    nohup $PY "$TR" $BASE --warm-start-actor-only --warm-start-from $SRC \
      --seed $sd --tag cereal_${arm}warm $FLAG \
      >> $O/train_${arm}warm_s${sd}.log 2>&1 &
    sleep 8
  done
done
# cold controls: 2 seeds x {base, align}
for sd in 0 1; do
  for arm in base align; do
    FLAG=""; [ "$arm" = "align" ] && FLAG="--align-grip"
    nohup $PY "$TR" $BASE --cold-start --seed $sd --tag cereal_${arm} $FLAG \
      >> $O/train_${arm}_s${sd}.log 2>&1 &
    sleep 8
  done
done
wait
