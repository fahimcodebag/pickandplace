#!/usr/bin/env bash
# 2M buffer + batch 1024 at 55k episodes, crossed with critic size.
#
# WHY THESE SETTINGS ARE FIXED, NOT VARIED:
#   buffer 2M   fills at ~50k episodes, so 55k is the shortest run at which 2M
#               is not simply identical to 1M. The 1M arm was still climbing
#               when the 25k sweep ended (92.7 -> 94.4), so there is headroom.
#   batch 1024  a larger buffer only pays if you sample its diversity; 512 out
#               of 2M sees a smaller fraction of the distribution than 512 out
#               of 200k did.
# Both are well supported, so they go in BOTH arms rather than being tested.
#
# WHAT IS VARIED: critic width. The earlier null (512/256, no reliable effect)
# was measured at ~4000 episodes on a 200k buffer -- a 159,233-parameter critic
# on the same budget as a 5,569-parameter one, almost certainly underfit. At
# 55k episodes with a 2M buffer that objection no longer applies, and the 60k
# diagnostic showed the big-critic arm ahead by ~3.4 grasp points at 17k.
#
# BOTH arms use --warm-start-actor-only so critics start fresh in both. A
# warm-started small-critic control against a necessarily-fresh large-critic
# arm would confound critic SIZE with INITIALISATION -- the exact error that
# made Results/capacity_experiments.txt unreadable.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
SRC=checkpoints/td3_grasp_rand_td3_ln_buf500k_s1/best
O=Results/big2m; mkdir -p $O
common () {
  echo --algo td3_ln --n-envs 5 --episodes 55000 --spawn-level 2.0 \
    --updates-per-step 2 --batch-size 1024 --buffer-size 2000000 --warmup 10000 \
    --lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
    --target-success 1.01 --builtin-reward --require-lift --best-window 200 \
    --best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant \
    --actor-fc1 64 --actor-fc2 32 --warm-start-actor-only --warm-start-from $SRC
}
for sd in 0 1 2 3; do
  setsid nohup $PY "$TR" $(common) --seed $sd --tag c2m512 --fc1 512 --fc2 256 \
    > $O/train_c2m512_s${sd}.log 2>&1 < /dev/null &
  setsid nohup $PY "$TR" $(common) --seed $sd --tag c2m64 --fc1 64 --fc2 32 \
    > $O/train_c2m64_s${sd}.log 2>&1 < /dev/null &
done
disown -a
echo "launched 8 runs: critic 512/256 vs 64/32, x4 seeds, 2M buffer, batch 1024, 55k eps"
