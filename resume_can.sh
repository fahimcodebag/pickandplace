#!/usr/bin/env bash
# Resume can seeds 0 and 1 to ~55k total. warm_start() checks chkpt_dir for an
# existing actor_td3 BEFORE consulting --warm-start-from, so passing the same
# flags resumes rather than restarting from the bread source.
# CAVEAT: the episode counter is NOT persisted, so --episodes is the budget for
# THIS LEG ONLY and must be recomputed after every restart.  Ledger:
#   s0  18850 + 28600 = 47450  ->  7550 remain to 55k
#   s1  18350 + 28200 = 46550  ->  8450 remain to 55k
# (a power outage on 2026-09-10 03:59 ended the second leg; the replay
#  buffers survived and resumed at 1,360,587 / 1,378,098 transitions.)
# The --critic-reset-every schedule also restarts from 0 for this leg.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
SRC=checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best
COMMON="--algo td3_ln --n-envs 5 --spawn-level 2.0 \
--updates-per-step 2 --batch-size 1024 --buffer-size 2000000 --warmup 10000 \
--lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
--target-success 1.01 --builtin-reward --require-lift --best-window 200 \
--best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant \
--actor-fc1 64 --actor-fc2 32 --fc1 512 --fc2 256 --object-type can \
--grasp-horizon 70 --warm-start-actor-only --warm-start-from $SRC"
nohup $PY "$TR" $COMMON --episodes 7550 --seed 0 --tag can_warm \
  >> Results/can/train_warm_s0.log 2>&1 &
sleep 6
nohup $PY "$TR" $COMMON --episodes 8450 --seed 1 --tag can_warm \
  >> Results/can/train_warm_s1.log 2>&1 &
echo "resumed can seeds 0 and 1"
