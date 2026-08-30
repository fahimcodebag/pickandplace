#!/usr/bin/env bash
# (a) Weight-range regularisation for per-tensor INT8.
# Per-tensor INT8 sets one scale per tensor from the largest weight, so max/std
# IS the quantisation cost (Results/int8_deployment.txt Finding 4): transport
# 5.7 quantises free, gripfix_s2 9.3 costs -7.2, bi_s0 10.0 costs -14.0.
# Config is bi_s0's manifest verbatim; --actor-wclip is the only variable.
# wclip0 is a fresh control at the same seed, so the comparison is internally
# valid rather than resting on a run from a previous week.
#
# Seeds are an argument because 4 arms leave the box 79% idle: n_envs=5 is fixed
# by bi_s0's manifest (updates_per_step = n_envs//2, so raising it would break
# comparability), and env stepping does not overlap gradient updates, so each
# arm draws only ~1.7 cores. Replication is the free axis.
#   ./run_wclip_sweep.sh 0        # seed 0 only
#   ./run_wclip_sweep.sh 1 2      # add replicates
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
WS=checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best
mkdir -p Results/wclip
SEEDS="${@:-0}"
for sd in $SEEDS; do
for k in 0 4 6 8; do
  setsid nohup $PY "$TR" --algo td3_ln --n-envs 5 --episodes 6000 --seed $sd \
    --tag wclip$k --spawn-level 2.0 --updates-per-step 2 --batch-size 512 \
    --buffer-size 200000 --warmup 10000 --lr-actor 3e-4 --lr-critic 3e-4 \
    --critic-reset-every 25000 --target-success 0.95 --warm-start-from $WS \
    --builtin-reward --require-lift --best-window 200 --best-margin 0.01 \
    --probe-every 25 --actor-wclip $k \
    > Results/wclip/train_wclip${k}_s${sd}.log 2>&1 < /dev/null &
done
done
disown -a
echo "launched grasp-training arms (wclip 0/4/6/8) for seeds: $SEEDS"
