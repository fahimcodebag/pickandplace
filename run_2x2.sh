#!/usr/bin/env bash
# 2x2: capacity x QAT-in-training, separating the two remaining levers.
#   size  64/32   = deployed (5,216 weights, 8KB INT8)
#         256/128 = 45,440 weights (~45KB INT8) -- fits: 4MB flash, 320KB SRAM,
#                   40KB arena, 9.5ms of a 50ms budget at 20Hz
#   qat   off / per-tensor INT8 fake-quant with STE on the RL return
# All arms carry --actor-wclip 8 (the confirmed +6.64 arm-level effect) and run
# FROM SCRATCH: 256/128 cannot warm-start from the 64/32 source, so warm-starting
# only the small arms would confound size with initialisation.
#
# THERMAL: 4 concurrent runs (~7 of 32 cores, ~22%). Measured earlier, 12
# concurrent arms gave only 1.2x the aggregate throughput of 4 -- contention ate
# the rest -- so throttling costs little wall time. GPU temp is logged each wave
# (no CPU sensor exists under WSL2).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
O=Results/x22; mkdir -p $O
EPISODES=${EPISODES:-12000}      # from scratch needs more than the warm-started 6000
SEEDS="${@:-0}"
run () {  # tag fc1 fc2 qatflag seed
  setsid nohup $PY "$TR" --algo td3_ln --n-envs 5 --episodes $EPISODES --seed $5 \
    --tag $1 --spawn-level 2.0 --updates-per-step 2 --batch-size 512 \
    --buffer-size 200000 --warmup 10000 --lr-actor 3e-4 --lr-critic 3e-4 \
    --critic-reset-every 25000 --target-success 0.95 --warm-start-from "" \
    --builtin-reward --require-lift --best-window 200 --best-margin 0.01 \
    --probe-every 25 --actor-wclip 8 --fc1 $2 --fc2 $3 $4 \
    > $O/train_${1}_s${5}.log 2>&1 < /dev/null &
}
for sd in $SEEDS; do
  run small_noqat 64  32  ""                 $sd
  run small_qat   64  32  "--actor-fakequant" $sd
  run big_noqat   256 128 ""                 $sd
  run big_qat     256 128 "--actor-fakequant" $sd
done
disown -a
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu --format=csv,noheader > $O/gpu_launch_s${SEEDS// /_}.txt 2>/dev/null
echo "launched 4 arms for seed(s): $SEEDS  (EPISODES=$EPISODES, ~7 of 32 cores)"
