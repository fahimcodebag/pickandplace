#!/usr/bin/env bash
# Route (b): QAT inside the RL loop, against the wclip8 arm as control.
#
# The control is NOT re-run: checkpoints/td3_grasp_rand_td3_ln_wclip8_s{0,1,2}
# were trained with this exact command minus --actor-fakequant, so they are a
# matched control at matched seeds. Only the QAT arm is new.
#
# This is a different objective from the refuted qat_finetune.py, which
# minimised MSE to an FP32 teacher on a frozen buffer -- Fnd 2 of
# Results/int8_deployment.txt measured that objective as ANTI-correlated with
# deployed success (best corr 0.952 -> worst behaviour 37.1%). Here the RL
# return itself is optimised under per-tensor fake-quant with STE gradients.
#
# THERMAL: 3 concurrent runs, ~5 of 32 cores (~16%).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
WS=checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best
O=Results/qat8; mkdir -p $O
for sd in 0 1 2; do
  setsid nohup $PY "$TR" --algo td3_ln --n-envs 5 --episodes 6000 --seed $sd \
    --tag qat8 --spawn-level 2.0 --updates-per-step 2 --batch-size 512 \
    --buffer-size 200000 --warmup 10000 --lr-actor 3e-4 --lr-critic 3e-4 \
    --critic-reset-every 25000 --target-success 0.95 --warm-start-from $WS \
    --builtin-reward --require-lift --best-window 200 --best-margin 0.01 \
    --probe-every 25 --actor-wclip 8 --actor-fakequant \
    > $O/train_qat8_s${sd}.log 2>&1 < /dev/null &
done
disown -a
echo "launched 3 QAT arms (control = existing wclip8 s0/s1/s2)"
