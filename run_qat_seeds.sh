#!/usr/bin/env bash
# Extend BOTH arms to 6 training seeds, using capacity that is otherwise idle.
# n_envs stays 5: it is fixed by the wclip8 control's manifest, and changing it
# would alter the update-to-data ratio and break the matched comparison. So the
# spare cores buy seeds (power), not speed -- the arms' spread (INT8 sd 6.59 at
# n=3) is what currently limits resolution, not episode throughput.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
WS=checkpoints/td3_grasp_rand_td3_ln_liftcert_phaseB_s2/best
mkdir -p Results/qat8 Results/wclip
common () {
  echo --algo td3_ln --n-envs 5 --episodes 6000 --spawn-level 2.0 \
    --updates-per-step 2 --batch-size 512 --buffer-size 200000 --warmup 10000 \
    --lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
    --target-success 0.95 --warm-start-from $WS --builtin-reward --require-lift \
    --best-window 200 --best-margin 0.01 --probe-every 25 --actor-wclip 8
}
for sd in "$@"; do
  setsid nohup $PY "$TR" $(common) --seed $sd --tag qat8 --actor-fakequant \
    > Results/qat8/train_qat8_s${sd}.log 2>&1 < /dev/null &
  setsid nohup $PY "$TR" $(common) --seed $sd --tag wclip8 \
    > Results/wclip/train_wclip8_s${sd}.log 2>&1 < /dev/null &
done
disown -a
echo "launched QAT + control for seeds: $*"
