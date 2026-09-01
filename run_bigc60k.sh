#!/usr/bin/env bash
# Does the large critic help GIVEN ENOUGH TRAINING?
#
# The 512/256 critic showed no reliable benefit at ~4000 episodes, but that test
# was underpowered by construction: a 159,233-parameter critic was given the same
# budget as a 5,569-parameter one, and larger approximators need more data. This
# resumes those exact runs -- critics are already 512/256, so they load and there
# is no fresh-critic collapse -- and trains to 60k with early stopping disabled.
#
# Paired against run_60k.sh's long60k arm (64/32 critics from smallc64_s5, same
# 60k budget, same flags), so critic size is again the variable. Caveat: the two
# arms resume from DIFFERENT parents, so this is not as clean as the matched
# 4000-episode test -- it answers "does the big-critic lineage catch up or pass
# the small-critic one when both train out", not a controlled size contrast.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
O=Results/long60k; mkdir -p $O
for sd in 0 1 2; do
  SRC=checkpoints/td3_grasp_rand_td3_ln_bigc512_s${sd}/best
  [ -f "$SRC/actor_td3" ] || { echo "MISSING $SRC"; continue; }
  setsid nohup $PY "$TR" --algo td3_ln --n-envs 5 --episodes 60000 --seed $sd \
    --tag bigc60k --spawn-level 2.0 --updates-per-step 2 --batch-size 512 \
    --buffer-size 200000 --warmup 10000 --lr-actor 3e-4 --lr-critic 3e-4 \
    --critic-reset-every 25000 --target-success 1.01 --builtin-reward \
    --require-lift --best-window 200 --best-margin 0.01 --probe-every 25 \
    --actor-wclip 8 --actor-fakequant --fc1 512 --fc2 256 --actor-fc1 64 \
    --actor-fc2 32 --warm-start-from $SRC \
    > $O/train_bigc60k_s${sd}.log 2>&1 < /dev/null &
done
disown -a
echo "launched 3x bigc60k (resume, 512/256 critics, no early stop)"
