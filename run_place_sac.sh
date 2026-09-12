#!/bin/bash
# SAC on the place stage, CEREAL, random spawn -- same recipe as the 2M TD3 arms
# (run_place_long2m.sh), algorithm swapped. Tests whether the late collapse every
# TD3 seed showed (peak 20-93% at 1k-9k, then 0-10% for 6,000+ episodes) is a
# property of TD3 rather than of the task or the reward.
#
#   ./run_place_sac.sh            launch all three seeds
#   ./run_place_sac.sh 30         launch one seed
#
# FROM SCRATCH, deliberately. SAC's GaussianActor has mean AND log-std heads, so
# a TD3 actor's weights do not fit it; there is no warm start to inherit.
#
# EXPLORATION is learned (entropy-regularised), so --noise-start/--noise-final do
# nothing here. That is the point: constant sigma 0.1 was a suspect in the TD3
# collapse, and SAC removes the knob instead of tuning it.
#
# ACTOR STAYS 64x32 -- the deployed artifact. sac.Agent takes actor_layer1/2
# separately from layer1/2_size, so the critics are 512x256 without dragging the
# actor up with them (verified: actor (64,46)(32,64)(7,32)x2, critic (512,53)).
#
# MEMORY: ~8.6 GB per run at a full 2M float64 buffer. The six TD3 runs already
# hold ~58 GB of 98 GB, so launch in WAVES and measure between them -- 18 runs at
# once once OOM-crashed this VM. Never let the projected total pass ~70 GB.
set -u
cd "$(dirname "$0")"
PY=/home/fahim/Thesis_fahim/venv/bin/python

_running() {  # per-ARG match on /proc/*/cmdline; never substring-match the whole line
  local p pid arg
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    case "$(cat "$p/comm" 2>/dev/null)" in python*) ;; *) continue ;; esac
    while IFS= read -r -d '' arg; do
      [ "$arg" = "../checkpoints/$1" ] && return 0
    done < "$p/cmdline" 2>/dev/null
  done
  return 1
}

BIG="--algo sac --random-spawn --layer-norm --n-envs 8 --batch-size 1024 \
--buffer-size 2000000 --critic-fc1 512 --critic-fc2 256 --episodes 55000 \
--place-horizon 150 --snapshot-every 1000 --object-type cereal \
--grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best"

SEEDS=${@:-30 31 32}
RUNS=""
cd "Decomposed state training"
for SD in $SEEDS; do
  NAME=sac_place_cer2M_s$SD
  LOG=../logs/train_place_sac_s$SD.log
  if _running "$NAME"; then echo "already running, left alone: $NAME"; continue; fi
  if [ -e ../checkpoints/$NAME/actor_sac ] || [ -e ../checkpoints/$NAME/actor ]; then
    START=""; echo "resuming: $NAME"       # see run_place_long2m.sh on resume vs warm start
  else
    START=""; : > "$LOG"; echo "starting: $NAME"
  fi
  nohup $PY -u train_place.py $BIG $START --seed $SD \
    --place-chkpt-dir ../checkpoints/$NAME >> "$LOG" 2>&1 &
  RUNS="$RUNS checkpoints/$NAME"
  sleep 4
done
cd ..
echo "launched:$RUNS"
