#!/bin/bash
# Two follow-up arms on the place stage, cereal, random spawn, same recipe as
# run_place_long2m.sh / run_place_sac.sh (critic 512x256, actor 64x32, horizon
# 150, 55,000 episodes) -- FROM SCRATCH, both.
#
#   ppo   On-policy, no replay buffer at all. Every TD3/SAC run here rises then
#         oscillates or decays; the 2M buffer hypothesis was refuted at full
#         budget, so an algorithm that cannot park on stale data is the sharpest
#         remaining test of whether the decay is a replay pathology.
#         --buffer-size is ignored. rollout_steps x n_envs then a clipped update.
#
#   sacE  SAC with target_entropy -3.5 instead of the -n_actions (-7) default.
#         The first SAC batch left every entropy knob at default and held 43-57%
#         without converging; less entropy demanded late is the SAC analogue of
#         annealing TD3's sigma. This is the knob, not a box-tick.
#
#   ./run_place_algo2.sh          both arms, seeds 40-42 / 50-52
#   ./run_place_algo2.sh ppo      one arm
set -u
cd "$(dirname "$0")"
WHICH=${1:-both}
PY=/home/fahim/Thesis_fahim/venv/bin/python

_running() {  # per-ARG match; never pkill -f
  local p pid arg
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}; [ "$pid" = "$$" ] && continue
    case "$(cat "$p/comm" 2>/dev/null)" in python*) ;; *) continue ;; esac
    while IFS= read -r -d '' arg; do
      [ "$arg" = "../checkpoints/$1" ] && return 0
    done < "$p/cmdline" 2>/dev/null
  done
  return 1
}

NENVS_ppo=24; NENVS_sacE=8
BASE="--random-spawn --layer-norm --batch-size 1024 \
--critic-fc1 512 --critic-fc2 256 --episodes 55000 --place-horizon 150 \
--snapshot-every 1000 --object-type cereal \
--grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best"

RUNS=""
cd "Decomposed state training"
for arm in ppo sacE; do
  [ "$WHICH" != both ] && [ "$WHICH" != "$arm" ] && continue
  case $arm in
    # n_envs 24 with rollout_steps 170 keeps the update batch at 170x24=4,080,
    # within 0.4% of the original 512x8=4,096 -- 3x the throughput, SAME algorithm.
    # Raising n_envs alone would have tripled the batch and cut the update count.
    ppo)  TAG=ppo_place_cer;  SEEDS="40 41 42"; EXTRA="--algo ppo --rollout-steps 170" ;;
    sacE) TAG=sacE_place_cer; SEEDS="50 51 52"; EXTRA="--algo sac --target-entropy -3.5 --buffer-size 2000000" ;;
  esac
  for SD in $SEEDS; do
    NAME=${TAG}_s$SD
    LOG=../logs/train_place_${TAG}_s$SD.log
    if _running "$NAME"; then echo "already running: $NAME"; continue; fi
    if [ -e ../checkpoints/$NAME/actor_td3 ]; then echo "resuming: $NAME"; else : > "$LOG"; echo "starting: $NAME"; fi
    eval NE=\$NENVS_$arm
    nohup $PY -u train_place.py $BASE --n-envs $NE $EXTRA --seed $SD \
      --place-chkpt-dir ../checkpoints/$NAME >> "$LOG" 2>&1 &
    RUNS="$RUNS checkpoints/$NAME"
    sleep 4
  done
done
cd ..
echo "launched:$RUNS"
