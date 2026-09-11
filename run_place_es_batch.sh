#!/bin/bash
# Place training batches on CEREAL, random spawn (Results/place_training_batches.txt).
#
#   ./run_place_es_batch.sh A           custom reward + early stopping on end-to-end eval
#   ./run_place_es_batch.sh B           B': potential-based built-in reward, full length
#   ./run_place_es_batch.sh C           C': B' with shaping annealed toward sparse once
#                                       training success reaches 60% (over 200 episodes),
#                                       w 1.0 -> 0.1 over 1000 episodes, full length
# The original B (bare built-in) and C (built-in minus an idle cost) were dropped after
# the pricing audit (Results/place_training_batches.txt): bare built-in pays stalling
# more than placing, and no idle cost orders both difficulties with a usable margin.
#
# Each batch: seeds 10 11 12 from scratch AND seeds 10 11 12 warm-started (actor only,
# no random warmup) from td3_place_cerealscratch_s2/best, 89.00% on 8 held-out seeds x
# 100.  Its critic learned the custom reward, so no batch loads it; A uses the same
# actor-only start so the three batches differ only in reward and stopping.
# Recipe as run_cereal_scratch.sh: 200k buffer, batch 512, critic 64x32, LayerNorm,
# 4000 episodes, snapshots every 250, cereal_alignwarm_s0 grasp.
# Batch A also starts es_watch.py (seeds 7 31 x 50 per snapshot, patience 4
# snapshots, not before episode 1000).  B' and C' start es_watch.py --no-stop: the same
# snapshot scoring, so all three batches have identical selection data, without stopping.
# B' and C' must pass the pricing audit (Results/place_reward_audit2) before launch.
set -u
cd "$(dirname "$0")"
BATCH=${1:?A, B or C}
ONLY=${2:-}   # optional single run, e.g. scr_s10: launch only that run, start no watcher
PY=/home/fahim/Thesis_fahim/venv/bin/python
case $BATCH in
  A) TAG=cerES;   RM="--reward-mode custom" ;;
  B) TAG=cerBP;   RM="--reward-mode builtin_potential" ;;
  C) TAG=cerBPann; RM="--reward-mode builtin_potential --anneal-shaping --anneal-threshold 0.6 --anneal-window 200 --anneal-episodes 1000 --anneal-min-weight 0.1" ;;
  *) echo "batch must be A, B or C"; exit 1 ;;
esac
COMMON="--random-spawn --layer-norm --n-envs 8 --episodes 4000 --snapshot-every 250 --object-type cereal --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best $RM"
WARM="--warm-start-from ../checkpoints/td3_place_cerealscratch_s2/best --skip-warmup"
RUNS=""
cd "Decomposed state training"
for kind in scr warm; do
  for SD in 10 11 12; do
    NAME=td3_place_${TAG}_${kind}_s$SD
    [ -n "$ONLY" ] && [ "${kind}_s$SD" != "$ONLY" ] && continue
    [ -e ../checkpoints/$NAME ] && { echo "exists, not relaunching: $NAME"; continue; }
    EXTRA=""; [ $kind = warm ] && EXTRA="$WARM"
    STOP=""; [ $BATCH = A ] && STOP="--stop-file ../checkpoints/$NAME/STOP"
    nohup $PY -u train_place.py $COMMON $EXTRA $STOP --seed $SD \
      --place-chkpt-dir ../checkpoints/$NAME > ../logs/train_place_${TAG}_${kind}_s$SD.log 2>&1 &
    RUNS="$RUNS checkpoints/$NAME"
    sleep 4
  done
done
cd ..
echo "launched batch $BATCH:$RUNS"
if [ -n "$RUNS" ] && [ -z "$ONLY" ]; then
  sleep 60
  if [ $BATCH = A ]; then
    nohup $PY -u es_watch.py --runs $RUNS --out Results/place_es > Results/place_es_watch.log 2>&1 &
    echo "es_watch.py started (early stopping)"
  else
    nohup $PY -u es_watch.py --no-stop --parallel 4 --runs $RUNS --out Results/place_es_$BATCH \
      > Results/place_es_watch_$BATCH.log 2>&1 &
    echo "es_watch.py started (scoring only, no stopping)"
  fi
fi
