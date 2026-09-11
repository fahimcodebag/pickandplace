#!/bin/bash
# Place training batches on CEREAL, random spawn (Results/place_training_batches.txt).
#
#   ./run_place_es_batch.sh A           custom reward + early stopping on end-to-end eval
#   ./run_place_es_batch.sh B           bare robosuite built-in reward, full length
#   ./run_place_es_batch.sh C COST      built-in reward minus COST per step, full length
#
# Each batch: seeds 10 11 12 from scratch AND seeds 10 11 12 warm-started (actor only,
# no random warmup) from td3_place_cerealscratch_s2/best, 89.00% on 8 held-out seeds x
# 100.  Its critic learned the custom reward, so no batch loads it; A uses the same
# actor-only start so the three batches differ only in reward and stopping.
# Recipe as run_cereal_scratch.sh: 200k buffer, batch 512, critic 64x32, LayerNorm,
# 4000 episodes, snapshots every 250, cereal_alignwarm_s0 grasp.
# Batch A also starts es_watch.py (seeds 7 31 x 50 per snapshot, patience 4
# snapshots, not before episode 1000); B and C are scored from snapshots afterwards.
set -u
cd "$(dirname "$0")"
BATCH=${1:?A, B or C}; COST=${2:-}
PY=/home/fahim/Thesis_fahim/venv/bin/python
case $BATCH in
  A) TAG=cerES;   RM="--reward-mode custom" ;;
  B) TAG=cerBI;   RM="--reward-mode builtin" ;;
  C) [ -n "$COST" ] || { echo "batch C needs the priced idle cost"; exit 1; }
     TAG=cerBIidle; RM="--reward-mode builtin_idle --idle-cost $COST" ;;
  *) echo "batch must be A, B or C"; exit 1 ;;
esac
COMMON="--random-spawn --layer-norm --n-envs 8 --episodes 4000 --snapshot-every 250 --object-type cereal --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best $RM"
WARM="--warm-start-from ../checkpoints/td3_place_cerealscratch_s2/best --skip-warmup"
RUNS=""
cd "Decomposed state training"
for kind in scr warm; do
  for SD in 10 11 12; do
    NAME=td3_place_${TAG}_${kind}_s$SD
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
if [ $BATCH = A ] && [ -n "$RUNS" ]; then
  sleep 60
  nohup $PY -u es_watch.py --runs $RUNS --out Results/place_es > Results/place_es_watch.log 2>&1 &
  echo "es_watch.py started"
fi
