#!/bin/bash
# Long 2M-recipe place training on CEREAL, random spawn: does the late decay survive a
# budget the recipe was actually designed for, and does annealing the exploration noise
# help?  (Results/place_training_batches.txt, thesis_context 9.18.)
#
#   ./run_place_long2m.sh            launch both arms (3 seeds each)
#   ./run_place_long2m.sh noise      launch only the annealed arm
#   ./run_place_long2m.sh plain      launch only the constant-noise control
#
# WHY THIS BUDGET.  The 2M buffer needs ~40-52k episodes to fill; every earlier arm was
# judged at 8,000, where it is 15-20% full and never cycles -- which is how the recipe
# earned its -28.44 verdict.  cerealbig_s2, the one run resumed to ~26,000 episodes,
# crossed curriculum 0.30 only at ~9,800 episodes of its resumed segment and ended at
# 0.74 with final weights scoring 67.62% held-out -- the only place run whose final
# weights beat its own best/.  55,000 episodes tests that directly.
#
# ARMS (identical except exploration noise)
#   plain  sigma 0.1 constant -- the incumbent, and what every place run so far used
#   noise  sigma 0.1 -> 0.02 over the first 20,000 episodes, then held.  Constant 0.1 is
#          ~20% of the scaled translation command at every step, including near the
#          0.10 m release radius where precision decides success vs release_miss.
#
# PLACE HORIZON 150.  Measured on recorded episodes: successful place episodes run to a
# median 16 steps at curriculum 0.3 and 104 at full distance, longest 145.  150 trims the
# tail without cutting successes (50 would cut ~97% of full-distance ones).  It counts
# only post-handoff steps; the grasp rollout, test-lift and scripted carry run inside
# reset() and are unaffected.
#
# Warm-started from td3_place_cerealscratch_s2/best (89.00% on 8 held-out seeds x 100),
# matching the condition cerealbig_s2 ran under and the warm arms of the three batches.
# Memory: ~8.8 GB per run (2M buffer), so 6 runs ~53 GB of 98 GB -- see
# run_place_queue.sh and the OOM incident before launching anything alongside these.
set -u
cd "$(dirname "$0")"
WHICH=${1:-both}
PY=/home/fahim/Thesis_fahim/venv/bin/python
BIG="--random-spawn --layer-norm --n-envs 8 --batch-size 1024 --buffer-size 2000000 \
--critic-fc1 512 --critic-fc2 256 --episodes 55000 --place-horizon 150 \
--snapshot-every 1000 --object-type cereal \
--grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best"
WARM="--warm-start-from ../checkpoints/td3_place_cerealscratch_s2/best --skip-warmup"
RUNS=""
cd "Decomposed state training"
for arm in plain noise; do
  [ "$WHICH" != both ] && [ "$WHICH" != "$arm" ] && continue
  case $arm in
    plain) TAG=cer2Mplain; EXTRA="" ;;
    noise) TAG=cer2Mnoise; EXTRA="--noise-start 0.1 --noise-final 0.02 --noise-anneal-episodes 20000" ;;
  esac
  for SD in 20 21 22; do
    NAME=td3_place_${TAG}_s$SD
    [ -e ../checkpoints/$NAME ] && { echo "exists, not relaunching: $NAME"; continue; }
    nohup $PY -u train_place.py $BIG $WARM $EXTRA --seed $SD \
      --place-chkpt-dir ../checkpoints/$NAME > ../logs/train_place_${TAG}_s$SD.log 2>&1 &
    RUNS="$RUNS checkpoints/$NAME"
    sleep 4
  done
done
cd ..
echo "launched:$RUNS"
if [ -n "$RUNS" ]; then
  sleep 90
  nohup $PY -u es_watch.py --no-stop --parallel 4 --runs $RUNS --out Results/place_long2m \
    > Results/place_long2m_watch.log 2>&1 &
  echo "es_watch.py started (scoring only, no stopping)"
fi
