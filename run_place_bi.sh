#!/usr/bin/env bash
# Transport retrained against the BUILT-IN-REWARD grasp distribution.
#
# Transport retraining was a negative result once before (55.0 and 25.0 vs the
# original 82%, Results/transport_retrain_negative.txt). Two things differ now:
#   1. That was against the OLD grasp distribution. The bi policy certifies
#      87.1% vs 79.2% and its grip quality is angle-independent, so what it
#      hands off is measurably different.
#   2. There is a measured mechanism pointing here -- transport_stall tripled
#      14 -> 30/36/41 per 400 episodes -- rather than a hypothesis.
#
# Lessons applied from that negative result:
#   - NO critic resets. They cost the place stage ~30 points (opposite sign to
#     the grasp stage, where the same remedy was worth +20 to +30).
#   - best_metric is frac x place-success, so it mixes curriculum progress with
#     skill. Judge these arms ONLY by end-to-end eval at matched episodes.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
GRASP=checkpoints/td3_grasp_rand_td3_ln_bi_s0/best
for s in 0 1 2; do
  python "Decomposed state training/train_place.py" \
    --n-envs 5 --episodes 8000 --random-spawn \
    --grasp-chkpt-dir "$GRASP" \
    --warm-start-from checkpoints/td3_place/best \
    --place-chkpt-dir "checkpoints/td3_place_bi_s${s}" \
    > "logs/place_bi_s${s}.out" 2>&1 &
  echo "launched seed $s (pid $!)"
done
wait
echo "PLACE-BI TRAINING COMPLETE"
