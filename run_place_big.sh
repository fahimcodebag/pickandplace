#!/usr/bin/env bash
# Two arms, both on the grasp campaign's measured recipe, both warm-started
# from the best cereal place actor (s2, 85.33% end-to-end vs 73.33% transfer).
#
# WHY THE BIG RECIPE. The 5-seed default-config cereal run finished at 8000
# episodes and evaluated (12 eval seeds x 50, deterministic, through fsm_sim)
# at 85.33 / 76.00 / 71.83 / 69.17 / 36.50 % -- mean 67.77, sd 18.52, LOSING to
# the bread-transfer control at 73.33% (pooled -5.57, t(59)=-2.31).
# The grasp campaign's headline for buffer 2M + batch 1024 was not mean
# success, it was that it COLLAPSED SEED VARIANCE from sd 17.18 to 2.84 -- and
# 17.18 is almost exactly the 18.52 seen here.  The remedy is matched to the
# observed failure mode rather than being a generic upgrade.
# CORRECTION: an earlier version of this header cited sd ~40 and values of
# 16.33/3.50/1.17%.  That evaluation read best/ checkpoints WHILE the trainers
# were still overwriting them.  Re-run after the runs finished, the three
# affected seeds moved 16.33->69.17, 3.50->71.83, 1.17->36.50, while the two
# whose checkpoints were already stable came back bit-identical.  Never
# evaluate checkpoints owned by a running trainer.
# Critic 512/256 rides along (+3.12 INT8 / +6.04 FP32 on grasp); the critic
# never deploys, so widening it costs nothing at inference. The ACTOR stays
# 64x32 -- it is the artifact that ships to the ESP32.
#
# ARM 1 (cereal): can the variance be tamed on the object that already has a
# working seed?
# ARM 2 (bread): thesis_context 9.13 records random-spawn transport retraining
# for BREAD as a settled negative -- twice, six seeds, both losing to the
# untouched fixed-spawn original, and the abandoned logs plateau at 8-20%
# training success.  Those attempts started from the fixed-spawn policy.  This
# one starts from a policy that already works UNDER RANDOM SPAWN (cereal s2).
# If it succeeds, the earlier negative was an initialisation/exploration
# problem, not a property of the pipeline.  Two things change at once versus
# 9.13 (recipe and warm-start source), so a win is an existence result, not an
# attribution.
set -u
cd "$(dirname "$0")"
PY=/home/fahim/Thesis_fahim/venv/bin/python
SRC=../checkpoints/td3_place_cereal_s2/best
BIG="--random-spawn --layer-norm --batch-size 1024 --buffer-size 2000000 \
--critic-fc1 512 --critic-fc2 256 --n-envs 8 --episodes 8000 \
--warm-start-from $SRC"
cd "Decomposed state training"
for SD in 0 1 2; do
  nohup $PY -u train_place.py $BIG --object-type cereal --seed $SD \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_cereal_alignwarm_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_cerealbig_s$SD \
    > ../logs/train_place_cerealbig_s$SD.log 2>&1 &
  sleep 4
done
for SD in 0 1 2; do
  nohup $PY -u train_place.py $BIG --object-type bread --seed $SD \
    --grasp-chkpt-dir ../checkpoints/td3_grasp_rand_td3_ln_bi_s0/best \
    --place-chkpt-dir ../checkpoints/td3_place_breadbig_s$SD \
    > ../logs/train_place_breadbig_s$SD.log 2>&1 &
  sleep 4
done
echo "launched 3 cereal-big + 3 bread-big place runs"
