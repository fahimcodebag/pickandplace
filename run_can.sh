#!/usr/bin/env bash
# CAN: the per-object policy that enables an EXTERNAL comparison.
#
# WHY THIS OBJECT. PickPlaceCan is robomimic's benchmark task (Mandlekar et
# al.), evaluated there under low_dim observations -- the same observation
# class this project uses. It is the only robosuite PickPlace variant with
# published third-party numbers, so it is the one place this work can be
# placed against something other than its own ablations. Bread and cereal have
# no external baseline.
#
# RECIPE = the bread/cereal winner, unchanged. Those levers were measured
# paired at 1200 episodes and none is object-specific:
#   buffer 2M + batch 1024   +15.17 INT8, collapsed seed variance sd 17.18->2.84
#   actor-wclip 8            +6.64
#   actor-fakequant          +4.43   (in-loop QAT)
#   critic 512/256           +3.12 INT8 / +6.04 FP32
#   grasp-horizon 70         adopted after measuring max successful grasp = 40
#
# NO --align-grip, AND THIS IS THE DIFFERENCE FROM CEREAL.
# Results/wrist_alignment_negative.txt refuted align-grip on BREAD: it barely
# moved corner prevalence and cost flat-grip quality. Cereal was the documented
# exception -- 30x100x150 mm, two dimensions exceeding the ~80 mm jaws, so
# exactly ONE approach axis works and yaw decides whether a grasp is possible
# at all. A CAN is a cylinder: radially symmetric about z, so yaw cannot affect
# graspability. The bread refutation transfers directly. Adding align-grip here
# would cost flat-grip quality to align an axis that does not exist.
#
# WARM START from the same bread actor as cereal, --warm-start-actor-only.
# The cereal probe settled cold-vs-warm: at episode 50 the warm run had 4
# successes and 8% grasp rate while cold runs sat at 0% with 50/50 t_no_reach
# at episode 2100. Reaching is object-independent and the bread actor supplies
# it free.
#
# EVALUATION, when these finish: compare through fsm_sim.py with
# --success-criterion robosuite, because that is what robomimic uses. For a can
# the two criteria are identical anyway (half-height 0.060 -> max(0.1, 0.090)
# = 0.1, unchanged; see bin_success.py), so the choice costs nothing and keeps
# the comparison honest.
#
# CAVEAT FOR THE PAPER: robomimic's Can numbers are IMITATION LEARNING from
# human demonstrations, not RL from scratch. They bound task difficulty; they
# are not a like-for-like baseline. Say so explicitly.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
SRC=checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best
O=Results/can; mkdir -p $O
BASE="--algo td3_ln --n-envs 5 --episodes 55000 --spawn-level 2.0 \
--updates-per-step 2 --batch-size 1024 --buffer-size 2000000 --warmup 10000 \
--lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
--target-success 1.01 --builtin-reward --require-lift --best-window 200 \
--best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant \
--actor-fc1 64 --actor-fc2 32 --fc1 512 --fc2 256 --object-type can \
--grasp-horizon 70"
for sd in 0 1 2; do
  nohup $PY "$TR" $BASE --warm-start-actor-only --warm-start-from $SRC \
    --seed $sd --tag can_warm >> $O/train_warm_s${sd}.log 2>&1 &
  sleep 8
done
echo "launched 3 can grasp runs (warm, no align-grip)"
