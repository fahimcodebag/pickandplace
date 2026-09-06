#!/usr/bin/env bash
# Cereal: the first per-object policy (Results/object_generalisation.txt).
#
# RECIPE = the bread winner (run_2m.sh), because those levers were measured
# paired over 1200 episodes and none of them is object-specific:
#   buffer 2M + batch 1024   +15.17 INT8, and it collapsed seed variance
#                            (sd 17.18 -> 2.84) -- the largest lever found
#   actor-wclip 8            +6.64
#   actor-fakequant          +4.43   (in-loop QAT)
#   critic 512/256           +3.12 INT8 / +6.04 FP32
#   55k episodes             +4.17
#
# WHAT IS VARIED: --align-grip. It was REFUTED for bread
# (Results/wrist_alignment_negative.txt: barely moved corner prevalence, cost
# flat-grip quality) -- but that was measured on a near-cube, where yaw does
# not affect whether a grasp is possible at all. Cereal is 30x100x150 mm: two
# of its dimensions exceed the ~80 mm jaws, so exactly ONE approach axis
# works. The bread refutation does not transfer, and this is the object where
# the mechanism it targets is decisive rather than inert.
#
# WARM START from the bread actor, --warm-start-actor-only. I first ran this
# cold, reasoning that the bread policy sits in the basin that ignores yaw --
# the one thing cereal needs. A probe settled it: at episode 50 the warm run
# had 4 successes and 8% grasp rate, while the cold runs were still at 0% with
# 50/50 t_no_reach at episode 2100. REACHING is object-independent and the
# bread actor supplies it free; the yaw concern applies to the grasp, not the
# approach. Cold seeds 0 and 1 are kept as a control, because warm start could
# still converge fast and plateau lower -- that is the open question.
#
# (previous reasoning, wrong:)
# yaw diagnostic shows it never learned to align the jaws -- the single thing
# cereal requires. Warm-starting would begin in the basin that ignores yaw.
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home/fahim/Thesis_fahim/venv/bin/python
TR="Decomposed state training/Random spawn model/train_rand.py"
O=Results/cereal; mkdir -p $O
common () {
  echo --algo td3_ln --n-envs 5 --episodes 55000 --spawn-level 2.0 \
    --updates-per-step 2 --batch-size 1024 --buffer-size 2000000 --warmup 10000 \
    --lr-actor 3e-4 --lr-critic 3e-4 --critic-reset-every 25000 \
    --target-success 1.01 --builtin-reward --require-lift --best-window 200 \
    --best-margin 0.01 --probe-every 25 --actor-wclip 8 --actor-fakequant \
    --actor-fc1 64 --actor-fc2 32 --fc1 512 --fc2 256 --object-type cereal --cold-start
}
for sd in 0 1 2 3; do
  setsid nohup $PY "$TR" $(common) --seed $sd --tag cereal_base \
    > $O/train_base_s${sd}.log 2>&1 < /dev/null &
  setsid nohup $PY "$TR" $(common) --seed $sd --tag cereal_align --align-grip \
    > $O/train_align_s${sd}.log 2>&1 < /dev/null &
done
disown -a
echo "launched 8 cereal runs: base vs --align-grip, 4 seeds each"
