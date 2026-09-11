#!/usr/bin/env python3
"""The one value that must be written into a[3:6] instead of 0.0.

robosuite's OSC updates the position goal unconditionally but the orientation
goal ONLY when the rotational delta is nonzero (controllers/osc.py:259):

    bools = [0.0 if math.isclose(e, 0.0) else 1.0 for e in scaled_delta[3:]]
    if sum(bools) > 0.0 or set_ori is not None:
        self.goal_ori = set_goal_orientation(...)      # CONDITIONAL
    self.goal_pos = set_goal_position(...)             # UNCONDITIONAL

So `a[3:6] = 0.0` does not mean "no orientation command".  It FREEZES goal_ori
at whatever absolute world attitude was last commanded, and the orientation PD
then fights to hold that attitude through the whole traverse.  Panda joint 5
has a one-sided range [-0.02, 3.75] and saturates against it: stalled carries
sit at q5 = 3.753, commanding full scale and achieving 4.6% of it.

1e-6 scales to ~5e-7 rad and moves nothing.  Its only job is to flip that
isclose() branch so goal_ori re-anchors to the current attitude each step.

Measured (Results/orientation_anchor.txt, 12 seeds x 100, paired):
    FP32 random spawn   90.67% -> 93.58%   +2.92  t(11)=+5.12
    INT8 random spawn   78.33% -> 83.58%   +5.25  t(11)=+8.19

DISTINCT from passing a policy's real rotational outputs through, which costs
-52 points and remains rejected (thesis_context.md 9.13).

ON THE DEPLOYED c2m512_s1 IT HURTS (Results/orientation_anchor_c2m512.txt):
    FP32 random spawn   94.92% -> 92.17%   -2.75  t(11)=-4.29
    INT8 random spawn   95.33% -> 92.83%   -2.50  t(11)=-3.74
It rescues the rare joint-5 pin (0.42% of episodes) but lets steep grasps drift.

WHO USES THIS VALUE (since 2026-09-11)
  * "Decomposed state training/place_env_wrapper.py" -- the TRAINING environment.
    Unchanged: every place run in Results/place_random_spawn_investigation.txt
    trained with it.
  * fsm_sim.py no longer defaults to it: its evaluation default is 0.0 (anchor
    off), matching the firmware and the deployed artifact.  Evaluating a place
    policy trained with the anchor needs --rot-anchor-eps 1e-6 passed explicitly,
    which eval_place_snapshot_job.sh, eval_cereal_scratch_job.sh and
    run_place_selection.sh do.
"""
ROT_ANCHOR_EPS = 1e-6
