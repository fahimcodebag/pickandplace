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

Imported by fsm_sim.py (evaluation) AND
"Decomposed state training/place_env_wrapper.py" (the training environment).
Both must use it: a place policy trained with a[3:6]=0.0 learns to cope with a
pinned wrist, and would then be evaluated in an environment that does not pin
it.  That mismatch is why this constant lives in one file.
"""
ROT_ANCHOR_EPS = 1e-6
