#!/usr/bin/env python3
"""FP32 host-side replica of the DEPLOYED FSM (pick_and_place_INT8_FSM.ino).

Validates the FSM that actually ships, not the Python training/eval wrapper.
The two are NOT equivalent -- most importantly the sketch has NO transport-stall
early termination, where place_env_wrapper kills a stalled carry after 50 steps
without new-best progress. transport_stall is the dominant remaining failure, so
that difference alone can move the headline number.

Phase logic, constants and ordering are transcribed from the .ino; the
respawn-and-retry loop and success/grasp flags are transcribed from hil_main.py.
Only the serial round-trip is removed -- the two FP32 actors run in-process.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Decomposed state training"))
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks import ActorNetwork

# --- constants, verbatim from pick_and_place_INT8_FSM.ino -------------------
OBJ_X, OBJ_Y, OBJ_Z = 0, 1, 2
# Object centre in the GRIPPER frame, straight out of the observation vector
# (verified identical to R_eef^T (obj_pos - eef_pos)). Available on-device.
OFF_X, OFF_Y, OFF_Z = 7, 8, 9
# Bread's target bin. PickPlace gives every object its OWN bin, so these are
# overwritten from the environment in main() -- hardcoding them would carry
# cereal, can and milk to bread's bin and score every episode a failure.
BIN_X, BIN_Y, BIN_Z = 0.1975, 0.1575, 0.80
# Rule-layer values re-tuned for RANDOM spawn; these MIRROR
# pick_and_place_INT8_FSM.ino. Keep the two in sync -- a default here that does
# not match the sketch reintroduces exactly the harness-vs-artifact gap that
# Sec 9.11 was written to close.
NEAR_TARGET_XY, RELEASE_TRIG_HOLD = 0.18, 3
PLACE_HORIZON, GRASP_HOLD, GRASP_CAP = 300, 8, 250
TRANSLATE_SCALE = 0.65
TL_STEPS, TL_DZ, TL_MIN_RISE = 20, 0.5, 0.03
CARRY_GAIN, CARRY_CLIP = 6.0, 0.5
# Fix C is ADOPTED, so 60 is the default rather than a flag. Every eval in
# Results/ passed --rc-steps 60 explicitly, so no measured number changes.
RC_STEPS, RC_TOL = 60, 0.03
# Height to carry at, for --scripted-transport. Bin rim is BIN_Z=0.80; this
# clears it without the 1.1-2.0 m the open-loop lift produces.
CARRY_Z = 0.95
# Common staging pose for --waypoint-transport. The pick region sits at
# y ~ -0.175 and the target bins at y +0.158..+0.402, so this sits between
# them at a height clear of the bin structure.
WP_X, WP_Y, WP_Z, WP_TOL = 0.10, 0.00, 1.20, 0.05
WP_LEG_CAP = 40          # max steps per leg -- a leg must be able to give up
DS_STEPS, DS_DZ, TOUCH_MARGIN = 30, -0.12, 0.02
OP_STEPS, RT_STEPS, RT_DZ = 8, 12, 0.3
MAX_GRASP_ATTEMPTS = 8            # hil_main.py
# Fix A -- lost-grip recovery in TRANSPORT. Measured (Results/stall_diag):
# 3.33% FP32 / 10.50% INT8 of episodes drop the object a median 9% into the
# carry, then fly an empty gripper for the full PLACE_HORIZON because the
# release trigger requires `grasped`. TEST_LIFT already has this branch;
# TRANSPORT did not. Off by default so the fix is measured, not assumed.
LOST_GRIP_STEPS, MAX_REGRASP = 5, 2
# Fix B -- unjam a blocked carry. Measured (Results/phase2): 7.33% of INT8
# episodes end with joint_margin ~= 0.000 (AT a joint limit, joint 5 in 76/88
# cases) at eef_z 1.53, still holding the object, commanding full scale, moving
# nothing. Successes sit at joint_margin 0.432. A carry-height clamp was
# REFUTED -- successes reach z 1.67 -- so this detects the jam directly, by
# object displacement, and briefly descends to change the arm configuration.
UNJAM_WIN, UNJAM_EPS, UNJAM_STEPS, UNJAM_DZ, MAX_UNJAM = 12, 0.006, 15, -0.6, 3
# Probe (Results/fixB_probe): the detector fires on 9.3% of episodes -- matching
# the 7.33% blocked rate -- but 0/28 recover, and 16/28 exhaust MAX_UNJAM. The
# jam is DETECTED correctly; descending is the wrong escape. Blocked carries are
# also more extended than successes (reach_xy 0.313 vs 0.254), and TRANSPORT
# discards every rotational command (a[3:6] = 0), which is the only actuation
# that can move a pinned wrist joint. UNJAM_MODE tests the alternatives.
UNJAM_MODE = "descend"          # descend | retract | rotate
KEEP_ROTATION = 0               # 1 = do not zero a[3:6] during TRANSPORT
# ROT_ANCHOR_EPS -- the value TRANSPORT writes into a[3:6] instead of exactly
# 0.0.  This is NOT a rotation command; 1e-6 scales to ~5e-7 rad and moves
# nothing.  It exists to flip a branch in robosuite's controller:
#     osc.py:259   bools = [0.0 if math.isclose(e, 0.0) else 1.0 for e in d[3:]]
#                  if sum(bools) > 0.0 or set_ori is not None:
#                      self.goal_ori = set_goal_orientation(...)
#                  self.goal_pos = set_goal_position(...)   # UNCONDITIONAL
# Position re-anchors to the measured eef every step; orientation only updates
# when the rotational delta is nonzero.  So a[3:6] = 0.0 does NOT mean "no
# orientation command" -- it freezes goal_ori at whatever absolute WORLD
# attitude was last commanded (back in GRASP), and the PD then fights to hold
# that attitude across the whole 0.65 m traverse from pick bin to place bin.
# Panda joint 5 has a one-sided range [-0.02, 3.75] (every other joint is
# +-2.9) and absorbs the arc until it saturates: stalled carries sit at
# q5 = 3.753 against the 3.75 stop, commanding full scale and achieving 4.6%
# of it.  Nothing in the loop bounds joint angles -- OSC takes Cartesian
# deltas and nullspace_torques never reads jnt_range -- so the arm jams
# silently and the FSM reads the stall as convergence.
# MEASURED, 12 seeds x 100 per cell, paired (Results/orientation_anchor.txt):
#   cereal + waypoint transport   57.4% -> 82.3%   +24.92  t(11)=+24.3  12u/0d
#   bread  + place actor (main)   88.7% -> 91.4%   + 2.75  t(11)= +4.75 11u/1d
# The bread gain matches the 3.08% "blocked at a joint limit" class that
# Results/transport_stall_diagnosis.txt Part 3 isolated and correctly found
# unfixable from the rule layer -- three escape maneuvers recovered 0/88.  It
# was never a rule-layer defect; it was the controller interface.
# Set to 0.0 to reproduce every number recorded before this was found.
# NOTE: distinct from KEEP_ROTATION=1, which passes the place policy's LARGE
# rotational outputs through and costs -52.33 points.  That stays rejected.
ROT_ANCHOR_EPS = 1e-6
# Fix D -- pose gate at handoff. Results/handoff_carry: the object's pose in the
# gripper predicts whether the carry survives (AUC 0.915 FP32, 0.826 INT8), and
# the INT8 pose shift accounts for 57% of its excess drop rate. A durable grip
# is seated deep and level; a drop-prone one is shifted back (off_x) and riding
# high (off_z). Finger opening carries NO signal, which is why _check_grasp and
# the 3 cm lift certification both pass these. Rejecting a bad pose requires
# actually setting the object down -- the policy is deterministic, so re-lifting
# without releasing reproduces the identical pose.
POSE_OFF_X_MIN, POSE_OFF_Z_MAX = -0.020, 0.020
MAX_POSE_REJECT, REGRIP_DOWN, REGRIP_OPEN = 2, 8, 8

GRASP, TEST_LIFT, TRANSPORT, RECENTER, DESCEND, OPEN, RETRACT, OK, FAIL = range(9)
NAMES = ["GRASP", "TEST_LIFT", "TRANSPORT", "RECENTER", "DESCEND",
         "OPEN", "RETRACT", "DONE_OK", "DONE_FAIL"]


def rests_in_bin(raw, settle_steps):
    """Step physics `settle_steps` times, then report whether the object has
    come to REST inside its target compartment.

    Uses robosuite's own xy bounds (pick_place.not_in_bin) so this differs from
    _check_success in the z test alone.  robosuite requires
    bin_z < obj_z < bin_z + 0.1; a cereal box lying flat rests at exactly 0.90,
    the exclusive upper bound, so it scores placed only if it settles a few mm
    low or is sampled mid-bounce.  Measured: 60 of 62 scored failures come to
    rest inside the correct compartment (Results/orientation_anchor.txt)."""
    for _ in range(settle_steps):
        raw.sim.step()
    bid = raw.obj_body_id[raw.objects[raw.object_id].name]
    pos = raw.sim.data.body_xpos[bid]
    b = raw.object_id
    x_lo = raw.bin2_pos[0] - (raw.bin_size[0] / 2 if b in (0, 2) else 0.0)
    y_lo = raw.bin2_pos[1] - (raw.bin_size[1] / 2 if b < 2 else 0.0)
    in_xy = (x_lo < pos[0] < x_lo + raw.bin_size[0] / 2
             and y_lo < pos[1] < y_lo + raw.bin_size[1] / 2)
    # resting IN the bin: above the floor, below the rim + a box height.
    in_z = raw.bin2_pos[2] < pos[2] < raw.bin2_pos[2] + 0.25
    return bool(in_xy and in_z), float(pos[2])


def placed_now(raw, criterion):
    """Placement test used for the OK transition.

    "robosuite" is PickPlace._check_success verbatim -- reproduces every number
    recorded before bin_success.py existed.  "object-aware" widens only the z
    upper bound, and only for objects tall enough to need it: bread and can are
    bit-identical, cereal's resting centre (0.900) sits exactly on robosuite's
    exclusive upper bound and is otherwise unscoreable."""
    if criterion == "robosuite":
        return bool(raw._check_success())
    from bin_success import check_success
    return check_success(raw)


def load_actor(d, fc1=64, fc2=32):
    """fc1/fc2 default to the deployed 64/32. A Net2WiderNet actor
    (Results/widen) is 128/64 while its critics stay 64/32, so the width has
    to be passed in -- inferring it from the file would silently accept a
    mismatched checkpoint."""
    a = ActorNetwork(46, fc1, fc2, 7, chkpt_dir=d)
    sd = T.load(os.path.join(d, "actor_td3"), map_location="cpu")
    a.load_state_dict({k: v for k, v in sd.items() if not k.startswith("log_std")})
    a.to(T.device("cpu")); a.device = T.device("cpu"); a.eval()
    return a


class FSM:
    def __init__(self):
        self.phase = GRASP
        self.grasp_hold = self.grasp_steps = self.tl_steps = 0
        self.tl_base_z = 0.0
        self.tr_steps = self.over_bin = self.ph_steps = 0
        self.prev_z = 1e9
        self.lost = self.regrasps = 0
        self.scripted_transport = False
        self.carry_ceiling = 0.0
        self.carry_stage = 0
        self.leg_steps = 0
        self.leg_prev_z = 1e9
        self.regrasp_enabled = False
        self.jam_buf = []
        self.unjams = self.unjam_left = 0
        self.unjam_enabled = False
        self.pose_rejects = self.regrip_left = 0
        self.pose_gate_enabled = False
        # staging instrumentation: how leg 1 handed over, and where the
        # object actually was when it did.
        self.stage_exit = ""
        self.stage_pos = (0.0, 0.0, 0.0)

    def _pose_ok(self, s):
        """Is the object seated well enough to survive a 300-step carry?"""
        return (s[OFF_X] >= POSE_OFF_X_MIN) and (s[OFF_Z] <= POSE_OFF_Z_MAX)

    def _escape(self, a, s):
        """Free a pinned arm. Which direction works is an empirical question:
        `descend` was measured to recover 0/28."""
        a[0:6] = 0.0; a[6] = 1.0
        if UNJAM_MODE == "descend":
            a[2] = UNJAM_DZ
        elif UNJAM_MODE == "retract":
            # pull in toward the base and down -- blocked carries are both
            # higher and more extended than successful ones
            v = np.array([s[OBJ_X], s[OBJ_Y]], dtype=np.float64)
            nrm = float(np.linalg.norm(v))
            if nrm > 1e-6:
                a[0:2] = -0.5 * v / nrm
            a[2] = -0.3
        elif UNJAM_MODE == "rotate":
            # the only actuation that can move a pinned wrist joint
            a[4] = 0.5
            a[2] = -0.3
        return a

    def p_xy_to_bin(self, s, a):
        a[0] = np.clip(CARRY_GAIN * (BIN_X - s[OBJ_X]), -CARRY_CLIP, CARRY_CLIP)
        a[1] = np.clip(CARRY_GAIN * (BIN_Y - s[OBJ_Y]), -CARRY_CLIP, CARRY_CLIP)

    # Ablation of the ORIENTATION half of what perception supplies. Of the 14
    # object dims in the 46-D observation, 6 come from the object's position
    # (obj_pos, obj_to_eef_pos) and 8 from its orientation (obj_quat at 3:7,
    # obj_to_eef_quat at 10:14). Replacing the orientation dims with a
    # constant identity quaternion asks whether the policy needs perception to
    # estimate orientation at all, or only position.
    ablate_quat = None        # None | "obj" | "both"

    def _ablate(self, s):
        if not self.ablate_quat:
            return s
        s = np.array(s, dtype=np.float32, copy=True)
        s[3:7] = (0.0, 0.0, 0.0, 1.0)                  # obj_quat -> identity
        if self.ablate_quat == "both":
            s[10:14] = (0.0, 0.0, 0.0, 1.0)            # obj_to_eef_quat
        return s

    def step(self, s, grasped, placed, grasp_actor, place_actor):
        s = self._ablate(s)
        a = np.zeros(7, dtype=np.float32)
        p = self.phase
        if p == GRASP:
            if self.regrip_left > 0:
                # Put the object back down and open, so the re-pick starts from
                # a different state instead of reproducing the rejected pose.
                a[:] = 0.0
                if self.regrip_left > REGRIP_OPEN:
                    a[2], a[6] = -0.4, 1.0
                else:
                    a[6] = -1.0
                self.regrip_left -= 1
                self.grasp_steps += 1
                if self.grasp_steps >= GRASP_CAP:
                    self.phase = FAIL
                return np.clip(a, -1.0, 1.0)
            with T.no_grad():
                a[:] = grasp_actor(T.tensor(s, dtype=T.float).unsqueeze(0)).squeeze(0).numpy()
            self.grasp_steps += 1
            if grasped:
                self.grasp_hold += 1
                if self.grasp_hold >= GRASP_HOLD:
                    self.phase, self.tl_steps, self.tl_base_z = TEST_LIFT, 0, s[OBJ_Z]
            else:
                self.grasp_hold = 0
            if self.grasp_steps >= GRASP_CAP:
                self.phase = FAIL
        elif p == TEST_LIFT:
            a[2], a[6] = TL_DZ, 1.0
            self.tl_steps += 1
            if not grasped:
                self.phase, self.grasp_hold = GRASP, 0
            elif self.tl_steps >= TL_STEPS:
                if (s[OBJ_Z] - self.tl_base_z) >= TL_MIN_RISE:
                    if (self.pose_gate_enabled and not self._pose_ok(s)
                            and self.pose_rejects < MAX_POSE_REJECT):
                        self.pose_rejects += 1
                        self.phase, self.grasp_hold = GRASP, 0
                        self.regrip_left = REGRIP_DOWN + REGRIP_OPEN
                    else:
                        self.phase, self.tr_steps, self.over_bin = TRANSPORT, 0, 0
                else:
                    self.phase, self.grasp_hold = GRASP, 0
        elif p == TRANSPORT:
            if self.scripted_transport == "waypoint":
                # Canonical three-leg transport:
                #   leg 0  move to ONE common staging pose (WP_X, WP_Y, WP_Z)
                #   leg 1  traverse to above the bin, holding that height
                #   then  hand over to RECENTER/DESCEND (the P releaser)
                # Every episode passes through the same pose, so the path from
                # staging to the bin is identical regardless of where the
                # grasp happened. OSC_POSE handles the IK; these are Cartesian
                # setpoints.
                # leg 0 is LIFT-ONLY when below the staging height. Moving
                # all three axes at once made this a diagonal, and with
                # 5 of 15 episodes entering transport above WP_Z it drove
                # them back DOWN while translating -- the same mistake as the
                # first staged attempt. WP_Z is a floor here, never a target.
                # EVERY leg needs a timeout and a stall check. Without them
                # leg 0 deadlocks: it lifts until z >= WP_Z, but at the edge
                # of the workspace (x ~ -0.3, y ~ -0.2) the arm CANNOT reach
                # 1.25 m, so it climbs forever and no amount of horizon helps.
                # That is why doubling PLACE_HORIZON moved this controller by
                # exactly 0.0 points. A scripted leg must always be able to
                # give up.
                dz = WP_Z - s[OBJ_Z]
                self.leg_steps += 1
                stalled = abs(s[OBJ_Z] - self.leg_prev_z) < 1e-4
                self.leg_prev_z = s[OBJ_Z]
                if self.carry_stage == 0:
                    if dz > 0.02 and self.leg_steps < WP_LEG_CAP and not stalled:
                        a[0] = a[1] = 0.0
                        a[2] = np.clip(CARRY_GAIN * dz, 0.0, CARRY_CLIP)
                    else:
                        self.carry_stage = 1; self.leg_steps = 0
                elif self.carry_stage == 1:
                    d = np.array([WP_X - s[OBJ_X], WP_Y - s[OBJ_Y]])
                    a[0:2] = np.clip(CARRY_GAIN * d, -CARRY_CLIP, CARRY_CLIP)
                    a[2] = np.clip(CARRY_GAIN * dz, 0.0, CARRY_CLIP)
                    if np.linalg.norm(d) <= WP_TOL or self.leg_steps >= WP_LEG_CAP:
                        self.stage_exit = ("tol" if np.linalg.norm(d) <= WP_TOL
                                           else "timeout")
                        self.stage_pos = (float(s[OBJ_X]), float(s[OBJ_Y]),
                                          float(s[OBJ_Z]))
                        self.carry_stage = 2; self.leg_steps = 0
                else:
                    self.p_xy_to_bin(s, a)
                    a[2] = np.clip(CARRY_GAIN * dz, 0.0, CARRY_CLIP)
                a[3:6] = ROT_ANCHOR_EPS
            elif self.scripted_transport == "staged":
                # Standard pick-and-place waypoint pattern, which the first
                # scripted attempt did NOT do: it drove x, y and z at once so
                # the object descended WHILE translating and swept diagonally
                # through the bin region.
                #   stage 0  lift straight up to CARRY_Z, no horizontal motion
                #   stage 1  traverse at constant height until over the bin
                #   then hand over to RECENTER/DESCEND as usual
                # CARRY_Z is a FLOOR, not a target. TEST_LIFT already leaves
                # the object at 1.3-2.0 m, so driving z toward CARRY_Z made
                # stage 0 DESCEND before traversing -- giving up the altitude
                # that the four earlier experiments showed is protective.
                # Climb only if below; never trade height for nothing.
                dz = CARRY_Z - s[OBJ_Z]
                if self.carry_stage == 0:
                    if dz <= 0.02:
                        self.carry_stage = 1          # already high enough
                    else:
                        a[0] = a[1] = 0.0
                        a[2] = np.clip(CARRY_GAIN * dz, 0.0, CARRY_CLIP)
                if self.carry_stage == 1:
                    self.p_xy_to_bin(s, a)
                    a[2] = np.clip(CARRY_GAIN * dz, 0.0, CARRY_CLIP)
                a[3:6] = ROT_ANCHOR_EPS
            elif self.scripted_transport:
                # P control straight to the bin, plus a HEIGHT TARGET. The
                # learned place actor was trained on bread; on cereal it
                # carries the box for the full 300-step horizon without ever
                # closing to NEAR_TARGET_XY (measured: min_xy 0.196-0.433
                # against a 0.18 threshold, grip intact throughout). It also
                # lifts to 1.1-2.0 m for a bin at 0.80, because TEST_LIFT is
                # open-loop (+0.5 for 20 steps) and nothing commands z back
                # down until DESCEND.
                self.p_xy_to_bin(s, a)
                a[2] = np.clip(CARRY_GAIN * (CARRY_Z - s[OBJ_Z]),
                               -CARRY_CLIP, CARRY_CLIP)
                a[3:6] = ROT_ANCHOR_EPS
            else:
                with T.no_grad():
                    a[:] = place_actor(T.tensor(s, dtype=T.float).unsqueeze(0)).squeeze(0).numpy()
                if not KEEP_ROTATION:
                    a[3:6] = ROT_ANCHOR_EPS
                a[0:3] *= TRANSLATE_SCALE
                # Optional ceiling: the actor climbs to 1.1-2.0 m for a bin at
                # 0.80. Height is PROTECTIVE below ~1.3 (0.95 -> 29%,
                # 1.30 -> 66%), so this only trims the excursion ABOVE the
                # useful range -- it never pushes the object down into the
                # region that fails. Bounds the actor rather than replacing
                # it, so whatever path shape it learned is preserved.
                if self.carry_ceiling and s[OBJ_Z] > self.carry_ceiling:
                    a[2] = min(a[2], 0.0)
            a[6] = 1.0
            self.tr_steps += 1
            if self.unjam_enabled:
                self.jam_buf.append((float(s[OBJ_X]), float(s[OBJ_Y]), float(s[OBJ_Z])))
                if len(self.jam_buf) > UNJAM_WIN:
                    self.jam_buf.pop(0)
                if self.unjam_left > 0:
                    self._escape(a, s); self.unjam_left -= 1
                elif (grasped and len(self.jam_buf) == UNJAM_WIN
                        and self.unjams < MAX_UNJAM
                        and np.linalg.norm(np.array(self.jam_buf[-1])
                                           - np.array(self.jam_buf[0])) < UNJAM_EPS):
                    # Object has not moved for UNJAM_WIN steps while held:
                    # the arm is pinned. Descend to change configuration.
                    self.unjams += 1
                    self.unjam_left = UNJAM_STEPS - 1
                    self.jam_buf = []
                    self._escape(a, s)
            over = np.hypot(s[OBJ_X] - BIN_X, s[OBJ_Y] - BIN_Y) <= NEAR_TARGET_XY
            self.over_bin = self.over_bin + 1 if over else 0
            self.lost = self.lost + 1 if not grasped else 0
            if grasped and self.over_bin >= RELEASE_TRIG_HOLD:
                self.phase, self.ph_steps = RECENTER, 0
            elif (self.regrasp_enabled and self.lost >= LOST_GRIP_STEPS
                    and self.regrasps < MAX_REGRASP):
                # Object is gone. Re-attempt the pick instead of burning the
                # horizon; bounded by MAX_REGRASP and the episode horizon.
                self.regrasps += 1
                self.phase, self.grasp_hold, self.grasp_steps = GRASP, 0, 0
                self.lost = 0
            elif self.tr_steps >= PLACE_HORIZON:
                self.phase = FAIL
        elif p == RECENTER:
            self.p_xy_to_bin(s, a); a[6] = 1.0
            self.ph_steps += 1
            if (np.hypot(s[OBJ_X] - BIN_X, s[OBJ_Y] - BIN_Y) <= RC_TOL
                    or self.ph_steps >= RC_STEPS or not grasped):
                self.phase, self.ph_steps, self.prev_z = DESCEND, 0, 1e9
        elif p == DESCEND:
            self.p_xy_to_bin(s, a); a[2], a[6] = DS_DZ, 1.0
            self.ph_steps += 1
            touched = s[OBJ_Z] <= BIN_Z + TOUCH_MARGIN
            stalled = s[OBJ_Z] >= self.prev_z - 1e-4
            self.prev_z = s[OBJ_Z]
            if (not grasped) or touched or stalled or self.ph_steps >= DS_STEPS:
                self.phase, self.ph_steps = OPEN, 0
        elif p == OPEN:
            a[6] = -1.0
            self.ph_steps += 1
            if self.ph_steps >= OP_STEPS:
                self.phase, self.ph_steps = RETRACT, 0
        elif p == RETRACT:
            a[2], a[6] = RT_DZ, -1.0
            self.ph_steps += 1
            if placed:
                self.phase = OK
            elif self.ph_steps >= RT_STEPS:
                self.phase = FAIL
        else:
            a[6] = -1.0
        return np.clip(a, -1.0, 1.0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp-ckpt", required=True)
    p.add_argument("--place-ckpt", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--carry-ceiling", type=float, default=0.0,
                   help="stop the place actor commanding further ascent once "
                        "the object is above this height. 0 = off. Bounds the "
                        "actor without replacing it.")
    p.add_argument("--carry-z", type=float, default=None,
                   help="height target for --scripted-transport (default 0.95)")
    p.add_argument("--waypoint-transport", action="store_true",
                   help="three-leg transport through ONE common staging pose: "
                        "move to (WP_X,WP_Y,WP_Z), traverse to above the bin "
                        "at that height, then hand to the P releaser.")
    p.add_argument("--wp", default=None,
                   help="override the staging pose as x,y,z")
    p.add_argument("--staged-transport", action="store_true",
                   help="scripted transport as a WAYPOINT sequence: lift "
                        "vertically to carry-z, traverse at constant height, "
                        "then descend. The standard pattern; the flat "
                        "--scripted-transport drives all three axes at once.")
    p.add_argument("--scripted-transport", action="store_true",
                   help="carry with P control to the bin plus a height "
                        "target, instead of the learned place actor. The "
                        "place actor is bread-trained and does not transfer.")
    p.add_argument("--video", default=None,
                   help="directory to write one MP4 per episode. Uses the "
                        "offscreen renderer, so it costs time -- for "
                        "diagnosis, not for scoring runs.")
    p.add_argument("--video-cam", default="frontview",
                   help="camera for --video (frontview, agentview, "
                        "birdview, robot0_eye_in_hand)")
    p.add_argument("--video-failures-only", action="store_true",
                   help="keep only episodes that FAIL -- what you want when "
                        "diagnosing why the task breaks after a good grasp")
    p.add_argument("--object-type", default="bread",
                   choices=["bread", "cereal", "can", "milk"],
                   help="the policies were trained on bread; the others are a "
                        "zero-shot generalisation test. The target bin differs "
                        "per object and is read from the env, not hardcoded.")
    p.add_argument("--ablate-quat", default=None, choices=["obj","both"],

                   help="replace the orientation dims perception supplies with an "

                        "identity quaternion: obj = obj_quat only, both = also "

                        "obj_to_eef_quat")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fixed-spawn", action="store_true",
                   help="Match hil_main.py's fixed pose. Default is native "
                        "random spawn (position + rotation), the condition the "
                        "current pipeline is measured under.")
    p.add_argument("--out", required=True)
    # --- FSM rule-layer parameters -------------------------------------------
    # Sec 5 of thesis_context: FSM parameters alone took end-to-end 78% -> 92%
    # at FIXED spawn, with identical weights. These values were tuned for that
    # regime and have never been re-tuned for random spawn, where the dominant
    # failure is now a transport horizon-out (7.5%) rather than anything the
    # policies control.
    p.add_argument("--near-target-xy", type=float, default=NEAR_TARGET_XY)
    p.add_argument("--release-trig-hold", type=int, default=RELEASE_TRIG_HOLD)
    p.add_argument("--place-horizon", type=int, default=PLACE_HORIZON)
    p.add_argument("--translate-scale", type=float, default=TRANSLATE_SCALE)
    p.add_argument("--carry-gain", type=float, default=CARRY_GAIN)
    p.add_argument("--carry-clip", type=float, default=CARRY_CLIP)
    p.add_argument("--rc-steps", type=int, default=RC_STEPS)
    p.add_argument("--rc-tol", type=float, default=RC_TOL)
    p.add_argument("--ds-steps", type=int, default=DS_STEPS)
    p.add_argument("--ds-dz", type=float, default=DS_DZ)
    p.add_argument("--touch-margin", type=float, default=TOUCH_MARGIN)
    p.add_argument("--rt-steps", type=int, default=RT_STEPS)
    p.add_argument("--rot-anchor-eps", type=float, default=ROT_ANCHOR_EPS,
                   help="value written to a[3:6] during TRANSPORT. 0.0 freezes "
                        "goal_ori (pre-fix behaviour); 1e-6 re-anchors it.")
    p.add_argument("--success-criterion", choices=("robosuite", "object-aware"),
                   default="robosuite",
                   help="'robosuite' reproduces every previously recorded "
                        "number and is bit-identical to 'object-aware' for "
                        "bread and can. Use 'object-aware' for cereal and "
                        "milk, whose resting height sits at or above "
                        "robosuite's constant z bound (see bin_success.py).")
    p.add_argument("--settle-steps", type=int, default=0,
                   help="after each episode step physics this many times and "
                        "record whether the object came to REST inside the "
                        "target bin compartment. robosuite needs "
                        "bin_z < obj_z < bin_z+0.1, and a cereal box resting "
                        "flat sits at exactly 0.90 = the exclusive upper "
                        "bound. 0 = off.")
    p.add_argument("--rt-dz", type=float, default=RT_DZ)
    p.add_argument("--grasp-cap", type=int, default=GRASP_CAP)
    p.add_argument("--regrasp", action="store_true",
                   help="Fix A: recover from a lost grip during TRANSPORT by "
                        "returning to GRASP, instead of carrying nothing to "
                        "the horizon.")
    p.add_argument("--lost-grip-steps", type=int, default=LOST_GRIP_STEPS)
    p.add_argument("--max-regrasp", type=int, default=MAX_REGRASP)
    p.add_argument("--unjam", action="store_true",
                   help="Fix B: detect a pinned carry by object displacement "
                        "and descend briefly to free the arm.")
    p.add_argument("--unjam-win", type=int, default=UNJAM_WIN)
    p.add_argument("--unjam-eps", type=float, default=UNJAM_EPS)
    p.add_argument("--unjam-steps", type=int, default=UNJAM_STEPS)
    p.add_argument("--unjam-dz", type=float, default=UNJAM_DZ)
    p.add_argument("--max-unjam", type=int, default=MAX_UNJAM)
    p.add_argument("--unjam-mode", choices=["descend","retract","rotate"],
                   default=UNJAM_MODE)
    p.add_argument("--pose-gate", action="store_true",
                   help="Fix D: reject a handoff whose grip pose predicts a "
                        "mid-carry drop, set the object down and re-pick.")
    p.add_argument("--pose-off-x-min", type=float, default=POSE_OFF_X_MIN)
    p.add_argument("--pose-off-z-max", type=float, default=POSE_OFF_Z_MAX)
    p.add_argument("--max-pose-reject", type=int, default=MAX_POSE_REJECT)
    p.add_argument("--keep-rotation", type=int, default=KEEP_ROTATION,
                   help="1 = pass the transport policy's rotational commands "
                        "through instead of zeroing them. a[3:6]=0 is a "
                        "fixed-spawn-era decision never revisited for random "
                        "spawn, where the wrist arrives at varied yaw.")
    p.add_argument("--tag", default="", help="label carried into the CSV")
    p.add_argument("--int8", action="store_true",
                   help="Run both actors as INT8 .tflite interpreters instead "
                        "of FP32 PyTorch -- the acceptance test for the "
                        "deployed artifacts. Output diff is NOT the test: a "
                        "full-scale sign flip on one action dim was seen at "
                        "corr 0.94 while behaviour survived.")
    p.add_argument("--grasp-fc1", type=int, default=64,
                   help="Grasp actor width (64 = deployed; 128 for a widened one)")
    p.add_argument("--grasp-fc2", type=int, default=32)
    p.add_argument("--grasp-tflite", default=None)
    p.add_argument("--place-tflite", default=None)
    p.add_argument("--dump-transport-states", default=None,
                   help="Save states seen by the place actor (TRANSPORT phase) "
                        "as an npz for INT8 calibration. The place model never "
                        "sees reach/grasp states, so the grasp buffer is the "
                        "wrong calibration distribution for it.")
    a = p.parse_args()

    # Apply rule-layer overrides to the module constants the FSM reads.
    g = globals()
    if a.wp:
        _v = [float(x) for x in a.wp.split(",")]
        globals()["WP_X"], globals()["WP_Y"], globals()["WP_Z"] = _v
    if a.carry_z is not None:
        globals()["CARRY_Z"] = a.carry_z
    for k in ("NEAR_TARGET_XY", "RELEASE_TRIG_HOLD", "PLACE_HORIZON",
              "TRANSLATE_SCALE", "CARRY_GAIN", "CARRY_CLIP", "RC_STEPS",
              "RC_TOL", "DS_STEPS", "DS_DZ", "TOUCH_MARGIN", "RT_STEPS",
              "RT_DZ", "GRASP_CAP", "LOST_GRIP_STEPS", "MAX_REGRASP",
              "UNJAM_WIN", "UNJAM_EPS", "UNJAM_STEPS", "UNJAM_DZ", "MAX_UNJAM",
              "UNJAM_MODE", "KEEP_ROTATION", "ROT_ANCHOR_EPS",
              "POSE_OFF_X_MIN",
              "POSE_OFF_Z_MAX", "MAX_POSE_REJECT"):
        g[k] = getattr(a, k.lower())

    np.random.seed(a.seed); T.manual_seed(a.seed)
    raw = suite.make("PickPlace", robots="Panda",
                     controller_configs=suite.load_controller_config(
                         default_controller="OSC_POSE"),
                     has_renderer=False,
                     has_offscreen_renderer=bool(a.video),
                     use_camera_obs=False, horizon=700, reward_shaping=True,
                     control_freq=20, single_object_mode=2,
                     object_type=a.object_type,
                     camera_names=[a.video_cam] if a.video else None,
                     camera_heights=480, camera_widths=640)
    raw.reset()
    global BIN_X, BIN_Y, BIN_Z
    _tb = raw.target_bin_placements[raw.object_id]
    BIN_X, BIN_Y, BIN_Z = float(_tb[0]), float(_tb[1]), float(_tb[2])
    print(f"object {a.object_type!r} (id {raw.object_id}) -> bin "
          f"({BIN_X:.4f}, {BIN_Y:.4f}, {BIN_Z:.4f})", flush=True)
    if a.fixed_spawn:
        _orig = raw._get_placement_initializer

        def _fixed():
            _orig()
            s = raw.placement_initializer.samplers["CollisionObjectSampler"]
            s.x_range = np.array([0.0, 0.0]); s.y_range = np.array([0.0, 0.0])
            s.rotation = 0.0
            s.ensure_object_boundary_in_range = False
            s.ensure_valid_placement = False
        raw._get_placement_initializer = _fixed
    env = GymWrapper(raw)
    env.seed(a.seed)
    lo, hi = env.action_space.low, env.action_space.high

    if a.int8:
        import tensorflow as tf

        class TFLiteActor:
            """Mirrors the sketch's runModel(): quantize input, invoke,
            dequantize output. Same int8 path the ESP32 executes."""

            def __init__(self, path):
                self.it = tf.lite.Interpreter(model_path=path)
                self.it.allocate_tensors()
                self.inp = self.it.get_input_details()[0]
                self.out = self.it.get_output_details()[0]

            def __call__(self, x):
                v = x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)
                v = v.reshape(1, -1).astype(np.float32)
                sc, zp = self.inp["quantization"]
                if self.inp["dtype"] == np.int8:
                    v = np.clip(np.round(v / sc + zp), -128, 127).astype(np.int8)
                self.it.set_tensor(self.inp["index"], v)
                self.it.invoke()
                o = self.it.get_tensor(self.out["index"])
                sc, zp = self.out["quantization"]
                if self.out["dtype"] == np.int8:
                    o = (o.astype(np.float32) - zp) * sc
                return T.tensor(o.astype(np.float32))

        # Mixed precision is allowed on purpose: swapping ONE stage to INT8
        # isolates which model loses the points. FP32 first-try handoff is 98%
        # and INT8 is 46%, so the handoff (grasp + test-lift) is the suspect.
        g_actor = (TFLiteActor(a.grasp_tflite) if a.grasp_tflite
                   else load_actor(a.grasp_ckpt, a.grasp_fc1, a.grasp_fc2))
        p_actor = (TFLiteActor(a.place_tflite) if a.place_tflite
                   else load_actor(a.place_ckpt))
    else:
        g_actor = load_actor(a.grasp_ckpt, a.grasp_fc1, a.grasp_fc2)
        p_actor = load_actor(a.place_ckpt)

    def flags():
        gr = pl = False
        try:
            gr = bool(raw._check_grasp(gripper=raw.robots[0].gripper,
                                       object_geoms=raw.objects[raw.object_id]))
        except Exception:
            pass
        try:
            pl = placed_now(raw, a.success_criterion)
        except Exception:
            pass
        return gr, pl

    rows = []
    tstates = []
    for ep in range(a.episodes):
        # --- handoff: respawn-and-retry until TRANSPORT (hil_main.do_handoff)
        obs, attempts, reached = None, 0, False
        frames = [] if a.video else None
        def _grab():
            if frames is None:
                return
            im = raw.sim.render(width=640, height=480, camera_name=a.video_cam)
            frames.append(im[::-1])          # MuJoCo renders bottom-up
        for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
            obs = env.reset(); fsm = FSM(); attempts = attempt
            fsm.regrasp_enabled = a.regrasp
            fsm.unjam_enabled = a.unjam
            fsm.pose_gate_enabled = a.pose_gate
            fsm.ablate_quat = a.ablate_quat
            fsm.scripted_transport = ("waypoint" if a.waypoint_transport
                                      else "staged" if a.staged_transport
                                      else a.scripted_transport)
            fsm.carry_ceiling = a.carry_ceiling
            for _ in range(GRASP_CAP + TL_STEPS + 5):
                gr, pl = flags()
                act = fsm.step(np.asarray(obs, dtype=np.float32), gr, pl,
                               g_actor, p_actor)
                obs, _, done, _ = env.step(np.clip(act, lo, hi))
                _grab()
                if fsm.phase == TRANSPORT:
                    reached = True; break
                if fsm.phase in (OK, FAIL) or done:
                    break
            if reached:
                break
        if not reached:
            rows.append(dict(episode=ep, attempts=attempts, success=0,
                             phase="handoff_failed", steps=0, tag=a.tag,
                             regrasps=fsm.regrasps, unjams=fsm.unjams,
                             pose_rejects=fsm.pose_rejects,
                         stage_exit=fsm.stage_exit,
                         stage_x=round(fsm.stage_pos[0], 4),
                         stage_y=round(fsm.stage_pos[1], 4),
                         stage_z=round(fsm.stage_pos[2], 4)))
            continue
        # --- scored portion
        n = 0
        while n < 700:
            gr, pl = flags()
            if a.dump_transport_states and fsm.phase == TRANSPORT:
                tstates.append(np.asarray(obs, dtype=np.float32).copy())
            act = fsm.step(np.asarray(obs, dtype=np.float32), gr, pl,
                           g_actor, p_actor)
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            _grab()
            n += 1
            if fsm.phase in (OK, FAIL) or done:
                break
        ok = int(fsm.phase == OK)
        rested, rest_z = (None, None)
        if a.settle_steps:
            rested, rest_z = rests_in_bin(raw, a.settle_steps)
        rows.append(dict(episode=ep, attempts=attempts,
                         success=ok,
                         rested=("" if rested is None else int(rested)),
                         rest_z=("" if rest_z is None else round(rest_z, 4)),
                         phase=NAMES[fsm.phase], steps=n, tag=a.tag,
                         regrasps=fsm.regrasps, unjams=fsm.unjams,
                         pose_rejects=fsm.pose_rejects,
                         stage_exit=fsm.stage_exit,
                         stage_x=round(fsm.stage_pos[0], 4),
                         stage_y=round(fsm.stage_pos[1], 4),
                         stage_z=round(fsm.stage_pos[2], 4)))
        if frames and not (a.video_failures_only and ok):
            import cv2, os as _os
            _os.makedirs(a.video, exist_ok=True)
            # Name it with the outcome and the phase it died in, so the
            # failures can be found without opening every file.
            fn = _os.path.join(a.video,
                               f"ep{ep:03d}_{'OK' if ok else 'FAIL'}_"
                               f"{NAMES[fsm.phase]}.mp4")
            vw = cv2.VideoWriter(fn, cv2.VideoWriter_fourcc(*"mp4v"), 20,
                                 (frames[0].shape[1], frames[0].shape[0]))
            for fr in frames:
                vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            vw.release()
            print(f"  wrote {fn} ({len(frames)} frames)", flush=True)
        if (ep + 1) % 25 == 0:
            s = sum(r["success"] for r in rows)
            print(f"  {ep+1}/{a.episodes}  success {s/len(rows)*100:.1f}%", flush=True)

    if a.dump_transport_states and tstates:
        arr = np.asarray(tstates, dtype=np.float32)
        np.savez(a.dump_transport_states, state_memory=arr,
                 mem_cntr=np.array([len(arr)]))
        print(f"wrote {len(arr)} transport states -> {a.dump_transport_states}")

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    s = sum(r["success"] for r in rows)
    print(f"\nFSM (FP32) {s}/{len(rows)} = {s/len(rows)*100:.1f}%   -> {a.out}")
    if a.settle_steps:
        r_ = sum(int(x["rested"]) for x in rows if x["rested"] != "")
        print(f"  came to REST in the target compartment: "
              f"{r_}/{len(rows)} = {r_/len(rows)*100:.1f}%"
              f"   (robosuite scores {s/len(rows)*100:.1f}%; the gap is the "
              f"bin_z+0.1 z-window, not a placement failure)")


if __name__ == "__main__":
    main()
