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
RC_STEPS, RC_TOL = 30, 0.03
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


def load_actor(d):
    a = ActorNetwork(46, 64, 32, 7, chkpt_dir=d)
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
        self.regrasp_enabled = False
        self.jam_buf = []
        self.unjams = self.unjam_left = 0
        self.unjam_enabled = False
        self.pose_rejects = self.regrip_left = 0
        self.pose_gate_enabled = False

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

    def step(self, s, grasped, placed, grasp_actor, place_actor):
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
            with T.no_grad():
                a[:] = place_actor(T.tensor(s, dtype=T.float).unsqueeze(0)).squeeze(0).numpy()
            if not KEEP_ROTATION:
                a[3:6] = 0.0
            a[0:3] *= TRANSLATE_SCALE
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
    for k in ("NEAR_TARGET_XY", "RELEASE_TRIG_HOLD", "PLACE_HORIZON",
              "TRANSLATE_SCALE", "CARRY_GAIN", "CARRY_CLIP", "RC_STEPS",
              "RC_TOL", "DS_STEPS", "DS_DZ", "TOUCH_MARGIN", "RT_STEPS",
              "RT_DZ", "GRASP_CAP", "LOST_GRIP_STEPS", "MAX_REGRASP",
              "UNJAM_WIN", "UNJAM_EPS", "UNJAM_STEPS", "UNJAM_DZ", "MAX_UNJAM",
              "UNJAM_MODE", "KEEP_ROTATION", "POSE_OFF_X_MIN",
              "POSE_OFF_Z_MAX", "MAX_POSE_REJECT"):
        g[k] = getattr(a, k.lower())

    np.random.seed(a.seed); T.manual_seed(a.seed)
    raw = suite.make("PickPlace", robots="Panda",
                     controller_configs=suite.load_controller_config(
                         default_controller="OSC_POSE"),
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, horizon=700, reward_shaping=True,
                     control_freq=20, single_object_mode=2, object_type="bread")
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
                   else load_actor(a.grasp_ckpt))
        p_actor = (TFLiteActor(a.place_tflite) if a.place_tflite
                   else load_actor(a.place_ckpt))
    else:
        g_actor, p_actor = load_actor(a.grasp_ckpt), load_actor(a.place_ckpt)

    def flags():
        gr = pl = False
        try:
            gr = bool(raw._check_grasp(gripper=raw.robots[0].gripper,
                                       object_geoms=raw.objects[raw.object_id]))
        except Exception:
            pass
        try:
            pl = bool(raw._check_success())
        except Exception:
            pass
        return gr, pl

    rows = []
    tstates = []
    for ep in range(a.episodes):
        # --- handoff: respawn-and-retry until TRANSPORT (hil_main.do_handoff)
        obs, attempts, reached = None, 0, False
        for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
            obs = env.reset(); fsm = FSM(); attempts = attempt
            fsm.regrasp_enabled = a.regrasp
            fsm.unjam_enabled = a.unjam
            fsm.pose_gate_enabled = a.pose_gate
            for _ in range(GRASP_CAP + TL_STEPS + 5):
                gr, pl = flags()
                act = fsm.step(np.asarray(obs, dtype=np.float32), gr, pl,
                               g_actor, p_actor)
                obs, _, done, _ = env.step(np.clip(act, lo, hi))
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
                             pose_rejects=fsm.pose_rejects))
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
            n += 1
            if fsm.phase in (OK, FAIL) or done:
                break
        rows.append(dict(episode=ep, attempts=attempts,
                         success=int(fsm.phase == OK),
                         phase=NAMES[fsm.phase], steps=n, tag=a.tag,
                         regrasps=fsm.regrasps, unjams=fsm.unjams,
                         pose_rejects=fsm.pose_rejects))
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


if __name__ == "__main__":
    main()
