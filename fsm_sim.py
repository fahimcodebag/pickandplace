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
BIN_X, BIN_Y, BIN_Z = 0.1975, 0.1575, 0.80
NEAR_TARGET_XY, RELEASE_TRIG_HOLD = 0.14, 3
PLACE_HORIZON, GRASP_HOLD, GRASP_CAP = 300, 8, 250
TRANSLATE_SCALE = 0.5
TL_STEPS, TL_DZ, TL_MIN_RISE = 20, 0.5, 0.03
CARRY_GAIN, CARRY_CLIP = 4.0, 0.5
RC_STEPS, RC_TOL = 30, 0.03
DS_STEPS, DS_DZ, TOUCH_MARGIN = 30, -0.12, 0.02
OP_STEPS, RT_STEPS, RT_DZ = 8, 12, 0.3
MAX_GRASP_ATTEMPTS = 8            # hil_main.py

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

    def p_xy_to_bin(self, s, a):
        a[0] = np.clip(CARRY_GAIN * (BIN_X - s[OBJ_X]), -CARRY_CLIP, CARRY_CLIP)
        a[1] = np.clip(CARRY_GAIN * (BIN_Y - s[OBJ_Y]), -CARRY_CLIP, CARRY_CLIP)

    def step(self, s, grasped, placed, grasp_actor, place_actor):
        a = np.zeros(7, dtype=np.float32)
        p = self.phase
        if p == GRASP:
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
                    self.phase, self.tr_steps, self.over_bin = TRANSPORT, 0, 0
                else:
                    self.phase, self.grasp_hold = GRASP, 0
        elif p == TRANSPORT:
            with T.no_grad():
                a[:] = place_actor(T.tensor(s, dtype=T.float).unsqueeze(0)).squeeze(0).numpy()
            a[3:6] = 0.0
            a[0:3] *= TRANSLATE_SCALE
            a[6] = 1.0
            self.tr_steps += 1
            over = np.hypot(s[OBJ_X] - BIN_X, s[OBJ_Y] - BIN_Y) <= NEAR_TARGET_XY
            self.over_bin = self.over_bin + 1 if over else 0
            if grasped and self.over_bin >= RELEASE_TRIG_HOLD:
                self.phase, self.ph_steps = RECENTER, 0
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
    a = p.parse_args()

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
    for ep in range(a.episodes):
        # --- handoff: respawn-and-retry until TRANSPORT (hil_main.do_handoff)
        obs, attempts, reached = None, 0, False
        for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
            obs = env.reset(); fsm = FSM(); attempts = attempt
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
                             phase="handoff_failed", steps=0)); continue
        # --- scored portion
        n = 0
        while n < 700:
            gr, pl = flags()
            act = fsm.step(np.asarray(obs, dtype=np.float32), gr, pl,
                           g_actor, p_actor)
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            n += 1
            if fsm.phase in (OK, FAIL) or done:
                break
        rows.append(dict(episode=ep, attempts=attempts,
                         success=int(fsm.phase == OK),
                         phase=NAMES[fsm.phase], steps=n))
        if (ep + 1) % 25 == 0:
            s = sum(r["success"] for r in rows)
            print(f"  {ep+1}/{a.episodes}  success {s/len(rows)*100:.1f}%", flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    s = sum(r["success"] for r in rows)
    print(f"\nFSM (FP32) {s}/{len(rows)} = {s/len(rows)*100:.1f}%   -> {a.out}")


if __name__ == "__main__":
    main()
