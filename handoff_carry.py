#!/usr/bin/env python3
"""Does the handoff POSE explain the INT8 carry failures?

The rule layer forces a[6] = 1.0 (gripper closed) for every step of TRANSPORT,
so the grasp policy exerts no control over the grip during the carry. INT8
therefore cannot be "holding worse" -- it is not holding at all. If quantization
triples the mid-carry drop rate (3.33% -> 10.50%), the damage must already be
present in the POSE the grasp stage hands over.

This records the object's pose in the gripper frame at the instant TRANSPORT is
entered, then joins it to what the carry did: survived, dropped, or jammed.
Both actors run through the same deployed FSM (fsm_sim), so the only difference
between arms is FP32 vs INT8 weights.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsm_sim as F
from fsm_sim import (FSM, load_actor, OBJ_X, OBJ_Y, OBJ_Z, BIN_X, BIN_Y,
                     TRANSPORT, OK, FAIL, NAMES, GRASP_CAP, TL_STEPS,
                     MAX_GRASP_ATTEMPTS)
import robosuite as suite
from robosuite.wrappers import GymWrapper


def quat_to_yaw(q):
    x, y, z, w = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_to_mat(q):
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp-ckpt"); p.add_argument("--place-ckpt")
    p.add_argument("--grasp-tflite"); p.add_argument("--place-tflite")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True); p.add_argument("--tag", default="")
    a = p.parse_args()

    np.random.seed(a.seed); T.manual_seed(a.seed)
    raw = suite.make("PickPlace", robots="Panda",
                     controller_configs=suite.load_controller_config(
                         default_controller="OSC_POSE"),
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, horizon=700, reward_shaping=True,
                     control_freq=20, single_object_mode=2, object_type="bread")
    env = GymWrapper(raw); env.seed(a.seed)
    lo, hi = env.action_space.low, env.action_space.high

    if a.int8:
        import tensorflow as tf

        class TFLiteActor:
            def __init__(self, path):
                self.it = tf.lite.Interpreter(model_path=path); self.it.allocate_tensors()
                self.inp = self.it.get_input_details()[0]
                self.out = self.it.get_output_details()[0]

            def __call__(self, x):
                v = x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)
                v = v.reshape(1, -1).astype(np.float32)
                sc, zp = self.inp["quantization"]
                if self.inp["dtype"] == np.int8:
                    v = np.clip(np.round(v / sc + zp), -128, 127).astype(np.int8)
                self.it.set_tensor(self.inp["index"], v); self.it.invoke()
                o = self.it.get_tensor(self.out["index"])
                sc, zp = self.out["quantization"]
                if self.out["dtype"] == np.int8:
                    o = (o.astype(np.float32) - zp) * sc
                return T.tensor(o.astype(np.float32))
        g_actor = TFLiteActor(a.grasp_tflite); p_actor = TFLiteActor(a.place_tflite)
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

    def handoff_pose():
        """Object centre and yaw in the GRIPPER frame, plus the finger opening."""
        od = raw._get_observations()
        name = raw.objects[raw.object_id].name
        obj_p = np.array(od[f"{name}_pos"]); obj_q = np.array(od[f"{name}_quat"])
        eef_p = np.array(od["robot0_eef_pos"]); eef_q = np.array(od["robot0_eef_quat"])
        off = quat_to_mat(eef_q).T @ (obj_p - eef_p)
        rel = quat_to_yaw(obj_q) - quat_to_yaw(eef_q)
        rel = (rel + np.pi / 2) % np.pi - np.pi / 2      # wrap to +/-90 deg
        gq = np.array(od.get("robot0_gripper_qpos", np.zeros(2)))
        return dict(off_x=float(off[0]), off_y=float(off[1]), off_z=float(off[2]),
                    off_xy=float(np.hypot(off[0], off[1])),
                    rel_yaw=float(np.degrees(rel)),
                    grip_w=float(abs(gq[0] - gq[1])) if gq.size >= 2 else 0.0,
                    eef_z=float(eef_p[2]))

    rows = []
    for ep in range(a.episodes):
        obs, attempts, reached = None, 0, False
        for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
            obs = env.reset(); fsm = FSM(); attempts = attempt
            for _ in range(GRASP_CAP + TL_STEPS + 5):
                gr, pl = flags()
                act = fsm.step(np.asarray(obs, np.float32), gr, pl, g_actor, p_actor)
                obs, _, done, _ = env.step(np.clip(act, lo, hi))
                if fsm.phase == TRANSPORT:
                    reached = True; break
                if fsm.phase in (OK, FAIL) or done:
                    break
            if reached:
                break
        if not reached:
            rows.append(dict(episode=ep, tag=a.tag, outcome="handoff_failed",
                             success=0, attempts=attempts)); continue

        pose = handoff_pose()                     # <-- the moment of interest
        held, n, moved = [], 0, []
        while n < 700:
            gr, pl = flags()
            s = np.asarray(obs, np.float32)
            if fsm.phase == TRANSPORT:
                held.append(int(gr)); moved.append((float(s[OBJ_X]), float(s[OBJ_Y])))
            act = fsm.step(s, gr, pl, g_actor, p_actor)
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            n += 1
            if fsm.phase in (OK, FAIL) or done:
                break

        ok = int(fsm.phase == OK)
        frac = float(np.mean(held)) if held else float("nan")
        if ok:
            outcome = "carried_ok"
        elif held and frac < 0.9:
            outcome = "dropped_in_carry"
        elif len(held) >= 300:
            outcome = "jammed"
        else:
            outcome = "other_fail"
        r = dict(episode=ep, tag=a.tag, outcome=outcome, success=ok,
                 attempts=attempts, tr_steps=len(held), frac_grasped=frac)
        r.update(pose)
        rows.append(r)
        if (ep + 1) % 25 == 0:
            print(f"  {ep+1}/{a.episodes}", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    s = sum(r.get("success", 0) for r in rows)
    print(f"\n{a.tag} {s}/{len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
