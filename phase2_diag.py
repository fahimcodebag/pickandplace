#!/usr/bin/env python3
"""Phase-2 stall diagnostic: the post-trigger phases and the blocked carries.

Results/stall_diag established that TRANSPORT stalls split three ways, and
refuted the obvious reading of two of them:

  * RELEASED_MISSED_BIN releases at z ~1.11-1.26 -- LOWER than the successes
    (1.25), at the same XY error (0.164 vs 0.161). Height and XY do not
    separate success from failure, so the cause must be in RECENTER/DESCEND/
    OPEN/RETRACT, which were never instrumented.
  * BLOCKED_no_motion sits at z 1.41-1.52, inside the success range (p90 1.44).
    Height is not the cause; being stuck is. Joint/EEF state is needed to say
    whether the arm is at a limit, at reach extent, or in contact.

This records both. It changes no behaviour -- fsm_sim's FSM is imported.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsm_sim as F
from fsm_sim import (FSM, load_actor, OBJ_X, OBJ_Y, OBJ_Z, BIN_X, BIN_Y, BIN_Z,
                     GRASP, TRANSPORT, RECENTER, DESCEND, OPEN, RETRACT,
                     OK, FAIL, NAMES, GRASP_CAP, TL_STEPS, MAX_GRASP_ATTEMPTS)
import robosuite as suite
from robosuite.wrappers import GymWrapper


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

    # Panda joint limits, for "is the arm pinned against a limit?"
    JLO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    JHI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

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

    def probe():
        """EEF pose, joint margin to the nearest limit, gripper opening."""
        od = raw._get_observations()
        eef = np.array(od["robot0_eef_pos"])
        q = raw.sim.data.qpos[raw.robots[0]._ref_joint_pos_indexes].copy()
        marg = np.minimum(q - JLO, JHI - q)          # rad to nearest limit
        gq = np.array(od.get("robot0_gripper_qpos", np.zeros(2)))
        return eef, float(marg.min()), int(np.argmin(marg)), \
            float(np.linalg.norm(eef[:2])), float(abs(gq[0] - gq[1]) if gq.size >= 2 else 0.0)

    def dist(s):
        return float(np.hypot(s[OBJ_X] - BIN_X, s[OBJ_Y] - BIN_Y))

    rows = []
    for ep in range(a.episodes):
        obs, reached = None, False
        for _ in range(MAX_GRASP_ATTEMPTS):
            obs = env.reset(); fsm = FSM()
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
            rows.append(dict(episode=ep, success=0, phase="handoff_failed", tag=a.tag)); continue

        r = dict(episode=ep, tag=a.tag)
        ph_count = {n: 0 for n in NAMES}
        snap = {}          # per-phase state at first entry and at exit
        prev_ph, n = fsm.phase, 0
        tr_d, tr_stuck = [], []
        while n < 700:
            gr, pl = flags()
            s = np.asarray(obs, np.float32)
            ph = fsm.phase
            if ph != prev_ph:                      # phase transition: snapshot
                eef, marg, jidx, reach, gw = probe()
                snap[f"{NAMES[prev_ph]}_exit_d"] = dist(s)
                snap[f"{NAMES[prev_ph]}_exit_z"] = float(s[OBJ_Z])
                snap[f"{NAMES[ph]}_in_d"] = dist(s)
                snap[f"{NAMES[ph]}_in_z"] = float(s[OBJ_Z])
                snap[f"{NAMES[ph]}_in_grasped"] = int(gr)
                prev_ph = ph
            ph_count[NAMES[ph]] += 1
            if ph == TRANSPORT:
                tr_d.append(dist(s))
            act = fsm.step(s, gr, pl, g_actor, p_actor)
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            n += 1
            if fsm.phase in (OK, FAIL) or done:
                break

        s = np.asarray(obs, np.float32)
        eef, marg, jidx, reach, gw = probe()
        gr, pl = flags()
        r.update(success=int(fsm.phase == OK), phase=NAMES[fsm.phase], steps=n,
                 end_d=dist(s), end_z=float(s[OBJ_Z]), end_grasped=int(gr),
                 eef_x=float(eef[0]), eef_y=float(eef[1]), eef_z=float(eef[2]),
                 joint_margin=marg, joint_idx=jidx, reach_xy=reach, grip_w=gw,
                 d_min=float(min(tr_d)) if tr_d else float("nan"))
        for k in ["TRANSPORT", "RECENTER", "DESCEND", "OPEN", "RETRACT"]:
            r[f"n_{k}"] = ph_count[k]
        r.update(snap)
        rows.append(r)
        if (ep + 1) % 25 == 0:
            print(f"  {ep+1}/{a.episodes}", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval=""); w.writeheader(); w.writerows(rows)
    s_ = sum(r.get("success", 0) for r in rows)
    print(f"\n{a.tag} {s_}/{len(rows)} -> {a.out}")


if __name__ == "__main__":
    main()
