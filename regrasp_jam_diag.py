#!/usr/bin/env python3
"""Two open questions from Results/transport_stall_diagnosis.txt.

Q1  Fix A recovers only 16.7% of the lost-grip class, yet 83% of dropped
    objects land ON THE TABLE. Where do they actually land, does the re-pick
    re-establish a grip, and what is binding -- MAX_REGRASP, the step budget,
    or the object sitting outside the distribution the grasp policy was
    trained on?

Q2  Three open-loop escapes all failed AFTER the arm was pinned. A guard that
    acts BEFORE needs warning time. Does joint_margin decay gradually, or
    collapse in a couple of steps? If it collapses, no preventive guard can
    work and item #2 is dead before it is built.

Changes no behaviour: fsm_sim's FSM is imported and run with --regrasp.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsm_sim as F
from fsm_sim import (FSM, load_actor, OBJ_X, OBJ_Y, OBJ_Z, BIN_X, BIN_Y,
                     GRASP, TRANSPORT, OK, FAIL, NAMES, GRASP_CAP, TL_STEPS,
                     MAX_GRASP_ATTEMPTS)
import robosuite as suite
from robosuite.wrappers import GymWrapper

JLO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JHI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


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

    # the spawn box the grasp policy was trained on
    env.reset()
    smp = raw.placement_initializer.samplers["CollisionObjectSampler"]
    SX = np.array(smp.x_range, float); SY = np.array(smp.y_range, float)
    ref = np.array(getattr(smp, "reference_pos", [0, 0, 0]), float)[:2]
    print(f"spawn box x {SX+ref[0]} y {SY+ref[1]}", flush=True)

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

    def margin():
        q = raw.sim.data.qpos[raw.robots[0]._ref_joint_pos_indexes]
        m = np.minimum(q - JLO, JHI - q)
        return float(m.min()), int(np.argmin(m))

    rows = []
    for ep in range(a.episodes):
        obs, reached = None, False
        for _ in range(MAX_GRASP_ATTEMPTS):
            obs = env.reset(); fsm = FSM(); fsm.regrasp_enabled = True
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
            rows.append(dict(episode=ep, tag=a.tag, outcome="handoff_failed", success=0))
            continue

        marg, n = [], 0
        n_transport_entries, prev_ph = 1, fsm.phase
        drop = None            # (x, y, z, step) at the moment the grip is lost
        held_prev = True
        while n < 700:
            gr, pl = flags()
            s = np.asarray(obs, np.float32)
            if fsm.phase == TRANSPORT:
                m, _j = margin(); marg.append(m)
                if held_prev and not gr and drop is None:
                    drop = (float(s[OBJ_X]), float(s[OBJ_Y]), float(s[OBJ_Z]), n)
                held_prev = gr
            act = fsm.step(s, gr, pl, g_actor, p_actor)
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            if fsm.phase == TRANSPORT and prev_ph != TRANSPORT:
                n_transport_entries += 1
            prev_ph = fsm.phase
            n += 1
            if fsm.phase in (OK, FAIL) or done:
                break

        s = np.asarray(obs, np.float32); gr, _ = flags()
        m_end, j_end = margin()
        r = dict(episode=ep, tag=a.tag, success=int(fsm.phase == OK),
                 steps=n, tr_len=len(marg), regrasps=fsm.regrasps,
                 n_transport_entries=n_transport_entries,
                 end_grasped=int(gr), margin_end=m_end, joint_end=j_end)
        # --- Q2: warning time before the jam
        if marg:
            M = np.asarray(marg)
            for k in (5, 10, 20, 40):
                r[f"margin_{k}_before"] = float(M[-k]) if len(M) >= k else float("nan")
            # Joint 5's lower bound is -0.0175 rad, so a normal posture with
            # q5 ~ 0 reads as "near limit" by construction. A transient dip is
            # not a pin -- what matters is the length of the FINAL sustained
            # run below threshold, and how long before the end it began.
            r["margin_min"] = float(M.min())
            for thr in (0.05, 0.02):
                below = M < thr
                run = 0
                for v in below[::-1]:
                    if v: run += 1
                    else: break
                r[f"pinned_run_{thr}"] = int(run)
                # warning time: steps from the start of that final run to the end
                r[f"warn_steps_{thr}"] = int(run) if run < len(M) else -1
        # --- Q1: where the object landed, and was it inside the spawn box
        if drop is not None:
            dx, dy, dz, dstep = drop
            r.update(drop_x=dx, drop_y=dy, drop_z=dz, drop_step=dstep,
                     steps_left=700 - n,
                     in_spawn_box=int(SX[0]+ref[0] <= dx <= SX[1]+ref[0]
                                      and SY[0]+ref[1] <= dy <= SY[1]+ref[1]),
                     regrabbed=int(n_transport_entries > 1))
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
