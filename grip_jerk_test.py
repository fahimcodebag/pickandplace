#!/usr/bin/env python3
"""Does the grip survive scripted carry / acceleration jerk, and does the
MONOLITHIC policy hold on better than the decomposed one?

Motivation. Results/rj_diag showed mid-carry drops are what Fix A can only
partly recover (regrab 10.7% once the object lands outside the training spawn
box). Retraining on a wider dropzone treats the symptom. The alternative is a
grip that does not let go in the first place. But "the transport policy shakes
the object loose" has never been tested against "the grip is simply weak":
every previous measurement ran the learned transport policy, so grip quality
and trajectory quality were confounded.

This harness removes the transport policy. It grasps, certifies the lift with
the deployed criterion, then hands the arm a SCRIPTED open-loop motion under
the same rule-layer treatment TRANSPORT applies (rotation zeroed, gripper
forced closed). Whatever drops now is the grip, not the navigation.

  hold   zero translation                     -- control; anything failing here
                                                 was never really holding
  carry  constant amp on one horizontal axis  -- "simple carrying motion"
  jerk   bang-bang +-amp every --jerk-period  -- acceleration/deceleration

Direction is (ep % 4) -> +x, -x, +y, -y, so every arm sees an identical
sequence of spawns AND disturbances and the arms can be compared pairwise.

Also records grip diagonal angle and joint margin at handoff, so survival can
be cross-tabbed against the two variables earlier work implicated.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robosuite as suite
from robosuite.wrappers import GymWrapper

# Deployed handoff criterion, verbatim from fsm_sim.py / the .ino
GRASP_HOLD, GRASP_CAP = 8, 250
TL_STEPS, TL_DZ, TL_MIN_RISE = 20, 0.5, 0.03
# Panda joint limits. Joint 5's lower bound is -0.0175 ~ 0, so a normal
# posture reads as "near limit" by construction -- see regrasp_jam_diag.py.
JLO = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JHI = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
DIRS = [(0, +1.0), (0, -1.0), (1, +1.0), (1, -1.0)]


def quat_yaw(q):
    x, y, z, w = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_mat(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mono-ckpt", default=None,
                   help="monolithic v7/v8 dir (networks_v2, 2048x1024, goal-conditioned)")
    p.add_argument("--actor-file", default="actor_td3")
    p.add_argument("--grasp-ckpt", default=None, help="decomposed grasp actor dir")
    p.add_argument("--grasp-tflite", default=None, help="decomposed grasp INT8")
    p.add_argument("--profile", choices=["hold", "carry", "jerk"], required=True)
    p.add_argument("--dist-steps", type=int, default=60)
    p.add_argument("--amp", type=float, default=0.5,
                   help="0.5 matches TRANSPORT's CARRY_CLIP")
    p.add_argument("--jerk-period", type=int, default=3,
                   help="steps between sign flips in the jerk profile")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", default="")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    np.random.seed(a.seed); T.manual_seed(a.seed)
    raw = suite.make("PickPlace", robots="Panda",
                     controller_configs=suite.load_controller_config(
                         default_controller="OSC_POSE"),
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, horizon=700, reward_shaping=True,
                     control_freq=20, single_object_mode=2, object_type="bread")
    env = GymWrapper(raw)
    env.seed(a.seed)

    if a.mono_ckpt:
        from networks_v2 import ActorNetwork as MonoActor
        net = MonoActor(obs_dim=46, goal_dim=3, fc1_dims=2048, fc2_dims=1024,
                        n_actions=7, name="actor", chkpt_dir=a.mono_ckpt)
        net.load_state_dict(T.load(os.path.join(a.mono_ckpt, a.actor_file),
                                   map_location="cpu"))
        net.to(T.device("cpu")); net.device = T.device("cpu"); net.eval()
        goal = T.tensor(np.array(raw.target_bin_placements[raw.object_to_id["bread"]],
                                 dtype=np.float32)).unsqueeze(0)

        def policy(s):
            with T.no_grad():
                return net(T.tensor(s, dtype=T.float).unsqueeze(0),
                           goal).squeeze(0).numpy()
    elif a.grasp_tflite:
        import tensorflow as tf
        it = tf.lite.Interpreter(model_path=a.grasp_tflite)
        it.allocate_tensors()
        inp, out = it.get_input_details()[0], it.get_output_details()[0]

        def policy(s):
            v = np.asarray(s).reshape(1, -1).astype(np.float32)
            sc, zp = inp["quantization"]
            if inp["dtype"] == np.int8:
                v = np.clip(np.round(v / sc + zp), -128, 127).astype(np.int8)
            it.set_tensor(inp["index"], v); it.invoke()
            o = it.get_tensor(out["index"]); sc, zp = out["quantization"]
            if out["dtype"] == np.int8:
                o = (o.astype(np.float32) - zp) * sc
            return o.astype(np.float32).reshape(-1)
    else:
        from networks import ActorNetwork
        net = ActorNetwork(46, 64, 32, 7, chkpt_dir=a.grasp_ckpt)
        sd = T.load(os.path.join(a.grasp_ckpt, "actor_td3"), map_location="cpu")
        net.load_state_dict({k: v for k, v in sd.items()
                             if not k.startswith("log_std")})
        net.to(T.device("cpu")); net.device = T.device("cpu"); net.eval()

        def policy(s):
            with T.no_grad():
                return net(T.tensor(s, dtype=T.float).unsqueeze(0)).squeeze(0).numpy()

    def grasped():
        try:
            return bool(raw._check_grasp(gripper=raw.robots[0].gripper,
                                         object_geoms=raw.objects[raw.object_id]))
        except Exception:
            return False

    def margin():
        q = raw.sim.data.qpos[raw.robots[0]._ref_joint_pos_indexes]
        m = np.minimum(q - JLO, JHI - q)
        return float(m.min()), int(np.argmin(m))

    nact = raw.action_spec[0].shape[0]
    rows = []
    for ep in range(a.episodes):
        s = env.reset()
        axis, sign = DIRS[ep % 4]
        r = dict(episode=ep, seed=a.seed, profile=a.profile, tag=a.tag,
                 axis=axis, sign=sign, grasp_ok=0, lift_ok=0, survived=0,
                 lost_at=-1, diag_deg="", off_xy0="", drift="", rise="",
                 eef_path="", eef_net="", eef_vmax="", eef_amax="",
                 margin_hand="", joint_hand="", grasp_step=-1)

        # ---- 1. grasp, deployed handoff criterion --------------------------
        hold, n, done = 0, 0, False
        while n < GRASP_CAP:
            s, _, done, _ = env.step(np.clip(policy(s), -1.0, 1.0)); n += 1
            hold = hold + 1 if grasped() else 0
            if hold >= GRASP_HOLD:
                break
            if done:
                break
        if hold < GRASP_HOLD:
            rows.append(r); continue
        r["grasp_ok"], r["grasp_step"] = 1, n

        od = raw._get_observations()
        oq, eq = np.array(od["Bread_quat"]), np.array(od["robot0_eef_quat"])
        off0 = quat_mat(eq).T @ (np.array(od["Bread_pos"])
                                 - np.array(od["robot0_eef_pos"]))
        rel = np.degrees(quat_yaw(oq) - quat_yaw(eq))
        r["diag_deg"] = float(abs((rel + 45) % 90 - 45))   # deg from a flat face
        r["off_xy0"] = float(np.hypot(off0[0], off0[1]))

        # ---- 2. scripted lift certification --------------------------------
        base = float(od["Bread_pos"][2])
        la = np.zeros(nact, dtype=np.float32); la[2], la[-1] = TL_DZ, 1.0
        best, ok = 0.0, True
        for _ in range(TL_STEPS):
            _, _, done, _ = env.step(la)
            best = max(best, float(raw._get_observations()["Bread_pos"][2] - base))
            if not grasped():
                ok = False; break
            if done:
                break
        r["rise"] = best
        if not (ok and best >= TL_MIN_RISE):
            rows.append(r); continue
        r["lift_ok"] = 1
        m, j = margin(); r["margin_hand"], r["joint_hand"] = m, j

        # ---- 3. scripted disturbance, TRANSPORT's rule layer ---------------
        # a[3:6] = 0 (rotation discarded) and a[6] = 1.0 (gripper forced shut)
        # are exactly what the deployed rule layer does during the carry.
        surv = True
        eefs = [np.array(raw._get_observations()["robot0_eef_pos"])]
        for k in range(a.dist_steps):
            act = np.zeros(nact, dtype=np.float32); act[-1] = 1.0
            if a.profile == "carry":
                act[axis] = sign * a.amp
            elif a.profile == "jerk":
                flip = -1.0 if (k // a.jerk_period) % 2 else 1.0
                act[axis] = sign * flip * a.amp
            _, _, done, _ = env.step(act)
            eefs.append(np.array(raw._get_observations()["robot0_eef_pos"]))
            if not grasped():
                surv = False; r["lost_at"] = k; break
            if done:
                break
        E = np.array(eefs)
        if len(E) > 3:
            v = np.diff(E, axis=0) * 20.0            # m/s at 20 Hz
            acc = np.diff(v, axis=0) * 20.0          # m/s^2
            r["eef_path"] = float(np.linalg.norm(np.diff(E, axis=0), axis=1).sum())
            r["eef_net"] = float(np.linalg.norm(E[-1] - E[0]))
            r["eef_vmax"] = float(np.linalg.norm(v, axis=1).max())
            r["eef_amax"] = float(np.linalg.norm(acc, axis=1).max())
        r["survived"] = int(surv)
        if surv:
            od = raw._get_observations()
            eq = np.array(od["robot0_eef_quat"])
            off = quat_mat(eq).T @ (np.array(od["Bread_pos"])
                                    - np.array(od["robot0_eef_pos"]))
            r["drift"] = float(np.linalg.norm(off - off0))
        rows.append(r)

        if (ep + 1) % 25 == 0:
            lc = [x for x in rows if x["lift_ok"]]
            sv = sum(x["survived"] for x in rows)
            print(f"  {ep+1}/{a.episodes} lift-cert {len(lc)} survived {sv}",
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
