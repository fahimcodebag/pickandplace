#!/usr/bin/env python3
"""Does the MONOLITHIC (v7/v8) policy align its wrist to the object?

The decomposed grasp policy does not: corr(wrist yaw, object yaw) ~ 0.1, and
24% of its grasps close on a corner. Four reward interventions failed to move
it. If the monolithic model -- same task, same observation, 400x the
parameters, 30M timesteps -- DOES align, then alignment is learnable and
something about the decomposed setup prevents it. If it does not, the finding
generalises.

Measures grip angle at first stable grasp, plus whether the lift survives.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robosuite as suite
from robosuite.wrappers import GymWrapper
from networks_v2 import ActorNetwork

N_HOLD   = 3      # consecutive grasped steps before we call the grip settled
LIFT_STEPS, LIFT_DZ, MIN_RISE = 20, 0.5, 0.03


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
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-hold", type=int, default=3,
                   help="consecutive grasped steps before lifting; set 8 to "
                        "match the decomposed stage's N_GRASP_HOLD_LIFT")
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--actor-file", default="actor_td3",
                   help="actor_td3 (final) or actor_td3_best (peak). The final "
                        "weights of a decayed run are not the peak policy.")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    np.random.seed(a.seed); T.manual_seed(a.seed)
    env = suite.make("PickPlace", robots="Panda",
                     controller_configs=suite.load_controller_config(
                         default_controller="OSC_POSE"),
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, horizon=a.horizon, reward_shaping=True,
                     control_freq=20, single_object_mode=2, object_type="bread")
    genv = GymWrapper(env)
    genv.seed(a.seed)
    goal = np.array(env.target_bin_placements[env.object_to_id["bread"]],
                    dtype=np.float32)

    actor = ActorNetwork(obs_dim=46, goal_dim=3, fc1_dims=2048,
                         fc2_dims=1024, n_actions=7, name="actor",
                         chkpt_dir=a.ckpt)
    actor.load_state_dict(T.load(os.path.join(a.ckpt, a.actor_file),
                                 map_location="cpu"))
    actor.to(T.device("cpu")); actor.device = T.device("cpu"); actor.eval()

    def grasped():
        try:
            return env._check_grasp(gripper=env.robots[0].gripper,
                                    object_geoms=env.objects[env.object_id])
        except Exception:
            return False

    rows = []
    gt = T.tensor(goal).unsqueeze(0)
    for ep in range(a.episodes):
        obs = genv.reset()
        hold, rec, n = 0, None, 0
        while n < a.horizon:
            with T.no_grad():
                act = actor(T.tensor(obs, dtype=T.float).unsqueeze(0),
                            gt).squeeze(0).numpy()
            obs, r, done, info = genv.step(act); n += 1
            if grasped():
                hold += 1
                if hold >= a.n_hold and rec is None:
                    od = env._get_observations()
                    oq = np.array(od["Bread_quat"]); eq = np.array(od["robot0_eef_quat"])
                    off = quat_mat(eq).T @ (np.array(od["Bread_pos"])
                                            - np.array(od["robot0_eef_pos"]))
                    rel = np.degrees(quat_yaw(oq) - quat_yaw(eq))
                    rel = abs((rel + 45) % 90 - 45)     # deg from a flat face
                    rec = dict(episode=ep, grasp_step=n, diag_deg=rel,
                               off_xy=float(np.hypot(off[0], off[1])),
                               obj_z=float(od["Bread_pos"][2]))
                    break
            else:
                hold = 0
            if done:
                break
        if rec is None:
            rows.append(dict(episode=ep, grasp_step=-1, diag_deg="",
                             off_xy="", obj_z="", lift_ok=0, rise=""))
            continue
        # scripted lift certification, identical to the decomposed stage
        base = env._get_observations()["Bread_pos"][2]
        la = np.zeros(env.action_spec[0].shape[0], dtype=np.float32)
        la[2] = LIFT_DZ; la[-1] = 1.0
        best = 0.0; ok = True
        for _ in range(LIFT_STEPS):
            env.step(la)
            cur = env._get_observations()["Bread_pos"][2]
            best = max(best, float(cur - base))
            if not grasped():
                ok = False; break
        rec["lift_ok"] = int(ok and best >= MIN_RISE); rec["rise"] = best
        rows.append(rec)
        if (ep + 1) % 25 == 0:
            got = [r for r in rows if r["grasp_step"] != -1]
            print(f"  {ep+1}/{a.episodes}  grasped {len(got)/len(rows)*100:.0f}%",
                  flush=True)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
