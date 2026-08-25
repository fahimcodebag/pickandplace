#!/usr/bin/env python3
"""Handoff-pose diagnostic.

Records the state of the object IN THE GRIPPER at the moment the grasp stage
hands off to the transport policy, then the episode outcome. Tests whether
transport_stall clusters in handoff-pose space -- i.e. whether the grasp policy
is presenting configurations the fixed-spawn transport policy never saw.

Columns: episode, reason, success, handoff_attempts, place_steps,
         rel_yaw (object yaw minus wrist yaw, wrapped to +/-90 deg),
         eef_yaw, obj_yaw, off_x, off_y, off_z (object centre in the GRIPPER
         frame, metres), off_xy (lateral offset magnitude), obj_z.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Decomposed state training"))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "Decomposed state training"))
import test_place as tp
from networks import ActorNetwork


def quat_to_yaw(q):
    """robosuite quats are xyzw."""
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
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--grasp-chkpt-dir", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    np.random.seed(a.seed); T.manual_seed(a.seed)
    env, raw = tp.make_place_eval_env(
        render=False, grasp_chkpt_dir=a.grasp_chkpt_dir, native_spawn=True)
    env._NEAR_TARGET_XY = 0.14
    env._RELEASE_TRIGGER_HOLD = 3
    env.PLACE_HORIZON = 300

    place_dir = os.path.join("..", "checkpoints", "td3_place", "best")
    actor = ActorNetwork(env.observation_space.shape[0], 64, 32,
                         env.action_space.shape[0], chkpt_dir=place_dir)
    sd = T.load(os.path.join(place_dir, "actor_td3"), map_location="cpu")
    actor.load_state_dict({k: v for k, v in sd.items()
                           if not k.startswith("log_std")})
    actor.to(T.device("cpu")); actor.device = T.device("cpu"); actor.eval()

    rows = []
    for i in range(a.episodes):
        obs = env.reset()
        od = raw._get_observations()
        name = env._get_target_obj_name()
        obj_p = np.array(od[f"{name}_pos"]); obj_q = np.array(od[f"{name}_quat"])
        eef_p = np.array(od["robot0_eef_pos"]); eef_q = np.array(od["robot0_eef_quat"])

        # object centre expressed in the gripper frame
        off = quat_to_mat(eef_q).T @ (obj_p - eef_p)
        jp = np.array(od.get("robot0_joint_pos", np.zeros(7)))
        gq = np.array(od.get("robot0_gripper_qpos", np.zeros(2)))
        oy, ey = quat_to_yaw(obj_q), quat_to_yaw(eef_q)
        # the bread is symmetric under 180 deg, and the gripper under 180 deg,
        # so the meaningful residual is wrapped to +/-90 deg
        rel = (oy - ey + np.pi / 2) % np.pi - np.pi / 2

        done, n, info = False, 0, {}
        while not done:
            with T.no_grad():
                act = actor(T.tensor(obs, dtype=T.float).unsqueeze(0)
                            ).squeeze(0).numpy()
            obs, r, done, info = env.step(act); n += 1

        rows.append(dict(
            episode=i, reason=info.get("place_done_reason", "?"),
            success=int(bool(info.get("place_success", False))),
            handoff_attempts=int(info.get("handoff_attempts", 1)),
            place_steps=n,
            rel_yaw=np.degrees(rel), eef_yaw=np.degrees(ey),
            obj_yaw=np.degrees(oy),
            off_x=off[0], off_y=off[1], off_z=off[2],
            off_xy=float(np.hypot(off[0], off[1])), obj_z=obj_p[2],
            eef_x=eef_p[0], eef_y=eef_p[1], eef_z=eef_p[2],
            grip_w=float(abs(gq[0] - gq[1])) if gq.size >= 2 else 0.0,
            **{f"j{k}": float(v) for k, v in enumerate(jp)}))
        if (i + 1) % 50 == 0:
            sr = np.mean([r["success"] for r in rows]) * 100
            print(f"  {i+1}/{a.episodes}  running {sr:.1f}%", flush=True)

    out = os.path.join("..", a.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
