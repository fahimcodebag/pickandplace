#!/usr/bin/env python3
"""Detection rate and pose error DURING an episode, not just at reset.

validate_tag_perception.py detects one frame per episode with the arm parked at
home, which measures the easiest possible case for a fixed camera and the
hardest for a wrist camera (it is not yet pointed at anything). What the
deployed system actually needs is the detection rate while the FSM is running:
during REACH the object is far, during GRASP the gripper is on top of it and may
occlude the tag, during TRANSPORT the object is held and moving.

Runs the real policy under the real FSM, renders every --every steps, runs the
real OpenCV AprilTag detector, and scores each detection against MuJoCo truth.
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriltag_sim import make_tagged_env, TagDetector
from networks import ActorNetwork
import fsm_sim as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp-ckpt", required=True)
    p.add_argument("--place-ckpt", default="checkpoints/td3_place/best")
    p.add_argument("--camera", default="robot0_eye_in_hand")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--every", type=int, default=5, help="render every N control steps")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--z-offset", type=float, default=0.026)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    env, meta = make_tagged_env(camera=a.camera, width=a.width, height=a.height,
                                z_offset=a.z_offset, horizon=700)
    det = TagDetector(env, camera=a.camera, width=a.width, height=a.height,
                      marker_size_m=meta["marker_size_m"])
    np.random.seed(a.seed); T.manual_seed(a.seed)
    env.seed(a.seed) if hasattr(env, "seed") else None

    g = F.load_actor(a.grasp_ckpt); pl = F.load_actor(a.place_ckpt)
    raw = env
    rows = []
    for ep in range(a.episodes):
        obs_dict = env.reset()
        fsm = F.FSM(); fsm.regrasp_enabled = True
        for t in range(700):
            s = np.concatenate([obs_dict[k].ravel() for k in
                                ("Bread_pos", "Bread_quat", "Bread_to_robot0_eef_pos",
                                 "Bread_to_robot0_eef_quat", "robot0_joint_pos_cos",
                                 "robot0_joint_pos_sin", "robot0_joint_vel",
                                 "robot0_eef_pos", "robot0_eef_quat",
                                 "robot0_gripper_qpos", "robot0_gripper_qvel")]).astype(np.float32)
            try:
                grasped = bool(raw._check_grasp(gripper=raw.robots[0].gripper,
                                                object_geoms=raw.objects[raw.object_id]))
                placed = bool(raw._check_success())
            except Exception:
                grasped = placed = False
            if t % a.every == 0:
                r = det.detect(obs_dict)
                truth = np.array(obs_dict["Bread_pos"]) + np.array([0, 0, a.z_offset])
                err = np.nan if r is None else float(np.linalg.norm(np.array(r[0]) - truth))
                eef = np.array(obs_dict["robot0_eef_pos"])
                rows.append(dict(episode=ep, step=t, phase=int(fsm.phase),
                                 detected=int(r is not None), err_mm=err * 1000 if r is not None else "",
                                 dist_eef_obj=float(np.linalg.norm(
                                     np.array(obs_dict["Bread_pos"]) - eef)),
                                 grasped=int(grasped)))
            act = fsm.step(s, grasped, placed, g, pl)
            obs_dict, _, done, _ = env.step(act)
            if done or fsm.phase in (F.OK, F.FAIL):
                break
        if (ep + 1) % 5 == 0:
            d = [x["detected"] for x in rows]
            print(f"  ep {ep+1}/{a.episodes}  detection so far {np.mean(d)*100:.1f}%", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} samples)")


if __name__ == "__main__":
    main()
