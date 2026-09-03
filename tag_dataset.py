#!/usr/bin/env python3
"""Collect (features -> pose residual) data to test whether AprilTag error is
LEARNABLE before committing to a model.

Target is the RESIDUAL (truth - detected), not the absolute pose: the detected
pose is already ~90% right, so regressing to absolute truth lets a model score
well by echoing its input. The residual is centred near zero and its variance
IS the quantity to reduce.

A constant-offset version of this model is already deployed
(TagDetector.calib_world_m), and it took median error 17.2 -> 11.2 mm. The
question here is whether what REMAINS has structure, or is isotropic detector
noise that no model can remove.

Features are all computable at runtime on the target: they come from the
detector output and the robot's own encoders. Ground truth is available only
in simulation, which is exactly why this question has to be settled here.
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriltag_sim import make_tagged_env, TagDetector
import robosuite.utils.camera_utils as CU
import robosuite.utils.transform_utils as TU
import fsm_sim as F

KEYS = ("Bread_pos", "Bread_quat", "Bread_to_robot0_eef_pos",
        "Bread_to_robot0_eef_quat", "robot0_joint_pos_cos",
        "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp-ckpt", required=True)
    p.add_argument("--place-ckpt", default="checkpoints/td3_place/best")
    p.add_argument("--camera", default="agentview")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--every", type=int, default=3)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    env, meta = make_tagged_env(camera=a.camera, width=a.width,
                                height=a.height, horizon=700)
    det = TagDetector(env, camera=a.camera, width=a.width, height=a.height,
                      marker_size_m=meta["marker_size_m"])
    np.random.seed(a.seed); T.manual_seed(a.seed)
    g = F.load_actor(a.grasp_ckpt); pl = F.load_actor(a.place_ckpt)
    Cx = CU.get_camera_extrinsic_matrix(env.sim, a.camera)
    cam_pos = Cx[:3, 3]

    rows = []
    for ep in range(a.episodes):
        od = env.reset(); fsm = F.FSM(); fsm.regrasp_enabled = True
        for t in range(700):
            s = np.concatenate([np.asarray(od[k]).ravel() for k in KEYS]).astype(np.float32)
            try:
                gr = bool(env._check_grasp(gripper=env.robots[0].gripper,
                                           object_geoms=env.objects[env.object_id]))
                pc = bool(env._check_success())
            except Exception:
                gr = pc = False
            if t % a.every == 0 and not gr:
                r = det.detect(od)
                if r is not None:
                    tp = np.asarray(od["Bread_pos"]); tq = np.asarray(od["Bread_quat"])
                    ee = np.asarray(od["robot0_eef_pos"])
                    dp = np.asarray(r[0]); dq = np.asarray(r[1])
                    res = tp - dp                       # THE TARGET
                    # EVERY feature below must be computable at RUNTIME, so
                    # geometry is derived from the DETECTED pose dp, never from
                    # the ground truth tp. Using tp here leaks the answer into
                    # the inputs and inflates the fit.
                    ray = dp - cam_pos; rng = np.linalg.norm(ray); ray = ray / rng
                    v = ee - cam_pos
                    perp = float(np.linalg.norm(v - np.dot(v, ray) * ray))
                    # viewing obliquity: angle between the tag normal and the
                    # camera ray. A tag seen edge-on localises corners badly.
                    nrm = TU.quat2mat(dq)[:, 2]
                    obliq = float(np.degrees(np.arccos(np.clip(abs(nrm @ ray), 0, 1))))
                    # yaw of the DETECTED quaternion, not the true one
                    yaw = float(np.arctan2(2*(dq[3]*dq[2]+dq[0]*dq[1]),
                                           1-2*(dq[1]**2+dq[2]**2)))
                    rows.append(dict(
                        ep=ep, step=t, phase=int(fsm.phase),
                        res_x=res[0], res_y=res[1], res_z=res[2],
                        res_norm=float(np.linalg.norm(res)),
                        reproj=det.last_err, area_px=det.last_area_px,
                        obliq_deg=obliq, cam_range=float(rng),
                        gripper_perp=perp, gripper_dz=float(ee[2] - dp[2]),
                        det_x=dp[0], det_y=dp[1], det_z=dp[2],
                        obj_yaw=yaw,
                        cx=float(det.last_corners[:, 0].mean()),
                        cy=float(det.last_corners[:, 1].mean())))
            od, _, done, _ = env.step(fsm.step(s, gr, pc, g, pl))
            if fsm.phase in (F.OK, F.FAIL) or done:
                break
        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}/{a.episodes}  {len(rows)} detections", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
