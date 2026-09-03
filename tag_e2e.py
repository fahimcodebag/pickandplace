#!/usr/bin/env python3
"""Full FSM end-to-end under REAL AprilTag perception.

Everything before this measured perception on the grasp stage alone
(sweep_perception.py scores info["grasp_success"] with the grasp actor and no
FSM). This runs the deployed pipeline -- grasp, test-lift, transport, place --
with poses coming from the rendered tag through OpenCV and solvePnP.

Modes:
  truth      ground-truth poses; the ceiling for this policy
  recompute  ZOH world pose + per-step relative recompute (the existing default)
  latch      recompute until grasped, then latch object->gripper and propagate
             the world pose from proprioception

Measured (Results/tag_inflight): tag detection is 0% during TEST_LIFT and
TRANSPORT at BOTH camera positions because the gripper occludes it. "latch"
exists because a held object does not need to be seen; "recompute" is expected
to fail there, holding a stale world pose while the arm carries the object away.
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriltag_sim import make_tagged_env, TagDetector
from perception_wrapper import relative_block, _compose_world, perturb_pose
import fsm_sim as F

KEYS = ("Bread_pos", "Bread_quat", "Bread_to_robot0_eef_pos",
        "Bread_to_robot0_eef_quat", "robot0_joint_pos_cos",
        "robot0_joint_pos_sin", "robot0_joint_vel", "robot0_eef_pos",
        "robot0_eef_quat", "robot0_gripper_qpos", "robot0_gripper_qvel")
OBJ_POS, OBJ_QUAT = slice(0, 3), slice(3, 7)
REL_POS, REL_QUAT = slice(7, 10), slice(10, 14)
EEF_POS, EEF_QUAT = slice(35, 38), slice(38, 42)


def flat(od):
    return np.concatenate([np.asarray(od[k]).ravel() for k in KEYS]).astype(np.float64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp-ckpt", required=True)
    p.add_argument("--place-ckpt", default="checkpoints/td3_place/best")
    p.add_argument("--mode", choices=["truth", "recompute", "latch"], required=True)
    p.add_argument("--camera", default="agentview",
                   help="comma-separated for multi-camera fusion, e.g. "
                        "agentview,robot0_robotview")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--period", type=int, default=5, help="control steps between detections")
    p.add_argument("--latch-median", type=int, default=1,
                   help="latch the MEDIAN of the last K detected world poses "
                        "instead of the single current one. The object is "
                        "stationary before the grasp, so averaging is free and "
                        "each detection carries ~11 mm of independent error.")
    p.add_argument("--episodes", type=int, default=25)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    cams = [c.strip() for c in a.camera.split(",") if c.strip()]
    env, meta = make_tagged_env(camera=cams[0], width=a.width, height=a.height,
                                horizon=700, camera_names=cams)
    dets = ([] if a.mode == "truth" else
            [TagDetector(env, camera=c, width=a.width, height=a.height,
                         marker_size_m=meta["marker_size_m"]) for c in cams])

    def detect_best(od):
        """Fuse detections across cameras by AVERAGING, not selecting.

        Both cameras are STATIC (measured 0.00 mm drift over 30 steps of arm
        motion), so extrinsics are exact constants -- no pose estimation, no
        IMU. The second view is purely geometric: the gripper rarely occludes
        the tag from two angles at once.

        Selecting the lower-reprojection-error view was tried and is WRONG.
        Reprojection error is a pixel-space fit measured inside each camera's
        own image, and is not comparable across views with different geometry:
        it chose robot0_robotview 60% of the time and landed at 15.0 mm median,
        worse than agentview alone at 10.4 mm. Averaging the two poses gives
        8.0 mm -- better than either camera by itself, since the two error
        sources are largely independent.
        """
        got = []
        for d in dets:
            r = d.detect(od)
            if r is not None:
                got.append(r)
        if not got:
            return None
        if len(got) == 1:
            return got[0]
        P = np.array([g[0] for g in got])
        Q = np.array([g[1] for g in got])
        Q = Q * np.sign(Q @ Q[0])[:, None]     # q and -q are the same rotation
        q = Q.mean(axis=0)
        return P.mean(axis=0), q / max(np.linalg.norm(q), 1e-9)
    np.random.seed(a.seed); T.manual_seed(a.seed)
    g = F.load_actor(a.grasp_ckpt); pl = F.load_actor(a.place_ckpt)

    rows = []
    for ep in range(a.episodes):
        od = env.reset()
        fsm = F.FSM(); fsm.regrasp_enabled = True
        held = None; latched = None; hist = []
        n_det = n_tick = n_truthfall = 0
        placed = False
        for t in range(700):
            s = flat(od)
            try:
                grasped = bool(env._check_grasp(gripper=env.robots[0].gripper,
                                                object_geoms=env.objects[env.object_id]))
                placed = bool(env._check_success())
            except Exception:
                grasped = False

            if a.mode != "truth":
                if t % a.period == 0:
                    n_tick += 1
                    r = detect_best(od)
                    if r is not None:
                        n_det += 1
                        held = (np.asarray(r[0]), np.asarray(r[1]))
                        if not grasped:      # only average while it is static
                            hist.append(held)
                            if len(hist) > a.latch_median:
                                hist.pop(0)
                if held is None:
                    n_truthfall += 1            # running on ground truth
                else:
                    if a.mode == "latch":
                        if latched is None and grasped:
                            if a.latch_median > 1 and len(hist) > 1:
                                # component-wise median of the recent static
                                # observations; quaternion sign-aligned first
                                P = np.array([h[0] for h in hist])
                                Q = np.array([h[1] for h in hist])
                                Q = Q * np.sign(Q @ Q[-1])[:, None]
                                lp = np.median(P, axis=0)
                                lq = np.median(Q, axis=0)
                                lq = lq / max(np.linalg.norm(lq), 1e-9)
                            else:
                                lp, lq = held
                            latched = relative_block(lp, lq,
                                                     s[EEF_POS], s[EEF_QUAT])
                        if latched is not None:
                            s[REL_POS], s[REL_QUAT] = latched
                            wp, wq = _compose_world(latched[0], latched[1],
                                                    s[EEF_POS], s[EEF_QUAT])
                            s[OBJ_POS], s[OBJ_QUAT] = wp, wq
                        else:
                            s[OBJ_POS], s[OBJ_QUAT] = held
                            s[REL_POS], s[REL_QUAT] = relative_block(
                                held[0], held[1], s[EEF_POS], s[EEF_QUAT])
                    else:                        # recompute
                        s[OBJ_POS], s[OBJ_QUAT] = held
                        s[REL_POS], s[REL_QUAT] = relative_block(
                            held[0], held[1], s[EEF_POS], s[EEF_QUAT])

            act = fsm.step(s.astype(np.float32), grasped, placed, g, pl)
            od, _, done, _ = env.step(act)
            if fsm.phase in (F.OK, F.FAIL) or done:
                break
        rows.append(dict(episode=ep, mode=a.mode, success=int(fsm.phase == F.OK),
                         steps=t + 1, phase=int(fsm.phase), ticks=n_tick,
                         detections=n_det, truth_fallback_steps=n_truthfall,
                         latched=int(latched is not None)))
        if (ep + 1) % 5 == 0:
            print(f"  {ep+1}/{a.episodes} success {np.mean([x['success'] for x in rows])*100:.0f}%",
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
