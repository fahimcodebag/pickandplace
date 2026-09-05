#!/usr/bin/env python3
"""Collect the t=0 wrist-camera detection dataset.

The deployment case is now: ONE detection at episode start, wrist camera,
320x240 (75 KB buffer -- fits ESP32 internal SRAM, no PSRAM). So the corrector
must be refit on THAT distribution: it is a per-setup calibration and the
existing one was fitted on agentview at 1280x960.

Because detection happens with the arm at its home pose, no episode needs to be
simulated -- reset, look, record. That is ~100x cheaper than the agentview
collection, which had to run the policy to reach varied arm configurations.

Features that varied over an episode (gripper_perp, gripper_dz) are near
constant here; they are kept so the feature vector stays identical to
corrector_model.h, and the fit will simply ignore them.
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from apriltag_sim import make_tagged_env, TagDetector
import robosuite.utils.camera_utils as CU
import robosuite.utils.transform_utils as TU


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="robot0_eye_in_hand")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="opencv", choices=["opencv", "esp32"],
                   help="esp32 = the actual firmware detector (esp32_apriltag/), "
                        "cropped to the deployment ROI and decimated as on device")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    env, meta = make_tagged_env(camera=a.camera, width=a.width, height=a.height,
                                horizon=200)
    det = TagDetector(env, camera=a.camera, width=a.width, height=a.height,
                      marker_size_m=meta["marker_size_m"], backend=a.backend)
    np.random.seed(a.seed)
    rows, seen = [], 0
    for i in range(a.samples):
        od = env.reset(); seen += 1
        r = det.detect(od)
        if r is None:
            continue
        # The wrist camera MOVES with the arm, so its extrinsics must be read
        # per sample -- unlike agentview, which is a fixed constant.
        cam_pos = CU.get_camera_extrinsic_matrix(env.sim, a.camera)[:3, 3]
        tp = np.asarray(od["Bread_pos"]); ee = np.asarray(od["robot0_eef_pos"])
        dp = np.asarray(r[0]); dq = np.asarray(r[1])
        res = tp - dp
        ray = dp - cam_pos; rng = float(np.linalg.norm(ray)); ray = ray / max(rng, 1e-9)
        v = ee - cam_pos
        perp = float(np.linalg.norm(v - np.dot(v, ray) * ray))
        nrm = TU.quat2mat(dq)[:, 2]
        obliq = float(np.degrees(np.arccos(np.clip(abs(nrm @ ray), 0, 1))))
        yaw = float(np.arctan2(2*(dq[3]*dq[2]+dq[0]*dq[1]),
                               1-2*(dq[1]**2+dq[2]**2)))
        rows.append(dict(res_x=res[0], res_y=res[1], res_z=res[2],
                         res_norm=float(np.linalg.norm(res)),
                         reproj=det.last_err, area_px=det.last_area_px,
                         obliq_deg=obliq, cam_range=rng, gripper_perp=perp,
                         gripper_dz=float(ee[2] - dp[2]),
                         cx=float(det.last_corners[:, 0].mean()),
                         cy=float(det.last_corners[:, 1].mean()),
                         obj_yaw=yaw, det_x=dp[0], det_y=dp[1], det_z=dp[2]))
        if len(rows) % 100 == 0:
            print(f"  {len(rows)} detections / {seen} resets "
                  f"({len(rows)/seen*100:.0f}%)", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    e = np.array([r["res_norm"] for r in rows]) * 1000
    print(f"wrote {a.out}: {len(rows)}/{seen} detected ({len(rows)/seen*100:.1f}%), "
          f"raw error median {np.median(e):.1f} mm")


if __name__ == "__main__":
    main()
