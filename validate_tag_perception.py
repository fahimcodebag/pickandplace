#!/usr/bin/env python3
"""Validate virtual AprilTag perception against MuJoCo ground truth.

Renders the tagged scene, detects the tag, solves its pose, and scores the
result against the simulator's true object pose. This is the measurement that
replaces "I assumed 5 mm noise" with a number, and it is also the end-to-end
check on the frame conventions (an inverted axis produces plausible-looking
poses that are silently mirrored, so it must be caught here).
"""

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np

from apriltag_sim import make_tagged_env, TagDetector


def tag_world_pose_truth(env, z_offset):
    """True tag-centre pose from MuJoCo: object pose composed with the offset."""
    obj = env.sim.data.body_xpos[env.sim.model.body_name2id("Bread_main")]
    R = env.sim.data.body_xmat[
        env.sim.model.body_name2id("Bread_main")].reshape(3, 3)
    return obj + R @ np.array([0.0, 0.0, z_offset]), R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="agentview")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--z-offset", type=float, default=0.026)
    p.add_argument("--half-size", type=float, default=0.022)
    p.add_argument("--save-frame", default="/tmp/tag_frame.png")
    args = p.parse_args()

    env, meta = make_tagged_env(camera=args.camera, width=args.width,
                                height=args.height, half_size=args.half_size,
                                z_offset=args.z_offset)
    det = TagDetector(env, camera=args.camera, width=args.width,
                      height=args.height,
                      marker_size_m=meta["marker_size_m"])

    print(f"camera={args.camera}  {args.width}x{args.height}  "
          f"marker={meta['marker_size_m']*100:.2f} cm")
    print("intrinsics K:\n", np.array2string(det.K, precision=1))
    print()

    errs, hits = [], 0
    for ep in range(args.episodes):
        obs = env.reset()
        result = det.detect(obs)
        truth_pos, _ = tag_world_pose_truth(env, args.z_offset)

        if ep == 0:
            cv2.imwrite(args.save_frame,
                        cv2.cvtColor(det.frame(obs), cv2.COLOR_RGB2BGR))

        if result is None:
            print(f"  ep{ep:2d}  NO DETECTION   truth={np.round(truth_pos,3)}")
            continue
        hits += 1
        pos, quat = result
        err = np.linalg.norm(pos - truth_pos)
        errs.append(err)
        print(f"  ep{ep:2d}  est={np.round(pos,4)}  truth={np.round(truth_pos,4)}"
              f"  err={err*1000:6.1f} mm")

    print(f"\nDetection rate : {hits}/{args.episodes}")
    if errs:
        e = np.array(errs) * 1000
        print(f"Position error : mean {e.mean():.1f} mm | median "
              f"{np.median(e):.1f} mm | p95 {np.percentile(e,95):.1f} mm | "
              f"max {e.max():.1f} mm")
    print(f"Frame saved    : {args.save_frame}")
    env.close()


if __name__ == "__main__":
    main()
