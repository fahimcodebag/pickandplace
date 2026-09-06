#!/usr/bin/env python3
"""Does the grasp policy align gripper yaw to object yaw, and does residual
misalignment predict the drops?

The task geometry says yaw should not matter for REACHABILITY -- the bread is
44x48 mm, diagonal 65 mm, inside the Panda's ~80 mm opening at any yaw. But it
matters for GRIP QUALITY: jaws closing on two flat faces have a large contact
patch, jaws closing on opposite corners have almost none, and a corner grip is
exactly the kind that survives a static lift and then slips under transport
acceleration.

A parallel-jaw gripper on a near-square cross-section is symmetric under 90
degrees, so the alignment error is folded into [-45, 45] deg. 0 = jaws square
to the faces, 45 = jaws on the corners.

Records the alignment error at the moment the grasp first closes.

The `dropped` column is NOT a valid drop measurement and is kept only so the
CSV matches what was run. This harness drives the grasp policy ALONE for 220
steps with no FSM, so after the grasp nothing takes over, the policy keeps
acting past its job and lets go: it reports ~83% "drops" against a real task
success of ~95%. Measuring whether misalignment causes drops needs the full
FSM (fsm_sim.py), not the bare policy.
"""
import argparse, csv, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch as T
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robosuite as suite
from robosuite.wrappers import GymWrapper
import robosuite.utils.transform_utils as TU
from fsm_sim import load_actor


def yaw_of(quat):
    """Z-rotation of a quaternion (xyzw), radians."""
    x, y, z, w = quat
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def fold(err):
    """Fold into [-45, 45] deg: the jaws are symmetric under 90 deg."""
    e = np.degrees(err) % 90.0
    return e - 90.0 if e > 45.0 else e


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp-ckpt", required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
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
    actor = load_actor(a.grasp_ckpt, 64, 32)
    lo, hi = env.action_space.low, env.action_space.high

    def grasped():
        try:
            return bool(raw._check_grasp(gripper=raw.robots[0].gripper,
                                         object_geoms=raw.objects[raw.object_id]))
        except Exception:
            return False

    rows = []
    for ep in range(a.episodes):
        obs = env.reset()
        align = None; lifted = False; dropped = False; hold = 0
        for t in range(220):
            with T.no_grad():
                act = actor(T.tensor(obs, dtype=T.float32)).numpy()
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            g = grasped()
            if g and align is None:
                # first closure: compare gripper yaw with object yaw
                oq = np.asarray(raw._observables["Bread_quat"].obs)
                eq = np.asarray(raw._observables["robot0_eef_quat"].obs)
                align = fold(yaw_of(eq) - yaw_of(oq))
                z0 = float(raw._observables["Bread_pos"].obs[2])
            if align is not None:
                # _check_grasp flickers on a stable grasp, so a single
                # ungrasped frame is not a drop. Use the FSM's own rule --
                # LOST_GRIP_STEPS = 5 consecutive -- or the metric reports
                # ~80% drops, which it did before this was fixed.
                hold = 0 if g else hold + 1
                z = float(raw._observables["Bread_pos"].obs[2])
                if z > z0 + 0.03:
                    lifted = True
                if lifted and hold >= 5:
                    dropped = True
            if done:
                break
        if align is not None:
            rows.append(dict(episode=ep, align_deg=round(align, 2),
                             lifted=int(lifted), dropped=int(dropped)))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    e = np.array([r["align_deg"] for r in rows])
    print(f"{a.out}: {len(rows)} grasps  |align| median {np.median(np.abs(e)):.1f} deg "
          f"p90 {np.percentile(np.abs(e),90):.1f}  dropped {sum(r['dropped'] for r in rows)}")


if __name__ == "__main__":
    main()
