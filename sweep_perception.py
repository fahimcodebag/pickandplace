#!/usr/bin/env python3
"""Perception-degradation sweep -> the thesis table for periodic perception.

Runs the deterministic grasp policy under graded perception degradation and
reports success rate per configuration. Uses the FITTED NOISE MODEL (no
rendering), which is both fast and scientifically cleaner: it isolates period,
latency, noise and dropout as independent variables instead of confounding them
inside one camera setup. Feed it the numbers measured by
`validate_tag_perception.py` (wrist camera @320x240 measured 5.6 mm median).

The `--mode` axis is the load-bearing ablation: "recompute" holds the world-frame
object pose and rebuilds the gripper-frame block from fresh proprioception every
step; "frozen" freezes the whole perception block, as a naive port would.

Usage:
  python3 sweep_perception.py --episodes 30
  python3 sweep_perception.py --episodes 30 --spawn random --grasp-ckpt checkpoints/td3_grasp_rand/best
"""

import argparse
import csv
import itertools
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "Decomposed state training"))
sys.path.insert(0, os.path.join(_HERE, "Decomposed state training",
                                "Random spawn model"))

import numpy as np
import torch

import robosuite as suite
from robosuite.wrappers import GymWrapper

from grasp_env_wrapper import GraspRewardWrapper
from networks import ActorNetwork
from perception_wrapper import PeriodicPerceptionWrapper
from grasp_spawn_wrapper import make_spawn_grasp_env


def make_env(spawn="fixed", seed=None, level=1.0):
    """Build the grasp env the sweep evaluates in.

    "random" MUST go through make_spawn_grasp_env, the same constructor the
    policy trained under and that eval_best.py scores with. Falling through to
    robosuite's stock PickPlace initializer instead randomises position AND
    full z-rotation -- that is Phase B (level 2.0) difficulty, not level 1.0,
    and it scored a level-1.0 policy at 6.7% with perception every step.
    """
    if spawn == "random":
        return make_spawn_grasp_env("PickPlace", seed=seed,
                                    curriculum=False, level=level)

    env = suite.make(
        "PickPlace", robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"),
        has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
        horizon=500, reward_shaping=False, control_freq=20,
        single_object_mode=2, object_type="bread")
    _orig = env._get_placement_initializer

    def _fixed():
        _orig()
        s = env.placement_initializer.samplers["CollisionObjectSampler"]
        s.x_range = np.array([0.0, 0.0])
        s.y_range = np.array([0.0, 0.0])
        s.rotation = 0.0
        s.ensure_object_boundary_in_range = False
        s.ensure_valid_placement = False

    env._get_placement_initializer = _fixed
    g = GymWrapper(GraspRewardWrapper(env))
    if seed is not None:
        g.seed(seed)
    return g


def evaluate(actor, cfg, episodes, spawn, seed, level=1.0):
    env = PeriodicPerceptionWrapper(make_env(spawn, seed, level),
                                    seed=seed, **cfg)
    dev = actor.device
    succ, steps_all = 0, []
    for _ in range(episodes):
        obs = env.reset()
        done, steps, info = False, 0, {}
        while not done and steps < 200:
            with torch.no_grad():
                a = actor(torch.tensor(obs, dtype=torch.float,
                                       device=dev).unsqueeze(0))
            obs, _, done, info = env.step(a.squeeze(0).cpu().numpy())
            steps += 1
        succ += bool(info.get("grasp_success", False))
        steps_all.append(steps)
    st = env.stats
    env.env.close()
    return succ, float(np.mean(steps_all)), st


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--spawn", choices=["fixed", "random"], default="fixed")
    p.add_argument("--grasp-ckpt", default="checkpoints/td3_grasp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", type=float, default=1.0,
                   help="spawn level for --spawn random; 1.0 = full position box, 2.0 adds z-rotation")
    p.add_argument("--out", default="Results/perception_sweep.csv")
    p.add_argument("--periods", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--noises", type=float, nargs="+",
                   default=[0.0, 0.002, 0.005, 0.010])
    p.add_argument("--dropouts", type=float, nargs="+", default=[0.0])
    p.add_argument("--latencies", type=int, nargs="+", default=[0])
    p.add_argument("--modes", nargs="+", default=["recompute", "frozen"])
    args = p.parse_args()

    actor = ActorNetwork(46, 64, 32, 7, chkpt_dir=args.grasp_ckpt)
    actor.load_checkpoint()
    actor.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    f = open(args.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["mode", "period", "latency", "noise_pos_m", "dropout",
                "episodes", "successes", "success_pct", "mean_steps",
                "perception_ticks", "detections", "dropouts"])

    print(f"grasp policy : {args.grasp_ckpt}")
    print(f"spawn        : {args.spawn}   episodes/config: {args.episodes}\n")
    print(f"{'mode':11s} {'per':>4s} {'lat':>4s} {'noise':>7s} {'drop':>5s}"
          f" {'success':>9s} {'steps':>7s}")
    print("-" * 54)

    for mode, per, lat, noi, dro in itertools.product(
            args.modes, args.periods, args.latencies, args.noises,
            args.dropouts):
        if mode == "frozen" and per == 1:
            continue          # identical to recompute at period 1
        cfg = dict(period=per, latency=lat, noise_pos=noi, dropout=dro,
                   mode=mode)
        succ, steps, st = evaluate(actor, cfg, args.episodes, args.spawn,
                                   args.seed, args.level)
        pct = 100.0 * succ / args.episodes
        print(f"{mode:11s} {per:4d} {lat:4d} {noi:7.3f} {dro:5.2f}"
              f" {succ:4d}/{args.episodes:<4d} {pct:5.1f}% {steps:7.1f}")
        w.writerow([mode, per, lat, noi, dro, args.episodes, succ,
                    f"{pct:.1f}", f"{steps:.1f}", st["ticks"],
                    st["detections"], st["dropouts"]])
        f.flush()

    f.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
