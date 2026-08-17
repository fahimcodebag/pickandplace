#!/usr/bin/env python3
# Last updated: 2026-07-04 23:05 +0600
"""
Evaluation for the random-spawn grasp sub-policy.

Measures grasp success (stable N-step grip, same criterion as training) at a
chosen spawn setting, deterministically (no exploration noise).

Usage:
    python3 test_grasp_rand.py --episodes 30 --best --random-spawn
    python3 test_grasp_rand.py --episodes 30 --best --spawn-range 0.05
    python3 test_grasp_rand.py --episodes 30 --best --level 1.0
    python3 test_grasp_rand.py --chkpt-dir ../../checkpoints/td3_grasp \
        --random-spawn        # baseline: the ORIGINAL fixed-spawn model
"""

import argparse
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from td3 import Agent
from grasp_spawn_wrapper import (
    make_spawn_grasp_env,
    SpawnCurriculumGraspWrapper,
)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the random-spawn grasp sub-policy")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--best", action="store_true",
                        help="Load checkpoints/td3_grasp_rand/best instead "
                             "of the live checkpoint")
    parser.add_argument("--chkpt-dir", type=str, default=None,
                        help="Explicit checkpoint dir (overrides --best); "
                             "e.g. the original fixed-spawn model for a "
                             "baseline comparison")
    parser.add_argument("--random-spawn", action="store_true",
                        help="Native full randomization: whole bin + rotation "
                             "(= spawn level 2.0, the training ceiling)")
    parser.add_argument("--spawn-range", type=float, default=None,
                        help="Uniform +/-R meters x/y around the nominal "
                             "pose, rotation fixed (graded test)")
    parser.add_argument("--level", type=float, default=None,
                        help="Spawn curriculum level 0.1-2.0 (see wrapper)")
    args = parser.parse_args()

    chkpt_dir = os.path.join(_HERE, "..", "..", "checkpoints", "td3_grasp_rand")
    if args.best:
        chkpt_dir = os.path.join(chkpt_dir, "best")
    if args.chkpt_dir is not None:
        chkpt_dir = args.chkpt_dir

    # Spawn setting: static spec (meters) > level > default fixed pose.
    static_spec = None
    level = 0.0
    if args.random_spawn:
        level = SpawnCurriculumGraspWrapper._LEVEL_MAX
        spawn_desc = "NATIVE (full bin + rotation, level 2.0)"
    elif args.spawn_range is not None:
        r = float(args.spawn_range)
        static_spec = {"x": (-r, r), "y": (-r, r), "rot": 0.0}
        spawn_desc = f"nominal +/- {r} m (rotation fixed)"
    elif args.level is not None:
        level = float(args.level)
        spawn_desc = f"curriculum level {level}"
    else:
        static_spec = {"x": (0.0, 0.0), "y": (0.0, 0.0), "rot": 0.0}
        spawn_desc = "FIXED (nominal training pose)"

    print(f"\n{'=' * 70}")
    print("RANDOM-SPAWN GRASP MODEL EVALUATION")
    print(f"{'=' * 70}")
    print(f"  Episodes:     {args.episodes}")
    print(f"  Checkpoint:   {chkpt_dir}")
    print(f"  Object spawn: {spawn_desc}")
    print(f"{'=' * 70}\n")

    env = make_spawn_grasp_env(render=args.render, curriculum=False,
                               level=level, static_spec=static_spec)

    agent = Agent(
        alpha=0.0003,
        beta=0.0003,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=64,
        layer2_size=32,
        batch_size=512,
        chkpt_dir=chkpt_dir,
    )
    print("Loading grasp model...")
    agent.load_models()
    print("Model loaded.\n")

    successes = 0
    steps_list = []

    for i in range(args.episodes):
        observation = env.reset()
        done = False
        step = 0
        print(f"Episode {i + 1}/{args.episodes}", end=" ", flush=True)
        while not done:
            action = agent.choose_action(observation, validation=True)
            observation, reward, done, info = env.step(action)
            if args.render:
                env.render()
                time.sleep(0.02)
            step += 1
        ok = info.get("grasp_success", False)
        if ok:
            successes += 1
        steps_list.append(step)
        status = "✓ GRASPED" if ok else "✗ failed"
        print(f"| Steps: {step:3d} | {status}")

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Grasp success: {successes}/{args.episodes} "
          f"({successes / args.episodes * 100:.1f}%)")
    print(f"  Mean steps:    {np.mean(steps_list):.1f} "
          f"± {np.std(steps_list):.1f}")
    print(f"{'=' * 70}\n")

    env.close()


if __name__ == "__main__":
    main()
