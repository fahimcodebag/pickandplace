#!/usr/bin/env python3
# Last updated: 2026-07-07 13:41 +0600
"""
Evaluation script for the trained Place sub-policy.

Tests the full pipeline: grasp model → place model in sequence.
Each episode runs the grasp model first (via PlaceRewardWrapper),
then evaluates the place model's ability to lift, transport, and place.

Usage:
    python test_place.py                    # 10 episodes, no render
    python test_place.py --episodes 20      # 20 episodes
    python test_place.py --render           # with on-screen rendering
"""

import argparse
import os
import sys
import time

import numpy as np

# Add parent directory for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3 import Agent
from place_env_wrapper import PlaceGymWrapper


def make_place_eval_env(render=False, grasp_chkpt_dir=None,
                        spawn_range=None, native_spawn=False):
    """Create a single robosuite environment with place reward for evaluation.

    Spawn modes (generalization testing — training used a FIXED spawn):
      default:            fixed spawn at the nominal training pose
      spawn_range=R:      uniform +/-R meters in x and y around it, rotation 0
      native_spawn=True:  robosuite's own full randomization over the source
                          bin, including rotation (the hardest setting)
    """
    if grasp_chkpt_dir is None:
        grasp_chkpt_dir = os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "td3_grasp"
        )

    env = suite.make(
        "PickPlace",
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=700,  # match training: grasp rollout + test-lift + place + release
        reward_shaping=False,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Spawn control. native_spawn leaves robosuite's sampler untouched (full
    # bin-area randomization + rotation); otherwise patch it to the nominal
    # training pose, widened by +/-spawn_range meters when requested.
    if not native_spawn:
        r = float(spawn_range) if spawn_range else 0.0
        _orig_gpi = env._get_placement_initializer
        def _fixed_placement():
            _orig_gpi()
            s = env.placement_initializer.samplers["CollisionObjectSampler"]
            s.x_range = np.array([-r, r])
            s.y_range = np.array([-r, r])
            s.rotation = 0.0
            s.ensure_object_boundary_in_range = False
            s.ensure_valid_placement = False
        env._get_placement_initializer = _fixed_placement

    raw_env = env
    gym_env = GymWrapper(raw_env)
    # curriculum=False is ESSENTIAL for honest evaluation: with it on (the
    # training default), reset() scripted-carries the object most of the way
    # to the bin and the "success rate" measures almost nothing.
    place_env = PlaceGymWrapper(gym_env, raw_env, grasp_chkpt_dir,
                                curriculum=False)
    return place_env, raw_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Place sub-policy")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true",
                        help="Enable on-screen rendering")
    parser.add_argument("--best", action="store_true",
                        help="Load the difficulty-weighted best policy from "
                             "checkpoints/td3_place/best instead of the live "
                             "(latest) checkpoint")
    parser.add_argument("--grasp-chkpt-dir", type=str, default=None,
                        help="Grasp checkpoint dir. Was hardcoded to "
                             "checkpoints/td3_grasp, which is the ORIGINAL "
                             "fixed-spawn model -- the end-to-end ladder has to "
                             "be re-run whenever the grasp stage changes.")
    parser.add_argument("--chkpt-dir", type=str, default=None,
                        help="Explicit place checkpoint directory (overrides --best)")
    # Deployment parameters (FSM rules, not learned). Defaults are the LOCKED
    # values from the parameter sweep that took the best policy from 78% to
    # 92% end-to-end (trigger 0.10->0.14, hold 5->3, horizon 200->300): the
    # policy delivers NEAR the bin and the scripted release's recenter phase
    # absorbs the wider trigger offset. Training-time wrapper values (0.10/5/
    # 200) are unchanged — these overrides apply to evaluation only.
    parser.add_argument("--trigger-xy", type=float, default=0.14,
                        help="Release trigger radius in m (locked sweep value)")
    parser.add_argument("--trigger-hold", type=int, default=3,
                        help="Consecutive over-bin steps to trigger release")
    parser.add_argument("--place-horizon", type=int, default=300,
                        help="Max place-phase steps")
    parser.add_argument("--random-spawn", action="store_true",
                        help="Robosuite-native full spawn randomization over the "
                             "source bin (position + rotation). Hardest setting.")
    parser.add_argument("--spawn-range", type=float, default=None,
                        help="Uniform +/-R meters of x/y spawn noise around the "
                             "nominal training pose (rotation fixed). Graded "
                             "generalization test, e.g. 0.02, 0.05.")
    args = parser.parse_args()

    place_chkpt_dir = os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "td3_place"
    )
    if args.best:
        place_chkpt_dir = os.path.join(place_chkpt_dir, "best")
    if args.chkpt_dir is not None:
        place_chkpt_dir = args.chkpt_dir
    grasp_chkpt_dir = args.grasp_chkpt_dir or os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "td3_grasp"
    )

    print(f"\n{'=' * 70}")
    print("PLACE MODEL EVALUATION (Full Pipeline: Grasp → Place)")
    print(f"{'=' * 70}")
    print(f"  Episodes:        {args.episodes}")
    print(f"  Render:          {args.render}")
    print(f"  Grasp checkpoint: {grasp_chkpt_dir}")
    print(f"  Place checkpoint: {place_chkpt_dir}")
    print(f"{'=' * 70}\n")

    # --- Create environment -------------------------------------------------
    env, raw_env = make_place_eval_env(render=args.render,
                                        grasp_chkpt_dir=grasp_chkpt_dir,
                                        spawn_range=args.spawn_range,
                                        native_spawn=args.random_spawn)
    if args.random_spawn:
        spawn_desc = "NATIVE (full bin randomization + rotation)"
    elif args.spawn_range:
        spawn_desc = f"nominal +/- {args.spawn_range} m (rotation fixed)"
    else:
        spawn_desc = "FIXED (training pose)"
    print(f"  Object spawn:     {spawn_desc}")

    # Render the scripted phases too (grasp rollout, test-lift, release run
    # inside reset()/step(), which the loop below can't render itself).
    if args.render:
        env._render_scripted = True

    # Apply deployment-parameter overrides (instance attrs shadow the class
    # constants; training code is untouched).
    if args.trigger_xy is not None:
        env._NEAR_TARGET_XY = args.trigger_xy
    if args.trigger_hold is not None:
        env._RELEASE_TRIGGER_HOLD = args.trigger_hold
    if args.place_horizon is not None:
        env.PLACE_HORIZON = args.place_horizon
    print(f"  Release trigger:  xy<={env._NEAR_TARGET_XY}m "
          f"for {env._RELEASE_TRIGGER_HOLD} steps | "
          f"place horizon: {env.PLACE_HORIZON}")

    # --- Create place agent -------------------------------------------------
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
        chkpt_dir=place_chkpt_dir,
    )

    print("Loading trained place model...")
    agent.load_models()
    print("Models loaded.\n")

    # --- Evaluation loop ----------------------------------------------------
    successes = 0
    scores = []
    steps_list = []
    reasons = []

    for i in range(args.episodes):
        # reset() runs the grasp policy internally
        observation = env.reset()
        done = False
        score = 0
        step = 0

        print(f"Episode {i + 1}/{args.episodes}", end=" ", flush=True)

        while not done:
            action = agent.choose_action(observation, validation=True)
            observation, reward, done, info = env.step(action)
            if args.render:
                env.render()
                time.sleep(0.02)
            score += reward
            step += 1

        place_ok = info.get("place_success", False)
        if place_ok:
            successes += 1

        scores.append(score)
        steps_list.append(step)

        status = "✓ PLACED" if place_ok else "✗ failed"
        reason = info.get("place_done_reason", "?")
        reasons.append(reason)
        print(f"| Steps: {step:3d} | Score: {score:8.2f} | {status} ({reason})")

    # --- Aggregate results --------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RESULTS (End-to-End: Grasp → Place)")
    print(f"{'=' * 70}")
    print(f"  Place success:    {successes}/{args.episodes} "
          f"({successes / args.episodes * 100:.1f}%)")
    print(f"  Mean score:       {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  Mean steps:       {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    from collections import Counter
    tally = "  ".join(f"{k}:{v}" for k, v in Counter(reasons).most_common())
    print(f"  Outcomes:         {tally}")
    print(f"{'=' * 70}\n")

    env.close()


if __name__ == "__main__":
    main()
