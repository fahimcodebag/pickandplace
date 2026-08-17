#!/usr/bin/env python3
# Last updated: 2026-07-01
"""
Evaluation script for the trained Grasp sub-policy.

Loads the grasp model from ./checkpoints/td3_grasp/ and runs evaluation
episodes, reporting per-episode and aggregate grasp success metrics.

Usage:
    python test_grasp.py                    # 10 episodes, no render
    python test_grasp.py --episodes 20      # 20 episodes
    python test_grasp.py --render           # with on-screen rendering
    python test_grasp.py --collect-states   # also save terminal states
"""

import argparse
import os
import sys
import time

import numpy as np

# Add parent directory to path for existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3 import Agent
from grasp_env_wrapper import GraspRewardWrapper


def make_grasp_eval_env(render=False):
    """Create a single robosuite environment with grasp-only reward for evaluation."""
    env = suite.make(
        "PickPlace",
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        reward_shaping=False,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Fix object spawn to constant position (must match training)
    _orig_gpi = env._get_placement_initializer
    def _fixed_placement():
        _orig_gpi()
        s = env.placement_initializer.samplers["CollisionObjectSampler"]
        s.x_range = np.array([0.0, 0.0])
        s.y_range = np.array([0.0, 0.0])
        s.rotation = 0.0
        s.ensure_object_boundary_in_range = False
        s.ensure_valid_placement = False
    env._get_placement_initializer = _fixed_placement

    # Keep raw env reference for state collection
    raw_env = env
    env = GraspRewardWrapper(env)
    env = GymWrapper(env)
    return env, raw_env


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Grasp sub-policy")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering")
    parser.add_argument("--collect-states", action="store_true",
                        help="Save terminal states for Place model training")
    args = parser.parse_args()

    chkpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "td3_grasp")

    print(f"\n{'=' * 70}")
    print("GRASP MODEL EVALUATION")
    print(f"{'=' * 70}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Render:      {args.render}")
    print(f"  Checkpoints: {chkpt_dir}")
    print(f"{'=' * 70}\n")

    # --- Create environment -------------------------------------------------
    env, raw_env = make_grasp_eval_env(render=args.render)

    # --- Create agent with same architecture as training --------------------
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

    print("Loading trained grasp model...")
    agent.load_models()
    print("Models loaded.\n")

    # --- Evaluation loop ----------------------------------------------------
    successes = 0
    total_steps_to_grasp = []
    scores = []
    terminal_states = []    # for --collect-states
    terminal_obs = []

    for i in range(args.episodes):
        observation = env.reset()
        done = False
        score = 0
        step = 0
        grasp_ok = False

        print(f"Episode {i + 1}/{args.episodes}", end=" ", flush=True)

        while not done:
            action = agent.choose_action(observation, validation=True)
            observation, reward, done, info = env.step(action)
            if args.render:
                env.render()
                time.sleep(0.02)
            score += reward
            step += 1

        grasp_ok = info.get("grasp_success", False)
        if grasp_ok:
            successes += 1
            total_steps_to_grasp.append(step)

            # Collect terminal state if requested
            if args.collect_states:
                try:
                    sim_state = raw_env.sim.get_state()
                    terminal_states.append(sim_state)
                    terminal_obs.append(observation)
                except Exception as e:
                    print(f" [WARNING: Could not save sim state: {e}]", end="")

        scores.append(score)
        status = "✓ GRASP" if grasp_ok else "✗ failed"
        print(f"| Steps: {step:3d} | Score: {score:8.2f} | {status}")

    # --- Aggregate results --------------------------------------------------
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Grasp success:     {successes}/{args.episodes} ({successes / args.episodes * 100:.1f}%)")
    print(f"  Mean score:        {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    if total_steps_to_grasp:
        print(f"  Mean steps to grasp: {np.mean(total_steps_to_grasp):.1f} ± {np.std(total_steps_to_grasp):.1f}")
    print(f"{'=' * 70}\n")

    # --- Save terminal states if collected ----------------------------------
    if args.collect_states and terminal_obs:
        save_path = os.path.join(os.path.dirname(__file__), "grasp_terminal_states.npz")
        np.savez_compressed(
            save_path,
            observations=np.array(terminal_obs),
            n_states=len(terminal_obs),
        )
        print(f"Saved {len(terminal_obs)} terminal observations to {save_path}")
        print("(MuJoCo sim states cannot be saved as numpy — use collect_grasp_states.py for full sim state collection)")

    env.close()


if __name__ == "__main__":
    main()
