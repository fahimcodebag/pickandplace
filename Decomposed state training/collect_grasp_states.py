#!/usr/bin/env python3
# Last updated: 2026-06-30
"""
Collect grasp terminal states for Place model training.

Runs the trained grasp policy for N episodes and saves the MuJoCo sim state
+ observation at each successful grasp termination. These states serve as
realistic initial conditions for training the Place sub-policy, avoiding
distribution shift at the handoff boundary.

Usage:
    python collect_grasp_states.py                   # 1000 episodes
    python collect_grasp_states.py --episodes 500    # custom count
    python collect_grasp_states.py --output my_states.pkl

Output:
    A pickle file containing a list of dicts, each with:
      - 'sim_state': MuJoCo MjSimState (qpos, qvel, act, udd_state)
      - 'observation': flat observation vector (46-dim)
      - 'obs_dict': raw robosuite observation dict
      - 'episode': source episode number
      - 'steps': steps taken to achieve grasp

Note: This script is only runnable AFTER the grasp model has been trained.
      It will fail if no checkpoint exists in ./checkpoints/td3_grasp/.
"""

import argparse
import os
import pickle
import sys

import numpy as np

# Add parent directory to path for existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3 import Agent
from grasp_env_wrapper import GraspRewardWrapper


def make_collection_env():
    """Create environment for state collection (no rendering)."""
    env = suite.make(
        "PickPlace",
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        reward_shaping=False,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Fix object spawn (must match training)
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

    return env


def collect_states(n_episodes=1000, output_path=None):
    """
    Run trained grasp policy and collect terminal states from successful grasps.

    Returns a list of state dicts saved to a pickle file.
    """
    chkpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "td3_grasp")
    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), "grasp_terminal_states.pkl")

    print(f"\n{'=' * 70}")
    print("GRASP TERMINAL STATE COLLECTION")
    print(f"{'=' * 70}")
    print(f"  Episodes:    {n_episodes}")
    print(f"  Checkpoints: {chkpt_dir}")
    print(f"  Output:      {output_path}")
    print(f"{'=' * 70}\n")

    # --- Create raw env (no GymWrapper for full obs_dict access) ------------
    raw_env = make_collection_env()
    grasp_env = GraspRewardWrapper(raw_env)
    gym_env = GymWrapper(grasp_env)

    # --- Create agent -------------------------------------------------------
    agent = Agent(
        alpha=0.0003,
        beta=0.0003,
        tau=0.005,
        input_dims=gym_env.observation_space.shape,
        env=gym_env,
        n_actions=gym_env.action_space.shape[0],
        layer1_size=64,
        layer2_size=32,
        batch_size=512,
        chkpt_dir=chkpt_dir,
    )

    print("Loading trained grasp model...")
    agent.load_models()
    print("Models loaded.\n")

    # --- Collection loop ----------------------------------------------------
    collected_states = []
    successes = 0
    failures = 0

    for ep in range(n_episodes):
        observation = gym_env.reset()
        done = False
        step = 0

        while not done:
            action = agent.choose_action(observation, validation=True)
            observation, reward, done, info = gym_env.step(action)
            step += 1

        grasp_ok = info.get("grasp_success", False)

        if grasp_ok:
            successes += 1
            # Capture the MuJoCo sim state at grasp termination
            try:
                sim_state = raw_env.sim.get_state()
                # Also capture the raw obs_dict for richer downstream use
                obs_dict = raw_env._get_observations()

                state_record = {
                    "sim_state": sim_state,
                    "observation": observation.copy(),
                    "obs_dict": {k: np.array(v).copy() if hasattr(v, 'copy') else v
                                 for k, v in obs_dict.items()},
                    "episode": ep,
                    "steps": step,
                }
                collected_states.append(state_record)
            except Exception as e:
                print(f"  [WARNING] Episode {ep}: Failed to capture sim state: {e}")
        else:
            failures += 1

        # Progress reporting
        if (ep + 1) % 100 == 0:
            total = successes + failures
            print(
                f"  Episode {ep + 1}/{n_episodes} | "
                f"Collected: {len(collected_states)} | "
                f"Success: {successes}/{total} ({successes / total * 100:.1f}%)"
            )

    # --- Save collected states ----------------------------------------------
    print(f"\nSaving {len(collected_states)} terminal states...")
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(collected_states, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{'=' * 70}")
    print("COLLECTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total episodes:     {n_episodes}")
    print(f"  Successful grasps:  {successes} ({successes / n_episodes * 100:.1f}%)")
    print(f"  Failed grasps:      {failures} ({failures / n_episodes * 100:.1f}%)")
    print(f"  States collected:   {len(collected_states)}")
    print(f"  Saved to:           {output_path}")
    print(f"  File size:          {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"{'=' * 70}\n")

    print("Next step:")
    print("  Use these states as initial conditions for Place model training:")
    print("    states = pickle.load(open('grasp_terminal_states.pkl', 'rb'))")
    print("    sim.set_state(states[i]['sim_state'])")

    gym_env.close()
    return collected_states


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect grasp terminal states for Place model")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Number of episodes to run (default: 1000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output pickle file path (default: grasp_terminal_states.pkl)")
    args = parser.parse_args()

    collect_states(n_episodes=args.episodes, output_path=args.output)
