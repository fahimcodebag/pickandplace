#!/usr/bin/env python3
"""Test script for the enhanced TD3 v2 model (goal-conditioned)."""

import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_v2 import Agent

BREAD_POS = slice(0, 3)


if __name__ == '__main__':
    env_name = "PickPlace"

    rs_env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE",
        ),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )

    # Fix bread spawn for deterministic visual testing
    _orig_gpi = rs_env._get_placement_initializer

    def _fixed_placement():
        _orig_gpi()
        s = rs_env.placement_initializer.samplers["CollisionObjectSampler"]
        s.x_range = np.array([0.0, 0.0])
        s.y_range = np.array([0.0, 0.0])
        s.rotation = 0.0
        s.ensure_object_boundary_in_range = False
        s.ensure_valid_placement = False

    rs_env._get_placement_initializer = _fixed_placement

    env = GymWrapper(rs_env)

    # Goal: target bin for bread
    goal = rs_env.target_bin_placements[
        rs_env.object_to_id['bread']
    ].copy()
    print(f"Goal (target bin): {goal}")

    agent = Agent(
        alpha=0.0005,
        beta=0.0005,
        obs_dims=env.observation_space.shape,
        goal_dim=3,
        tau=0.005,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=1024,
        layer2_size=512,
        batch_size=1024,
        n_critics=4,
        chkpt_dir='./checkpoints/td3_v2',
    )

    print("Loading trained models from ./checkpoints/td3_v2/ ...")
    try:
        agent.load_best_models()
        print("Best models loaded.\n")
    except FileNotFoundError:
        print("Best models not found. Loading latest models instead.")
        agent.load_models()
        print("Latest models loaded.\n")

    n_test_episodes = 5
    for ep in range(1, n_test_episodes + 1):
        obs = env.reset()
        done = False
        score = 0.0
        steps = 0

        while not done:
            action = agent.choose_action(obs, goal, validation=True)
            obs, reward, done, info = env.step(action)
            score += reward
            steps += 1
            rs_env.render()
            time.sleep(0.02)

        # Check success
        bread_pos = obs[BREAD_POS]
        dist_to_goal = np.linalg.norm(bread_pos - goal)
        success = dist_to_goal < 0.10

        try:
            success = success or rs_env._check_success()
        except Exception:
            pass

        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(
            f"Episode {ep}/{n_test_episodes} | "
            f"Score: {score:.2f} | Steps: {steps} | "
            f"Dist: {dist_to_goal:.3f} | {status}"
        )

    env.close()
    print("\nDone.")
