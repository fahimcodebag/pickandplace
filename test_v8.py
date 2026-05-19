#!/usr/bin/env python3
"""Test script for the TD3 v8 model (curriculum trained)."""

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

    # Full random spawn (no fixed position — test generalisation)
    env = GymWrapper(rs_env)

    goal = rs_env.target_bin_placements[
        rs_env.object_to_id['bread']
    ].copy()
    print(f"Goal (target bin): {goal}")

    agent = Agent(
        alpha=0.0003,
        beta=0.0003,
        obs_dims=env.observation_space.shape,
        goal_dim=3,
        tau=0.005,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=2048,
        layer2_size=1024,
        batch_size=1024,
        n_critics=4,
        chkpt_dir='./checkpoints/td3_v8',
    )

    print("Loading trained models from ./checkpoints/td3_v8/ ...")
    try:
        agent.load_models()
        print("Latest models loaded.\n")
    except FileNotFoundError:
        print("Latest not found. Trying best models...")
        agent.load_best_models()
        print("Best models loaded.\n")

    n_test_episodes = 10
    successes = 0
    scores = []

    for ep in range(1, n_test_episodes + 1):
        obs = env.reset()
        bread_start = obs[BREAD_POS].copy()
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

        bread_pos = obs[BREAD_POS]
        dist_to_goal = np.linalg.norm(bread_pos - goal)
        success = dist_to_goal < 0.10

        try:
            success = success or rs_env._check_success()
        except Exception:
            pass

        if success:
            successes += 1
        scores.append(score)

        status = "✓ SUCCESS" if success else "✗ FAIL"
        print(
            f"Episode {ep}/{n_test_episodes} | "
            f"Score: {score:.2f} | Steps: {steps} | "
            f"Dist: {dist_to_goal:.3f} | {status}"
        )

    env.close()
    print(f"\nResults: {successes}/{n_test_episodes} successful")
    print(f"Average score: {np.mean(scores):.2f}")
    print("Done.")
