#!/usr/bin/env python3
"""Test script for the built-in reward shaping model (train_vectorized_builtin.py)."""

import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3 import Agent


if __name__ == '__main__':
    env_name = "PickPlace"

    rs_env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Fix object spawn to match training (must override the method
    # because hard_reset=True re-calls _get_placement_initializer each reset)
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

    agent = Agent(
        alpha=0.0005,
        beta=0.0005,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=512,
        layer2_size=256,
        batch_size=1024,
        chkpt_dir='./checkpoints/td3_builtin',
    )

    print("Loading trained models from ./checkpoints/td3_builtin/ ...")
    agent.load_models()
    print("Models loaded.\n")

    n_test_episodes = 5
    for ep in range(1, n_test_episodes + 1):
        obs = env.reset()
        done = False
        score = 0.0
        steps = 0

        while not done:
            action = agent.choose_action(obs, validation=True)
            obs, reward, done, info = env.step(action)
            score += reward
            steps += 1
            rs_env.render()
            time.sleep(0.02)

        success = False
        try:
            success = rs_env._check_success()
        except Exception:
            pass

        status = "SUCCESS" if success else "FAIL"
        print(f"Episode {ep}/{n_test_episodes} | Score: {score:.2f} | Steps: {steps} | {status}")

    env.close()
    print("\nDone.")