#!/usr/bin/env python3
"""Test script for the vision-based TD3 model (train_vision.py).

Loads the trained CNN-based actor and runs evaluation episodes
with the robosuite renderer visible.
"""

import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_vision import VisionAgent


# Camera config must match training
IMG_HEIGHT = 84
IMG_WIDTH = 84
CAMERA_NAME = "agentview"
IMG_SHAPE = (3, IMG_HEIGHT, IMG_WIDTH)


def obs_to_img(obs_flat, height=IMG_HEIGHT, width=IMG_WIDTH):
    """Convert GymWrapper flat observation to (C, H, W) uint8."""
    img = obs_flat[:height * width * 3].reshape(height, width, 3)
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    return np.transpose(img_uint8, (2, 0, 1))


if __name__ == '__main__':
    env_name = "PickPlace"

    rs_env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE",
        ),
        has_renderer=True,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=CAMERA_NAME,
        camera_heights=IMG_HEIGHT,
        camera_widths=IMG_WIDTH,
        horizon=500,
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )

    # Fix object spawn to match training
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

    env = GymWrapper(rs_env, keys=[CAMERA_NAME + "_image"])

    # Get goal
    goal = rs_env.target_bin_placements[
        rs_env.object_to_id['bread']
    ].copy()

    # Create agent (same architecture as training)
    agent = VisionAgent(
        alpha=0.0001,
        beta=0.0003,
        img_shape=IMG_SHAPE,
        goal_dim=3,
        tau=0.005,
        env=env,
        n_actions=env.action_space.shape[0],
        latent_dim=256,
        fc1_dims=512,
        fc2_dims=256,
        batch_size=128,
        chkpt_dir='./checkpoints/td3_vision',
    )

    print("Loading trained vision models from ./checkpoints/td3_vision/ ...")
    agent.load_models()
    print("Models loaded.\n")

    n_test_episodes = 5
    for ep in range(1, n_test_episodes + 1):
        obs_flat = env.reset()
        img = obs_to_img(obs_flat)
        done = False
        score = 0.0
        steps = 0

        while not done:
            action = agent.choose_action(img, goal, validation=True)
            obs_flat, reward, done, info = env.step(action)
            img = obs_to_img(obs_flat)
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
        print(f"Episode {ep}/{n_test_episodes} | "
              f"Score: {score:.2f} | Steps: {steps} | {status}")

    env.close()
    print("\nDone.")
