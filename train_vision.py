#!/usr/bin/env python3
"""Vision-based training loop for TD3 with camera feedback.

Uses robosuite camera observations (84×84 RGB) instead of state vectors.
The camera image is the primary observation; proprioceptive state is NOT
used. The goal (target bin XYZ) is still provided as a low-dimensional
vector.

Key differences from train_v2.py:
  - Environment created with use_camera_obs=True, has_offscreen_renderer=True
  - Observations are images (H, W, 3) converted to (3, H, W) uint8
  - CNN-based actor/critic networks (networks_vision.py)
  - Image replay buffer stores uint8 to save memory (buffer_vision.py)
  - Smaller buffer (500K) and batch size (128) due to image memory cost
  - Fewer parallel envs (4) since offscreen rendering is expensive

Camera configuration:
  - Camera: "agentview" (overhead angled view of the workspace)
  - Resolution: 84×84 pixels
  - Channels: 3 (RGB)
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import multiprocessing as mp
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import torch
from torch.utils.tensorboard import SummaryWriter

from td3_vision import VisionAgent
from utils_rl import compute_reward, compute_staged_reward


# ---------------------------------------------------------------------------
# Camera / image configuration
# ---------------------------------------------------------------------------
IMG_HEIGHT = 84
IMG_WIDTH = 84
CAMERA_NAME = "agentview"
IMG_SHAPE = (3, IMG_HEIGHT, IMG_WIDTH)  # (C, H, W)

# Observation indices in the state vector (used to extract bread_pos
# for reward computation even though the agent sees only images)
BREAD_POS = slice(0, 3)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(env_name="PickPlace", seed=None):
    """Create a robosuite environment with camera observations enabled.

    Returns both the raw robosuite env (for internal state access) and
    the GymWrapper (for step/reset interface). The GymWrapper observation
    will contain both camera pixels and robot state.
    """
    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE",
        ),
        has_renderer=False,
        has_offscreen_renderer=True,    # required for camera obs
        use_camera_obs=True,            # enable camera feedback
        camera_names=CAMERA_NAME,
        camera_heights=IMG_HEIGHT,
        camera_widths=IMG_WIDTH,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    gym_env = GymWrapper(env, keys=[CAMERA_NAME + "_image"])
    if seed is not None:
        gym_env.seed(seed)
    return env, gym_env


def obs_to_img(obs_flat, height=IMG_HEIGHT, width=IMG_WIDTH):
    """Convert GymWrapper flat observation back to (C, H, W) uint8 image.

    GymWrapper flattens the camera image to (H*W*3,) float64 in [0, 255].
    We reshape and convert to (3, H, W) uint8 for storage.
    """
    img = obs_flat[:height * width * 3].reshape(height, width, 3)
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    # HWC → CHW (PyTorch convention)
    return np.transpose(img_uint8, (2, 0, 1))


# ---------------------------------------------------------------------------
# Subprocess vectorised environment
# ---------------------------------------------------------------------------

def _worker(remote, parent_remote, env_name, seed):
    """Worker process: creates an environment and processes commands."""
    parent_remote.close()
    raw_env, gym_env = make_single_env(env_name, seed=seed)
    goal = raw_env.target_bin_placements[
        raw_env.object_to_id['bread']
    ].copy()

    # We also need access to the underlying state for reward computation
    # So we wrap step/reset to return both image and state info
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            obs_flat, reward, done, info = gym_env.step(data)
            # Get the full state observation for reward computation
            obs_dict = raw_env._get_observations()
            state_obs = np.concatenate([
                obs_dict[k].flatten()
                for k in obs_dict
                if k != CAMERA_NAME + "_image" and isinstance(obs_dict[k], np.ndarray)
            ])
            img = obs_to_img(obs_flat)
            if done:
                obs_flat = gym_env.reset()
                img = obs_to_img(obs_flat)
            remote.send((img, state_obs, reward, done, info))

        elif cmd == "reset":
            obs_flat = gym_env.reset()
            obs_dict = raw_env._get_observations()
            state_obs = np.concatenate([
                obs_dict[k].flatten()
                for k in obs_dict
                if k != CAMERA_NAME + "_image" and isinstance(obs_dict[k], np.ndarray)
            ])
            img = obs_to_img(obs_flat)
            remote.send((img, state_obs))

        elif cmd == "close":
            gym_env.close()
            remote.close()
            break

        elif cmd == "get_spaces":
            remote.send((gym_env.observation_space, gym_env.action_space))

        elif cmd == "get_goal":
            remote.send(goal)


class VisionSubprocVecEnv:
    """Subprocess-parallel vectorised environment for vision-based training."""

    def __init__(self, env_name, n_envs=4):
        self.n_envs = n_envs
        self.closed = False
        ctx = mp.get_context("fork")

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(n_envs)]
        )

        print(f"Spawning {n_envs} vision subprocess environments (fork)...")
        self.processes = []
        for i, (work_remote, remote) in enumerate(
            zip(self.work_remotes, self.remotes)
        ):
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote, env_name, i),
                daemon=True,
            )
            p.start()
            work_remote.close()
            self.processes.append(p)
        print(f"  {n_envs} workers started.")

        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()

        self.remotes[0].send(("get_goal", None))
        self.goal = self.remotes[0].recv()

        print(f"✓ {n_envs} vision environments ready  |  goal = {self.goal}")
        print(f"  Camera: {CAMERA_NAME} @ {IMG_HEIGHT}×{IMG_WIDTH}\n")

    def reset(self):
        """Reset all environments. Returns (imgs, states)."""
        for remote in self.remotes:
            remote.send(("reset", None))
        results = [remote.recv() for remote in self.remotes]
        imgs = np.array([r[0] for r in results])      # (n_envs, C, H, W)
        states = np.array([r[1] for r in results])     # (n_envs, state_dim)
        return imgs, states

    def step(self, actions):
        """Step all environments.

        Returns:
            imgs: (n_envs, C, H, W) uint8
            states: (n_envs, state_dim) float for reward computation
            rewards: (n_envs,) float
            dones: (n_envs,) bool
            infos: list of dicts
        """
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        return (
            np.array([r[0] for r in results]),   # imgs
            np.array([r[1] for r in results]),   # states
            np.array([r[2] for r in results]),   # rewards
            np.array([r[3] for r in results]),   # dones
            [r[4] for r in results],             # infos
        )

    def close(self):
        if self.closed:
            return
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join(timeout=5)
        self.closed = True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env_name="PickPlace", n_envs=4, n_episodes=20000):
    print("=" * 70)
    print(f"TRAINING: {env_name} (Vision – CNN + HER + PER + 4-critic)")
    print("=" * 70)

    vec_env = VisionSubprocVecEnv(env_name, n_envs)
    goal = vec_env.goal  # constant goal (3,)

    actor_lr = 0.0001    # lower LR for CNN
    critic_lr = 0.0003
    batch_size = 128     # smaller batch (images are large)
    latent_dim = 256
    fc1_dims = 512
    fc2_dims = 256
    tau = 0.005

    n_actions = vec_env.action_space.shape[0]

    agent = VisionAgent(
        alpha=actor_lr,
        beta=critic_lr,
        img_shape=IMG_SHAPE,
        goal_dim=3,
        tau=tau,
        env=vec_env,
        n_actions=n_actions,
        latent_dim=latent_dim,
        fc1_dims=fc1_dims,
        fc2_dims=fc2_dims,
        batch_size=batch_size,
        max_size=500_000,      # ~10 GB with 84×84 uint8
        warmup=10000,
        noise_start=0.3,
        noise_end=0.05,
        n_critics=4,
        n_gradient_steps=2,
        max_grad_norm=1.0,
        lr_total_steps=2_000_000,
        chkpt_dir='./checkpoints/td3_vision',
    )

    # Print model sizes
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    critic_params = sum(p.numel() for p in agent.critics[0].parameters())
    print(f"  Actor parameters:   {actor_params:,}")
    print(f"  Critic parameters:  {critic_params:,} (×{agent.n_critics} critics)")
    print(f"  Device:             {agent.device}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Buffer capacity:    500,000 transitions")
    print(f"  Image shape:        {IMG_SHAPE}")
    print()

    # --- Resume from checkpoint ---
    try:
        agent.load_models()
        print("Resumed network weights from saved checkpoint.")
    except Exception:
        print("No network checkpoint found, starting from scratch.")

    buf_loaded = agent.memory.load(agent.buf_path)
    if buf_loaded:
        print(f"Resumed replay buffer "
              f"({min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions).")
    else:
        print("No replay buffer found, starting fresh.")

    # --- Resume training state ---
    training_state = agent.load_training_state()
    total_episodes = 0
    best_score = -np.inf
    score_history = []
    if training_state:
        total_episodes = training_state.get('total_episodes', 0)
        best_score = training_state.get('best_score', -np.inf)
        score_history = training_state.get('score_history_last100', [])
        print(f"Resumed: episode={total_episodes}, best={best_score:.1f}, "
              f"avg100={np.mean(score_history) if score_history else 0:.1f}")
    elif buf_loaded and agent.memory.mem_cntr >= agent.warmup:
        agent.time_step = agent.warmup

    # --- TensorBoard ---
    log_dir = os.path.join(".", "logs", f"{env_name}_vision")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Episode tracking (per-env) ---
    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    episode_buffers = [[] for _ in range(n_envs)]

    imgs, states = vec_env.reset()
    goals_batch = np.tile(goal, (n_envs, 1))

    print(f"\nStarting vision training "
          f"(warmup={agent.warmup:,}, batch_size={batch_size})...\n")

    while total_episodes < n_episodes:
        actions = agent.choose_action_batch(imgs, goals_batch)
        next_imgs, next_states, env_rewards, dones, infos = vec_env.step(
            actions,
        )

        for i in range(n_envs):
            # Bread position from state (for reward computation)
            achieved_goal = next_states[i, BREAD_POS].copy()

            # Staged reward using state information
            reward = compute_staged_reward(
                states[i], next_states[i], goal,
            )

            episode_scores[i] += reward
            episode_steps[i] += 1

            # Accumulate episode for HER (store images, not states)
            episode_buffers[i].append({
                'img': imgs[i].copy(),            # (C, H, W) uint8
                'action': actions[i].copy(),
                'reward': reward,
                'next_img': next_imgs[i].copy(),  # (C, H, W) uint8
                'done': dones[i],
                'goal': goal.copy(),
                'achieved_goal': achieved_goal,
            })

            if dones[i]:
                # Store full episode with HER relabeling
                agent.memory.store_episode(
                    episode_buffers[i], compute_reward,
                )
                episode_buffers[i] = []

                score = episode_scores[i]
                steps = int(episode_steps[i])
                episode_scores[i] = 0
                episode_steps[i] = 0

                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                total_episodes += 1

                writer.add_scalar("Score/Episode", score, total_episodes)
                writer.add_scalar("Score/Average_100", avg_score,
                                  total_episodes)
                writer.add_scalar("Steps/Episode", steps, total_episodes)
                writer.add_scalar(
                    "Noise/Scale",
                    agent.noise_scheduler.scale, total_episodes,
                )

                if score > best_score:
                    best_score = score
                    agent.save_best_models()
                    agent.save_training_state({
                        'total_episodes': total_episodes,
                        'best_score': best_score,
                        'score_history_last100': score_history[-100:],
                    })
                    print(
                        f"Episode {total_episodes:5d} | ★ BEST! "
                        f"Score: {score:8.3f} | Avg: {avg_score:8.3f} | "
                        f"Steps: {steps:3d} | "
                        f"Noise: {agent.noise_scheduler.scale:.3f}"
                    )
                elif total_episodes % 50 == 0:
                    print(
                        f"Episode {total_episodes:5d} | "
                        f"Score: {score:8.3f} | Avg: {avg_score:8.3f} | "
                        f"Best: {best_score:8.3f} | "
                        f"Buf: {min(agent.memory.mem_cntr, agent.memory.mem_size):,}"
                    )

                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.save_training_state({
                        'total_episodes': total_episodes,
                        'best_score': best_score,
                        'score_history_last100': score_history[-100:],
                    })
                    print(
                        f"\n  Checkpoint at ep {total_episodes}: "
                        f"Avg={avg_score:.3f}, "
                        f"Buffer={min(agent.memory.mem_cntr, agent.memory.mem_size):,}\n"
                    )

        # Learn (internally does n_gradient_steps)
        agent.learn()

        imgs = next_imgs
        states = next_states

    # --- Cleanup ---
    vec_env.close()
    writer.close()
    agent.save_models()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Best score:     {best_score:.3f}")
    if score_history:
        print(f"Final average:  {np.mean(score_history[-100:]):.3f}")
    print("=" * 70 + "\n")

    return agent


if __name__ == "__main__":
    ENV_NAME = "PickPlace"
    N_ENVS = 4       # fewer envs (offscreen rendering is expensive)
    N_EPISODES = 20000

    print(f"\n{'=' * 70}")
    print("VISION TRAINING (CNN + HER + PER + 4-critic + OU noise)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME}")
    print(f"  Camera:         {CAMERA_NAME} @ {IMG_HEIGHT}×{IMG_WIDTH}")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Buffer size:    500,000 transitions")
    print(f"  Horizon:        300 steps")
    print(f"{'=' * 70}\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES)
