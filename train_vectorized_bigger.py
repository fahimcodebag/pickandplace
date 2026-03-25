#!/usr/bin/env python3
"""
Training for PickPlace with subprocess-parallel vectorized environments.
Uses robosuite's built-in reward_shaping=True and bigger networks (1024/512).
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
from td3_bigger import Agent


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(env_name="PickPlace", seed=None):
    """Create a single robosuite environment with built-in reward shaping."""
    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Keep default robosuite behavior: randomized object spawn each episode.
    env = GymWrapper(env)
    if seed is not None:
        env.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Subprocess-parallel vectorized environment
# ---------------------------------------------------------------------------

def _worker(remote, parent_remote, env_name, seed):
    parent_remote.close()
    env = make_single_env(env_name, seed=seed)
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == "step":
            obs, reward, done, info = env.step(data)
            if done:
                obs = env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = env.reset()
            remote.send(obs)
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))


class SubprocVecEnv:
    def __init__(self, env_name, n_envs=8):
        self.n_envs = n_envs
        self.waiting = False
        self.closed = False

        ctx = mp.get_context("fork")

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(n_envs)]
        )

        print(f"Spawning {n_envs} subprocess environments (fork)...")
        self.processes = []
        for i, (work_remote, remote) in enumerate(
            zip(self.work_remotes, self.remotes)
        ):
            print(f"  Spawning env {i + 1}/{n_envs}...", end=" ", flush=True)
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote, env_name, i),
                daemon=True,
            )
            p.start()
            work_remote.close()
            self.processes.append(p)
            print("✓", flush=True)
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        print(f"✓ {n_envs} environments ready (subprocess-parallel)\n")

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.array([remote.recv() for remote in self.remotes])

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        observations = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]
        return observations, rewards, dones, infos

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

def train(env_name="PickPlace", n_envs=8, n_episodes=20000):
    print("=" * 70)
    print(f"TRAINING: {env_name} (bigger network, built-in reward shaping)")
    print("=" * 70)
    print(f"Environments:   {n_envs} (subprocess-parallel, fork)")
    print(f"Total episodes: {n_episodes}")
    print("=" * 70 + "\n")

    vec_env = SubprocVecEnv(env_name, n_envs)

    actor_lr = 0.0005
    critic_lr = 0.0005
    batch_size = 2048
    layer1_size = 1024
    layer2_size = 512
    tau = 0.005

    input_dims = vec_env.observation_space.shape
    n_actions = vec_env.action_space.shape[0]

    agent = Agent(
        alpha=actor_lr,
        beta=critic_lr,
        input_dims=input_dims,
        tau=tau,
        env=vec_env,
        n_actions=n_actions,
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
        max_size=2000000,
        warmup=25000,
        chkpt_dir='./checkpoints/td3_bigger',
    )

    try:
        agent.load_models()
        print("Resumed network weights from saved checkpoint.")
    except Exception:
        print("No network checkpoint found, starting from scratch.")

    buf_loaded = agent.memory.load(agent.buf_path)
    if buf_loaded:
        print(f"Resumed replay buffer ({min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions).\n")
        if agent.memory.mem_cntr >= agent.warmup:
            agent.time_step = agent.warmup
    else:
        print("No replay buffer found, starting fresh.\n")

    log_dir = os.path.join(".", "logs", f"{env_name}_vectorized_bigger")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    total_episodes = 0
    best_score = -np.inf
    score_history = []

    observations = vec_env.reset()

    print("Starting training...\n")

    while total_episodes < n_episodes:
        actions = agent.choose_action_batch(observations)
        next_observations, rewards, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            episode_scores[i] += rewards[i]
            episode_steps[i] += 1

            agent.remember(
                observations[i],
                actions[i],
                rewards[i],
                next_observations[i],
                dones[i],
            )

            if dones[i]:
                score = episode_scores[i]
                steps = int(episode_steps[i])

                episode_scores[i] = 0
                episode_steps[i] = 0

                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                total_episodes += 1

                writer.add_scalar("Score/Episode", score, total_episodes)
                writer.add_scalar("Score/Average_100", avg_score, total_episodes)
                writer.add_scalar("Steps/Episode", steps, total_episodes)

                if score > best_score:
                    best_score = score
                    agent.save_best_models()
                    agent.memory.save(agent.buf_path)
                    print(
                        f"Episode {total_episodes:5d} | ★ BEST! Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Steps: {steps:3d}"
                    )
                elif total_episodes % 50 == 0:
                    print(
                        f"Episode {total_episodes:5d} | Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Best: {best_score:7.2f}"
                    )

                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.memory.save(agent.buf_path)
                    print(f"\nCheckpoint at {total_episodes}: Avg={avg_score:.2f} (buffer: {min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions)\n")

        if len(score_history) >= 100 and np.mean(score_history[-100:]) >= 3000:
            print(f"\nTarget reached! Avg: {np.mean(score_history[-100:]):.2f}")
            break

        # Reverted to 1 update per step so the CPU doesn't choke
        agent.learn()
            
        observations = next_observations

    vec_env.close()
    writer.close()
    agent.save_models()
    agent.memory.save(agent.buf_path)
    print("Replay buffer saved.")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Best score:     {best_score:.2f}")
    if score_history:
        print(f"Final average:  {np.mean(score_history[-100:]):.2f}")
    print("=" * 70 + "\n")

    return agent


if __name__ == "__main__":
    ENV_NAME = "PickPlace"
    N_ENVS = 8
    N_EPISODES = 20000

    print(f"\n{'=' * 70}")
    print("TRAINING (bigger network, built-in reward shaping)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME}")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Replay buffer:  1,000,000 transitions")
    print(f"{'=' * 70}\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES)
