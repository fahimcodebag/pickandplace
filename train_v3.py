#!/usr/bin/env python3
"""Training loop v3: built-in reward shaping + HER + all v2 enhancements.

Key difference from train_v2.py:
  - Uses robosuite's built-in dense reward (reward_shaping=True) instead
    of the custom compute_staged_reward(). The built-in reward is
    well-engineered with smooth gradients for reach → grasp → lift → place.
  - HER provides goal generalization for random bread spawn positions.
  - Can warm-start from v2 network weights (same architecture).

Uses subprocess-parallel vectorized environments. Each env worker
returns observations from which we extract bread_pos (achieved goal).
The goal (target bin) is constant and extracted once at startup.
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

from td3_v2 import Agent
from utils_rl import compute_reward  # sparse reward for HER relabeling only


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------
BREAD_POS = slice(0, 3)   # obs[0:3]


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(env_name="PickPlace", seed=None):
    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE",
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,              # longer horizon than v2 (was 300)
        reward_shaping=True,      # ← built-in dense reward
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    gym_env = GymWrapper(env)
    if seed is not None:
        gym_env.seed(seed)
    return env, gym_env


# ---------------------------------------------------------------------------
# Subprocess vectorised environment (returns goal on reset)
# ---------------------------------------------------------------------------

def _worker(remote, parent_remote, env_name, seed):
    parent_remote.close()
    raw_env, gym_env = make_single_env(env_name, seed=seed)
    goal = raw_env.target_bin_placements[
        raw_env.object_to_id['bread']
    ].copy()

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == "step":
            obs, reward, done, info = gym_env.step(data)
            if done:
                obs = gym_env.reset()
            remote.send((obs, reward, done, info))
        elif cmd == "reset":
            obs = gym_env.reset()
            remote.send(obs)
        elif cmd == "close":
            gym_env.close()
            remote.close()
            break
        elif cmd == "get_spaces":
            remote.send((gym_env.observation_space, gym_env.action_space))
        elif cmd == "get_goal":
            remote.send(goal)


class SubprocVecEnv:
    def __init__(self, env_name, n_envs=8):
        self.n_envs = n_envs
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

        # Get goal from first worker (constant for all)
        self.remotes[0].send(("get_goal", None))
        self.goal = self.remotes[0].recv()

        print(f"✓ {n_envs} environments ready  |  goal = {self.goal}\n")

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return np.array([remote.recv() for remote in self.remotes])

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]
        return (
            np.array([r[0] for r in results]),
            np.array([r[1] for r in results]),
            np.array([r[2] for r in results]),
            [r[3] for r in results],
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

def train(env_name="PickPlace", n_envs=12, n_episodes=20000,
          warmstart_from=None):
    """
    Args:
        env_name: robosuite environment name
        n_envs: number of parallel environments
        n_episodes: total training episodes
        warmstart_from: path to v2 checkpoint dir to load weights from
                        (e.g., './checkpoints/td3_v2')
    """
    print("=" * 70)
    print(f"TRAINING v3: {env_name} (built-in reward + HER + PER + 4-critic)")
    print("=" * 70)

    vec_env = SubprocVecEnv(env_name, n_envs)
    goal = vec_env.goal  # constant goal (3,)

    actor_lr = 0.0003
    critic_lr = 0.0003
    batch_size = 1024
    layer1_size = 2048       # bigger network (actor >4 MB FP32)
    layer2_size = 1024
    tau = 0.005

    input_dims = vec_env.observation_space.shape
    n_actions = vec_env.action_space.shape[0]

    agent = Agent(
        alpha=actor_lr,
        beta=critic_lr,
        obs_dims=input_dims,
        goal_dim=3,
        tau=tau,
        env=vec_env,
        n_actions=n_actions,
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
        max_size=4_000_000,       # larger buffer (128 GB RAM)
        warmup=25000,
        noise_start=0.3,
        noise_end=0.05,
        n_critics=4,
        n_gradient_steps=2,       # more gradient steps per env step
        max_grad_norm=1.0,
        lr_total_steps=4_000_000,
        chkpt_dir='./checkpoints/td3_v3',
    )

    # --- Warm-start from v2 weights (if requested) ---
    if warmstart_from and os.path.isdir(warmstart_from):
        print(f"\nWarm-starting from {warmstart_from}...")
        try:
            # Temporarily swap chkpt_dir to load v2 weights
            orig_dir = agent.chkpt_dir
            agent.chkpt_dir = warmstart_from

            # Update checkpoint file paths in networks
            for net in [agent.actor, agent.target_actor]:
                net.checkpoint_file = os.path.join(
                    warmstart_from, net.name + '_td3',
                )
            for nets in [agent.critics, agent.target_critics]:
                for net in nets:
                    net.checkpoint_file = os.path.join(
                        warmstart_from, net.name + '_td3',
                    )

            agent.load_models()
            print("✓ Loaded v2 network weights as warm start.")

            # Restore v3 checkpoint paths
            agent.chkpt_dir = orig_dir
            for net in [agent.actor, agent.target_actor]:
                net.checkpoint_file = os.path.join(
                    orig_dir, net.name + '_td3',
                )
            for nets in [agent.critics, agent.target_critics]:
                for net in nets:
                    net.checkpoint_file = os.path.join(
                        orig_dir, net.name + '_td3',
                    )

            # Do NOT load replay buffer — rewards are incompatible
            print("  (Replay buffer NOT loaded — reward function changed.)")
            print(f"  Saving warm-started weights to {orig_dir}/ ...")
            agent.save_models()
        except Exception as e:
            print(f"  Warm-start failed ({e}), starting from scratch.")
    else:
        # --- Resume from own v3 checkpoint ---
        try:
            agent.load_models()
            print("Resumed v3 network weights from saved checkpoint.")
        except Exception:
            print("No v3 network checkpoint found, starting from scratch.")

        buf_loaded = agent.memory.load(agent.buf_path)
        if buf_loaded:
            print(f"Resumed replay buffer "
                  f"({min(agent.memory.mem_cntr, agent.memory.mem_size):,} "
                  f"transitions).")
        else:
            print("No replay buffer found, starting fresh.")

    # --- Resume training state (only if not warm-starting) ---
    training_state = agent.load_training_state()
    total_episodes = 0
    best_score = -np.inf
    score_history = []
    if training_state and not warmstart_from:
        total_episodes = training_state.get('total_episodes', 0)
        best_score = training_state.get('best_score', -np.inf)
        score_history = training_state.get('score_history_last100', [])
        print(f"Resumed: episode={total_episodes}, best={best_score:.1f}, "
              f"avg100={np.mean(score_history) if score_history else 0:.1f}")

    # --- TensorBoard ---
    log_dir = os.path.join(".", "logs", f"{env_name}_v3")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Episode tracking (per-env) ---
    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    episode_buffers = [[] for _ in range(n_envs)]

    observations = vec_env.reset()
    goals_batch = np.tile(goal, (n_envs, 1))  # same goal for all envs

    print(f"\n--- v3 key changes ---")
    print(f"  Reward:  robosuite built-in (reward_shaping=True)")
    print(f"  Horizon: 500 steps (was 300 in v2)")
    print(f"  HER:     active (sparse relabeling for goal generalization)")
    print(f"  Buffer:  {'fresh' if warmstart_from else 'resumed if available'}")
    print(f"\nStarting training "
          f"(warmup={agent.warmup:,}, batch_size={batch_size})...\n")

    while total_episodes < n_episodes:
        actions = agent.choose_action_batch(observations, goals_batch)
        next_observations, env_rewards, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            achieved_goal = next_observations[i, BREAD_POS].copy()

            # ============================================================
            # v3: Use robosuite's built-in shaped reward directly
            # ============================================================
            reward = float(env_rewards[i])

            episode_scores[i] += reward
            episode_steps[i] += 1

            # Accumulate episode for HER
            episode_buffers[i].append({
                'state': observations[i].copy(),
                'action': actions[i].copy(),
                'reward': reward,
                'next_state': next_observations[i].copy(),
                'done': dones[i],
                'goal': goal.copy(),
                'achieved_goal': achieved_goal,
            })

            if dones[i]:
                # Store full episode with HER relabeling
                # HER uses sparse compute_reward for relabeled goals
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
                    agent.memory.save(agent.buf_path)
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
                    agent.memory.save(agent.buf_path)
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

        observations = next_observations

    # --- Cleanup ---
    vec_env.close()
    writer.close()
    agent.save_models()
    agent.memory.save(agent.buf_path)

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
    N_ENVS = 24               # utilize 16-core / 32-thread CPU
    N_EPISODES = 60000

    # Set to './checkpoints/td3_v2' to warm-start from v2 weights,
    # or None to start fresh / resume from v3 checkpoint.
    # NOTE: v3 uses 2048/1024 layers vs v2's 1024/512, so weights
    #       are NOT compatible. Set to None.
    WARMSTART_FROM = None

    print(f"\n{'=' * 70}")
    print("TRAINING v3 (built-in reward + HER + PER + 4-critic + OU noise)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME}")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Network:        2048 → 1024 (actor >4 MB FP32)")
    print(f"  Batch size:     1024")
    print(f"  Buffer size:    4,000,000 transitions")
    print(f"  Horizon:        500 steps")
    print(f"  Reward:         robosuite built-in (smooth dense)")
    if WARMSTART_FROM:
        print(f"  Warm-start:     {WARMSTART_FROM}")
    print(f"{'=' * 70}\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES,
                  warmstart_from=WARMSTART_FROM)
