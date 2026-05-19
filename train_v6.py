#!/usr/bin/env python3
"""Training loop v6: curriculum learning + file logging.

Key features:
  - CURRICULUM LEARNING: bread spawns at fixed position first, then
    gradually increases randomisation. This teaches the agent the FULL
    pick-and-place trajectory (including placing) before requiring
    generalisation across spawn positions.
  - Built-in robosuite reward (proven for reach/grasp/lift)
  - Gentle potential-based placement nudge (scale=0.1)
  - Higher noise floor (0.10) for late-game exploration
  - FILE LOGGING: writes episode data + timestamps to a log file
    so you always know how far training got
  - 28 parallel environments (for 32-thread CPU on WSL)
  - Fresh start (no warm-start from previous versions)

Curriculum schedule (configurable):
  Phase 1 (ep 0-15K):      spawn_frac linearly increases 0.0 → 1.0
  Phase 2 (ep 15K-60K):    spawn_frac = 1.0 (full random)
"""

import os
import time
import datetime

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
# Configuration
# ---------------------------------------------------------------------------
BREAD_POS = slice(0, 3)

PLACEMENT_SCALE = 0.1

# Curriculum: linearly increase spawn randomisation
CURRICULUM_START_EP = 0
CURRICULUM_END_EP = 15000    # full random by this episode


def placement_nudge(obs, next_obs, goal, gamma=0.99):
    """Truly potential-based placement shaping for ALL states.

    Phi(s) = -||bread_pos - goal||
    F(s,s') = gamma * Phi(s') - Phi(s)

    Defined for every state (no conditional), so guaranteed not to
    change the optimal policy. Scale is very small (0.1).
    """
    prev_bread = obs[0:3]
    bread = next_obs[0:3]
    phi_prev = -np.linalg.norm(prev_bread - goal)
    phi_next = -np.linalg.norm(bread - goal)
    return float((gamma * phi_next - phi_prev) * PLACEMENT_SCALE)


def curriculum_fraction(episode, start_ep=CURRICULUM_START_EP,
                        end_ep=CURRICULUM_END_EP):
    """Return spawn randomisation fraction [0.0, 1.0] for given episode."""
    if episode >= end_ep:
        return 1.0
    if episode <= start_ep:
        return 0.0
    return (episode - start_ep) / (end_ep - start_ep)


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
        horizon=500,
        reward_shaping=True,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    gym_env = GymWrapper(env)
    if seed is not None:
        gym_env.seed(seed)
    return env, gym_env


# ---------------------------------------------------------------------------
# Subprocess vectorised environment with curriculum support
# ---------------------------------------------------------------------------

def _worker(remote, parent_remote, env_name, seed):
    """Worker with curriculum spawn control."""
    parent_remote.close()
    raw_env, gym_env = make_single_env(env_name, seed=seed)
    goal = raw_env.target_bin_placements[
        raw_env.object_to_id['bread']
    ].copy()

    # Capture default spawn ranges for curriculum interpolation
    raw_env.reset()
    sampler = raw_env.placement_initializer.samplers["CollisionObjectSampler"]
    default_x_range = np.array(sampler.x_range, dtype=float)
    default_y_range = np.array(sampler.y_range, dtype=float)

    # Current curriculum fraction (0 = fixed, 1 = full random)
    spawn_frac = 0.0

    def _apply_curriculum():
        """Set spawn ranges based on current curriculum fraction."""
        s = raw_env.placement_initializer.samplers["CollisionObjectSampler"]
        s.x_range = default_x_range * spawn_frac
        s.y_range = default_y_range * spawn_frac
        if spawn_frac < 0.01:
            s.ensure_object_boundary_in_range = False
            s.ensure_valid_placement = False

    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break

        if cmd == "step":
            obs, reward, done, info = gym_env.step(data)
            if done:
                _apply_curriculum()
                obs = gym_env.reset()
            remote.send((obs, reward, done, info))

        elif cmd == "reset":
            _apply_curriculum()
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

        elif cmd == "set_curriculum":
            spawn_frac = float(data)


class CurriculumVecEnv:
    """Subprocess-parallel vectorised environment with curriculum control."""

    def __init__(self, env_name, n_envs=28):
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

    def set_curriculum(self, frac):
        """Update spawn randomisation fraction on all workers."""
        for remote in self.remotes:
            remote.send(("set_curriculum", frac))

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
# File logger
# ---------------------------------------------------------------------------

class TrainingLogger:
    """Logs training progress to a file with timestamps."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self._write_header()

    def _write_header(self):
        with open(self.path, 'a') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Training started: {datetime.datetime.now()}\n")
            f.write(f"{'='*80}\n")
            f.flush()

    def log_episode(self, episode, score, avg, best, curriculum_frac,
                    noise, buf_size):
        line = (
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"Ep {episode:6d} | Score: {score:8.3f} | Avg: {avg:8.3f} | "
            f"Best: {best:8.3f} | Curr: {curriculum_frac:.2f} | "
            f"Noise: {noise:.3f} | Buf: {buf_size:,}\n"
        )
        with open(self.path, 'a') as f:
            f.write(line)
            f.flush()

    def log_checkpoint(self, episode, avg, buf_size):
        line = (
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"*** CHECKPOINT ep {episode}: Avg={avg:.3f}, Buf={buf_size:,}\n"
        )
        with open(self.path, 'a') as f:
            f.write(line)
            f.flush()

    def log_message(self, msg):
        with open(self.path, 'a') as f:
            f.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
            f.flush()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env_name="PickPlace", n_envs=28, n_episodes=60000):
    print("=" * 70)
    print(f"TRAINING v6: {env_name} (curriculum + placement nudge)")
    print("=" * 70)

    vec_env = CurriculumVecEnv(env_name, n_envs)
    goal = vec_env.goal

    actor_lr = 0.0003
    critic_lr = 0.0003
    batch_size = 1024
    layer1_size = 2048
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
        max_size=4_000_000,
        warmup=25000,
        noise_start=0.3,
        noise_end=0.10,
        n_critics=4,
        n_gradient_steps=2,
        max_grad_norm=1.0,
        lr_total_steps=4_000_000,
        chkpt_dir='./checkpoints/td3_v6',
    )

    # Print model info
    actor_params = sum(p.numel() for p in agent.actor.parameters())
    actor_mb = sum(p.numel() * p.element_size()
                   for p in agent.actor.parameters()) / 1e6
    print(f"  Actor:  {actor_params:,} params ({actor_mb:.2f} MB FP32)")
    print(f"  Device: {agent.device}")

    # --- File logger ---
    log_file = os.path.join('.', 'logs', f'{env_name}_v6', 'training.log')
    logger = TrainingLogger(log_file)
    logger.log_message(
        f"Config: envs={n_envs}, batch={batch_size}, "
        f"layers={layer1_size}/{layer2_size}, "
        f"curriculum={CURRICULUM_START_EP}-{CURRICULUM_END_EP}, "
        f"noise=0.3->0.10, placement_scale={PLACEMENT_SCALE}"
    )

    # --- Resume from checkpoint ---
    try:
        agent.load_models()
        print("Resumed network weights from saved checkpoint.")
        logger.log_message("Resumed network weights from checkpoint.")
    except Exception:
        print("No checkpoint found, starting from scratch.")
        logger.log_message("Starting from scratch (fresh model).")

    buf_loaded = agent.memory.load(agent.buf_path)
    if buf_loaded:
        n_buf = min(agent.memory.mem_cntr, agent.memory.mem_size)
        print(f"Resumed replay buffer ({n_buf:,} transitions).")
        logger.log_message(f"Resumed replay buffer ({n_buf:,} transitions).")
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
        print(f"Resumed: episode={total_episodes}, best={best_score:.1f}")
        logger.log_message(f"Resumed: ep={total_episodes}, best={best_score:.1f}")

    # --- TensorBoard ---
    tb_dir = os.path.join(".", "logs", f"{env_name}_v6")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # --- Episode tracking ---
    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    episode_buffers = [[] for _ in range(n_envs)]

    # --- Initial curriculum ---
    curr_frac = curriculum_fraction(total_episodes)
    vec_env.set_curriculum(curr_frac)
    print(f"\n  Curriculum fraction: {curr_frac:.3f}")
    logger.log_message(f"Initial curriculum fraction: {curr_frac:.3f}")

    observations = vec_env.reset()
    goals_batch = np.tile(goal, (n_envs, 1))

    print(f"\n--- v6 key changes ---")
    print(f"  Reward:      robosuite built-in + placement nudge (×{PLACEMENT_SCALE})")
    print(f"  Curriculum:  spawn_frac 0→1 over ep {CURRICULUM_START_EP}-{CURRICULUM_END_EP}")
    print(f"  Noise:       0.3 → 0.10")
    print(f"  Envs:        {n_envs}")
    print(f"  Logging:     {log_file}")
    print(f"\nStarting training "
          f"(warmup={agent.warmup:,}, batch_size={batch_size})...\n")

    t_start = time.time()

    while total_episodes < n_episodes:
        actions = agent.choose_action_batch(observations, goals_batch)
        next_observations, env_rewards, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            achieved_goal = next_observations[i, BREAD_POS].copy()

            # Built-in reward + gentle placement nudge
            builtin_reward = float(env_rewards[i])
            nudge = placement_nudge(
                observations[i], next_observations[i], goal,
            )
            reward = builtin_reward + nudge

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

                # --- Update curriculum ---
                new_frac = curriculum_fraction(total_episodes)
                if abs(new_frac - curr_frac) > 0.01:
                    curr_frac = new_frac
                    vec_env.set_curriculum(curr_frac)

                # --- TensorBoard ---
                writer.add_scalar("Score/Episode", score, total_episodes)
                writer.add_scalar("Score/Average_100", avg_score,
                                  total_episodes)
                writer.add_scalar("Steps/Episode", steps, total_episodes)
                writer.add_scalar("Noise/Scale",
                                  agent.noise_scheduler.scale,
                                  total_episodes)
                writer.add_scalar("Curriculum/SpawnFrac", curr_frac,
                                  total_episodes)

                buf_size = min(agent.memory.mem_cntr,
                               agent.memory.mem_size)

                if score > best_score:
                    best_score = score
                    agent.save_best_models()
                    agent.save_training_state({
                        'total_episodes': total_episodes,
                        'best_score': best_score,
                        'score_history_last100': score_history[-100:],
                        'curriculum_frac': curr_frac,
                    })
                    print(
                        f"Episode {total_episodes:5d} | ★ BEST! "
                        f"Score: {score:8.3f} | Avg: {avg_score:8.3f} | "
                        f"Steps: {steps:3d} | "
                        f"Curr: {curr_frac:.2f} | "
                        f"Noise: {agent.noise_scheduler.scale:.3f}"
                    )
                    logger.log_episode(
                        total_episodes, score, avg_score, best_score,
                        curr_frac, agent.noise_scheduler.scale, buf_size,
                    )

                elif total_episodes % 50 == 0:
                    elapsed = time.time() - t_start
                    eps_per_hr = total_episodes / (elapsed / 3600) \
                        if elapsed > 0 else 0
                    print(
                        f"Episode {total_episodes:5d} | "
                        f"Score: {score:8.3f} | Avg: {avg_score:8.3f} | "
                        f"Best: {best_score:8.3f} | "
                        f"Curr: {curr_frac:.2f} | "
                        f"Buf: {buf_size:,} | "
                        f"~{eps_per_hr:.0f} ep/hr"
                    )
                    logger.log_episode(
                        total_episodes, score, avg_score, best_score,
                        curr_frac, agent.noise_scheduler.scale, buf_size,
                    )

                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.memory.save(agent.buf_path)
                    agent.save_training_state({
                        'total_episodes': total_episodes,
                        'best_score': best_score,
                        'score_history_last100': score_history[-100:],
                        'curriculum_frac': curr_frac,
                    })
                    logger.log_checkpoint(total_episodes, avg_score,
                                          buf_size)
                    print(
                        f"\n  Checkpoint at ep {total_episodes}: "
                        f"Avg={avg_score:.3f}, Curr={curr_frac:.2f}, "
                        f"Buf={buf_size:,}\n"
                    )

        # Learn
        agent.learn()
        observations = next_observations

    # --- Cleanup ---
    vec_env.close()
    writer.close()
    agent.save_models()
    agent.memory.save(agent.buf_path)

    elapsed = time.time() - t_start
    logger.log_message(
        f"TRAINING COMPLETE: {total_episodes} episodes in "
        f"{elapsed/3600:.1f} hours, best={best_score:.3f}"
    )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Best score:     {best_score:.3f}")
    print(f"Time:           {elapsed/3600:.1f} hours")
    if score_history:
        print(f"Final average:  {np.mean(score_history[-100:]):.3f}")
    print(f"Log file:       {log_file}")
    print("=" * 70 + "\n")

    return agent


if __name__ == "__main__":
    ENV_NAME = "PickPlace"
    N_ENVS = 28
    N_EPISODES = 60000

    print(f"\n{'=' * 70}")
    print("TRAINING v6 (curriculum + built-in reward + placement nudge)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME}")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Network:        2048 → 1024 (actor >4 MB FP32)")
    print(f"  Batch size:     1024")
    print(f"  Buffer size:    4,000,000 transitions")
    print(f"  Horizon:        500 steps")
    print(f"  Reward:         robosuite built-in + placement nudge (×{PLACEMENT_SCALE})")
    print(f"  Noise:          0.3 → 0.10")
    print(f"  Curriculum:     spawn_frac 0→1 over ep {CURRICULUM_START_EP}-{CURRICULUM_END_EP}")
    print(f"  Fresh start:    yes (no warm-start)")
    print(f"{'=' * 70}\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES)
