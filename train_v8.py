#!/usr/bin/env python3
"""Training loop v8: two-phase reward (grasp + carry-to-bin).

Key idea: The v7 agent grasps and lifts perfectly but jitters after lifting.
This version detects a successful grasp and then applies a STRONG carry
reward that guides the end-effector toward the goal bin while penalising
gripper opening (to prevent drops).

Phase 1 (pre-grasp):  robosuite built-in reward (unchanged, proven)
Phase 2 (post-grasp): built-in + strong carry reward + grip-hold penalty

Warm-starts from v7 weights (keeps excellent grasping skill).
"""

import os
import time
import datetime
import shutil

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
BREAD_POS = slice(0, 3)    # obs[0:3]
EEF_POS = slice(35, 38)    # obs[35:38]
GRIPPER = slice(42, 44)    # obs[42:44]
TABLE_Z = 0.845

# Carry reward parameters
CARRY_SCALE = 10.0          # strong signal for moving bread → goal
GRIP_PENALTY = -1.0         # penalty per step for opening gripper while grasped
HEIGHT_BONUS = 0.2          # bonus per step for keeping bread elevated
CLOSE_BONUS = 2.0           # bonus when bread XY within 10cm of goal
SUCCESS_BONUS = 10.0        # bonus when bread within 5cm of goal (3D)

# Grasp detection thresholds
GRASP_HEIGHT_THRESH = 0.03  # bread must be this far above table
GRASP_GRIPPER_THRESH = 0.025  # gripper_open must be below this

# Actor freezing: let critics adapt to new reward before updating actor
ACTOR_FREEZE_STEPS = 50000   # gradient steps with actor frozen


def two_phase_reward(obs, next_obs, goal, env_reward):
    """Two-phase reward: built-in for grasping, carry reward after grasp.

    Returns (reward, grasped_flag).
    """
    bread = next_obs[0:3]
    prev_bread = obs[0:3]
    gripper = next_obs[42:44]

    bread_height = bread[2] - TABLE_Z
    gripper_open = gripper[0] - gripper[1]   # ~0.04 open, ~0.01 closed

    # Start with robosuite built-in reward (always active)
    reward = float(env_reward)

    # --- Grasp detection ---
    grasped = (bread_height > GRASP_HEIGHT_THRESH and
               gripper_open < GRASP_GRIPPER_THRESH)

    if grasped:
        # === PHASE 2: carry to bin ===

        # Strong reward for reducing XY distance to goal
        prev_dist_xy = np.linalg.norm(prev_bread[:2] - goal[:2])
        curr_dist_xy = np.linalg.norm(bread[:2] - goal[:2])
        carry = (prev_dist_xy - curr_dist_xy) * CARRY_SCALE
        reward += carry

        # Bonus for being close to goal (XY)
        if curr_dist_xy < 0.10:
            reward += CLOSE_BONUS

        # Full 3D success bonus
        dist_3d = np.linalg.norm(bread - goal)
        if dist_3d < 0.05:
            reward += SUCCESS_BONUS

        # Height maintenance bonus (keep bread elevated during transport)
        if bread_height > 0.05:
            reward += HEIGHT_BONUS

        # Penalise gripper opening (to prevent drops)
        if gripper_open > GRASP_GRIPPER_THRESH:
            reward += GRIP_PENALTY

    return reward, grasped


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
# Subprocess vectorised environment (full random spawn)
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

    def log_episode(self, episode, score, avg, best, grasp_rate, noise,
                    buf_size):
        line = (
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"Ep {episode:6d} | Score: {score:8.3f} | Avg: {avg:8.3f} | "
            f"Best: {best:8.3f} | Grasp%: {grasp_rate:5.1f} | "
            f"Noise: {noise:.3f} | Buf: {buf_size:,}\n"
        )
        with open(self.path, 'a') as f:
            f.write(line)
            f.flush()

    def log_checkpoint(self, episode, avg, grasp_rate, buf_size):
        line = (
            f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] "
            f"*** CHECKPOINT ep {episode}: Avg={avg:.3f}, "
            f"Grasp%={grasp_rate:.1f}, Buf={buf_size:,}\n"
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
    print(f"TRAINING v8: {env_name} (two-phase: grasp + carry)")
    print("=" * 70)

    vec_env = SubprocVecEnv(env_name, n_envs)
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
        warmup=10000,             # shorter warmup (already pre-trained)
        noise_start=0.2,          # lower initial noise (grasping is learned)
        noise_end=0.08,
        n_critics=4,
        n_gradient_steps=2,
        max_grad_norm=1.0,
        lr_total_steps=4_000_000,
        chkpt_dir='./checkpoints/td3_v8',
    )

    actor_params = sum(p.numel() for p in agent.actor.parameters())
    actor_mb = sum(p.numel() * p.element_size()
                   for p in agent.actor.parameters()) / 1e6
    print(f"  Actor:  {actor_params:,} params ({actor_mb:.2f} MB FP32)")
    print(f"  Device: {agent.device}")

    # --- File logger ---
    log_file = os.path.join('.', 'logs', f'{env_name}_v8', 'training.log')
    logger = TrainingLogger(log_file)
    logger.log_message(
        f"Config: envs={n_envs}, batch={batch_size}, "
        f"layers={layer1_size}/{layer2_size}, "
        f"carry_scale={CARRY_SCALE}, grip_penalty={GRIP_PENALTY}, "
        f"noise=0.2->0.08"
    )

    # --- Resume from checkpoint ---
    try:
        agent.load_models()
        print("Resumed network weights from saved checkpoint.")
        logger.log_message("Resumed network weights from checkpoint.")
    except Exception:
        print("No checkpoint found, starting from scratch.")
        logger.log_message("Starting from scratch.")

    # --- Freeze actor (critics adapt to new reward first) ---
    actor_frozen = True
    for p in agent.actor.parameters():
        p.requires_grad = False
    print(f"  Actor FROZEN for first {ACTOR_FREEZE_STEPS:,} gradient steps.")
    logger.log_message(f"Actor frozen for {ACTOR_FREEZE_STEPS:,} gradient steps.")

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
        logger.log_message(
            f"Resumed: ep={total_episodes}, best={best_score:.1f}"
        )

    # --- TensorBoard ---
    tb_dir = os.path.join(".", "logs", f"{env_name}_v8")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)

    # --- Episode tracking ---
    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    episode_grasp_steps = np.zeros(n_envs, dtype=int)  # steps in grasped state
    episode_buffers = [[] for _ in range(n_envs)]

    observations = vec_env.reset()
    goals_batch = np.tile(goal, (n_envs, 1))

    # Track grasp success rate (rolling)
    grasp_history = []  # fraction of episode spent grasped

    print(f"\n--- v8 key changes ---")
    print(f"  Reward:  two-phase (builtin + carry after grasp)")
    print(f"  Carry:   scale={CARRY_SCALE}, close_bonus={CLOSE_BONUS}, "
          f"success={SUCCESS_BONUS}")
    print(f"  Grip:    penalty={GRIP_PENALTY}, height_bonus={HEIGHT_BONUS}")
    print(f"  Noise:   0.2 → 0.08 (lower — grasping is learned)")
    print(f"  Warmup:  10,000 (shorter)")
    print(f"  Envs:    {n_envs}")
    print(f"  Logging: {log_file}")
    print(f"  Freeze:  actor frozen for {ACTOR_FREEZE_STEPS:,} grad steps")
    print(f"\nStarting training "
          f"(warmup={agent.warmup:,}, batch_size={batch_size})...\n")

    t_start = time.time()

    while total_episodes < n_episodes:
        actions = agent.choose_action_batch(observations, goals_batch)
        next_observations, env_rewards, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            achieved_goal = next_observations[i, BREAD_POS].copy()

            # === Two-phase reward ===
            reward, grasped = two_phase_reward(
                observations[i], next_observations[i], goal,
                env_rewards[i],
            )

            episode_scores[i] += reward
            episode_steps[i] += 1
            if grasped:
                episode_grasp_steps[i] += 1

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
                g_steps = int(episode_grasp_steps[i])
                grasp_pct = (g_steps / steps * 100) if steps > 0 else 0

                episode_scores[i] = 0
                episode_steps[i] = 0
                episode_grasp_steps[i] = 0

                score_history.append(score)
                grasp_history.append(grasp_pct)
                avg_score = np.mean(score_history[-100:])
                avg_grasp = np.mean(grasp_history[-100:])
                total_episodes += 1

                # --- TensorBoard ---
                writer.add_scalar("Score/Episode", score, total_episodes)
                writer.add_scalar("Score/Average_100", avg_score,
                                  total_episodes)
                writer.add_scalar("Steps/Episode", steps, total_episodes)
                writer.add_scalar("Grasp/Percent", grasp_pct,
                                  total_episodes)
                writer.add_scalar("Grasp/Average_100", avg_grasp,
                                  total_episodes)
                writer.add_scalar("Noise/Scale",
                                  agent.noise_scheduler.scale,
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
                    })
                    print(
                        f"Episode {total_episodes:5d} | ★ BEST! "
                        f"Score: {score:8.3f} | Avg: {avg_score:8.3f} | "
                        f"Grasp%: {grasp_pct:5.1f} | "
                        f"Noise: {agent.noise_scheduler.scale:.3f}"
                    )
                    logger.log_episode(
                        total_episodes, score, avg_score, best_score,
                        avg_grasp, agent.noise_scheduler.scale, buf_size,
                    )

                elif total_episodes % 50 == 0:
                    elapsed = time.time() - t_start
                    eps_per_hr = total_episodes / (elapsed / 3600) \
                        if elapsed > 0 else 0
                    print(
                        f"Episode {total_episodes:5d} | "
                        f"Score: {score:8.3f} | Avg: {avg_score:8.3f} | "
                        f"Best: {best_score:8.3f} | "
                        f"Grasp%: {avg_grasp:5.1f} | "
                        f"Buf: {buf_size:,} | "
                        f"~{eps_per_hr:.0f} ep/hr"
                    )
                    logger.log_episode(
                        total_episodes, score, avg_score, best_score,
                        avg_grasp, agent.noise_scheduler.scale, buf_size,
                    )

                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.memory.save(agent.buf_path)
                    agent.save_training_state({
                        'total_episodes': total_episodes,
                        'best_score': best_score,
                        'score_history_last100': score_history[-100:],
                    })
                    logger.log_checkpoint(total_episodes, avg_score,
                                          avg_grasp, buf_size)
                    print(
                        f"\n  Checkpoint at ep {total_episodes}: "
                        f"Avg={avg_score:.3f}, Grasp%={avg_grasp:.1f}, "
                        f"Buf={buf_size:,}\n"
                    )

        # Learn
        agent.learn()

        # --- Unfreeze actor after critics have adapted ---
        if actor_frozen and agent.learn_step_cntr >= ACTOR_FREEZE_STEPS:
            for p in agent.actor.parameters():
                p.requires_grad = True
            actor_frozen = False
            print(f"\n  ★ ACTOR UNFROZEN at grad step {agent.learn_step_cntr:,} "
                  f"(ep {total_episodes}) — critics adapted to new reward.\n")
            logger.log_message(
                f"ACTOR UNFROZEN at grad step {agent.learn_step_cntr:,}, "
                f"ep {total_episodes}"
            )

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
    if grasp_history:
        print(f"Final grasp%:   {np.mean(grasp_history[-100:]):.1f}")
    print(f"Log file:       {log_file}")
    print("=" * 70 + "\n")

    return agent


if __name__ == "__main__":
    ENV_NAME = "PickPlace"
    N_ENVS = 28
    N_EPISODES = 60000

    # --- Warm-start from v7 ---
    WARMSTART_FROM = './checkpoints/td3_v7'

    print(f"\n{'=' * 70}")
    print("TRAINING v8 (two-phase reward: grasp + carry-to-bin)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME}")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Network:        2048 → 1024 (actor >4 MB FP32)")
    print(f"  Batch size:     1024")
    print(f"  Buffer size:    4,000,000 transitions")
    print(f"  Horizon:        500 steps")
    print(f"  Reward:         two-phase (builtin + carry after grasp)")
    print(f"  Carry scale:    {CARRY_SCALE}")
    print(f"  Noise:          0.2 → 0.08")
    print(f"  Warm-start:     {WARMSTART_FROM} (weights + NO buffer)")
    print(f"{'=' * 70}\n")

    # --- Warm-start: copy v7 weights only (not buffer) ---
    v8_dir = './checkpoints/td3_v8'
    if WARMSTART_FROM and not os.path.isdir(v8_dir):
        print(f"Copying v7 network weights to v8 dir...")
        os.makedirs(v8_dir, exist_ok=True)
        # Copy only model files, not replay buffer
        import glob
        for f in glob.glob(os.path.join(WARMSTART_FROM, '*_td3')):
            dst = os.path.join(v8_dir, os.path.basename(f))
            shutil.copy2(f, dst)
            print(f"  {os.path.basename(f)}")
        # Copy best models too
        for f in glob.glob(os.path.join(WARMSTART_FROM, '*_best_td3')):
            dst = os.path.join(v8_dir, os.path.basename(f))
            shutil.copy2(f, dst)
            print(f"  {os.path.basename(f)}")
        print(f"✓ Copied weights (NOT buffer — reward changed)")
        print(f"  v8 will learn new Q-values with two-phase reward\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES)
