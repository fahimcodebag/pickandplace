#!/usr/bin/env python3
# Last updated: 2026-06-30
"""
Training script for the Grasp sub-policy (Stage 1 of decomposed RL).

Trains a small 64→32 MLP actor to reach and grasp the bread object.
Uses subprocess-parallel vectorized environments, same as the monolithic
train_vectorized.py but with:
  - GraspRewardWrapper (reach + grasp rewards only, early termination)
  - Smaller network (64→32 vs 512→256)
  - Shorter horizon (200 steps vs 500)
  - Separate checkpoints (./checkpoints/td3_grasp/)

Reuses networks.py, td3.py, buffer.py from the parent directory.
"""

import os
import sys

# Must be set before MuJoCo / robosuite are imported to avoid GL segfaults
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Add parent directory to path so we can import existing modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import multiprocessing as mp

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import torch
from torch.utils.tensorboard import SummaryWriter
from td3 import Agent
from grasp_env_wrapper import GraspRewardWrapper


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_grasp_env(env_name="PickPlace", seed=None):
    """Create a single robosuite environment with grasp-only reward."""
    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,              # robosuite hard limit (wrapper enforces 200)
        reward_shaping=False,     # we provide our own grasp reward
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Fix object spawn to a constant position (matching monolithic training)
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

    env = GraspRewardWrapper(env)  # grasp-only rewards + early termination
    env = GymWrapper(env)
    if seed is not None:
        env.seed(seed)
    return env


# ---------------------------------------------------------------------------
# Subprocess-parallel vectorized environment
# ---------------------------------------------------------------------------

def _worker(remote, parent_remote, env_name, seed):
    """Worker loop that runs in a child process."""
    parent_remote.close()
    env = make_grasp_env(env_name, seed=seed)
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
    """
    Runs N environments in separate subprocesses communicating via pipes.

    Uses 'fork' on Linux for copy-on-write memory sharing.
    """

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

        # Fetch spaces from the first worker
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        print(f"✓ {n_envs} environments ready (subprocess-parallel)\n")

    def reset(self):
        """Reset all environments in parallel and return stacked observations."""
        for remote in self.remotes:
            remote.send(("reset", None))
        observations = np.array([remote.recv() for remote in self.remotes])
        return observations

    def step(self, actions):
        """Step all environments in parallel."""
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = [remote.recv() for remote in self.remotes]

        observations = np.array([r[0] for r in results])
        rewards = np.array([r[1] for r in results])
        dones = np.array([r[2] for r in results])
        infos = [r[3] for r in results]

        return observations, rewards, dones, infos

    def close(self):
        """Send close command and join all subprocesses."""
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

def train(env_name="PickPlace", n_envs=8, n_episodes=10000):
    """
    Train the Grasp sub-policy using TD3 with subprocess-parallel envs.

    Uses smaller networks (64→32) since the grasp task is narrow.
    """
    chkpt_dir = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "td3_grasp")

    print("=" * 70)
    print(f"GRASP MODEL TRAINING: {env_name}")
    print("=" * 70)
    print(f"Environments:   {n_envs} (subprocess-parallel, fork)")
    print(f"Total episodes: {n_episodes}")
    print(f"Network:        64 → 32 (actor & critic)")
    print(f"Checkpoints:    {chkpt_dir}")
    print("=" * 70 + "\n")

    # --- Create environments ------------------------------------------------
    vec_env = SubprocVecEnv(env_name, n_envs)

    # --- Hyperparameters (tuned for simpler grasp task) ---------------------
    actor_lr = 0.0003
    critic_lr = 0.0003
    batch_size = 512
    layer1_size = 64
    layer2_size = 32
    tau = 0.005
    warmup = 10000
    max_buffer_size = 200000
    noise = 0.1

    input_dims = vec_env.observation_space.shape
    n_actions = vec_env.action_space.shape[0]

    # --- Agent --------------------------------------------------------------
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
        max_size=max_buffer_size,
        warmup=warmup,
        chkpt_dir=chkpt_dir,
    )

    # --- Resume from checkpoint if available --------------------------------
    try:
        agent.load_models()
        print("Resumed network weights from saved checkpoint.")
    except Exception:
        print("No network checkpoint found, starting from scratch.")

    buf_path = os.path.join(chkpt_dir, "replay_buffer.npz")
    buf_loaded = agent.memory.load(buf_path)
    if buf_loaded:
        print(f"Resumed replay buffer ({min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions).\n")
        if agent.memory.mem_cntr >= agent.warmup:
            agent.time_step = agent.warmup
    else:
        print("No replay buffer found, starting fresh.\n")

    # --- Logging ------------------------------------------------------------
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"{env_name}_grasp")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Episode tracking (per-env) ----------------------------------------
    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    total_episodes = 0
    best_score = -np.inf
    score_history = []
    grasp_successes = []   # track per-episode grasp success

    # --- Initial reset ------------------------------------------------------
    observations = vec_env.reset()

    print("Starting grasp training...\n")

    while total_episodes < n_episodes:
        # ---- Batched action selection (single forward pass) ----------------
        actions = agent.choose_action_batch(observations)

        # ---- Step all environments -----------------------------------------
        next_observations, rewards, dones, infos = vec_env.step(actions)

        # ---- Store transitions ---------------------------------------------
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

            # — Handle completed episodes ------------------------------------
            if dones[i]:
                score = episode_scores[i]
                steps = int(episode_steps[i])
                grasp_ok = infos[i].get("grasp_success", False)

                # Reset per-env tracking
                episode_scores[i] = 0
                episode_steps[i] = 0

                score_history.append(score)
                grasp_successes.append(1.0 if grasp_ok else 0.0)
                avg_score = np.mean(score_history[-100:])
                avg_grasp = np.mean(grasp_successes[-100:]) * 100
                total_episodes += 1

                # TensorBoard
                writer.add_scalar("Grasp/Score_Episode", score, total_episodes)
                writer.add_scalar("Grasp/Score_Avg100", avg_score, total_episodes)
                writer.add_scalar("Grasp/Steps_Episode", steps, total_episodes)
                writer.add_scalar("Grasp/Success_Rate_100", avg_grasp, total_episodes)

                # Save best model
                if score > best_score:
                    best_score = score
                    agent.save_models()
                    print(
                        f"Episode {total_episodes:5d} | ★ BEST! Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Grasp%: {avg_grasp:5.1f}% | Steps: {steps:3d}"
                    )
                elif total_episodes % 50 == 0:
                    print(
                        f"Episode {total_episodes:5d} | Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Grasp%: {avg_grasp:5.1f}% | Best: {best_score:7.2f}"
                    )

                # Periodic checkpoint
                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.memory.save(buf_path)
                    print(
                        f"\nCheckpoint at {total_episodes}: Avg={avg_score:.2f} "
                        f"Grasp%={avg_grasp:.1f}% "
                        f"(buffer: {min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions)\n"
                    )

        # ---- Early stopping: 95% grasp success over last 100 episodes ------
        if len(grasp_successes) >= 100 and np.mean(grasp_successes[-100:]) >= 0.95:
            print(
                f"\n🎯 Grasp target reached! "
                f"Success rate: {np.mean(grasp_successes[-100:]) * 100:.1f}%"
            )
            break

        # ---- Learn ONCE per timestep (not per env) -------------------------
        agent.learn()

        observations = next_observations

    # --- Cleanup ------------------------------------------------------------
    vec_env.close()
    writer.close()
    agent.save_models()
    agent.memory.save(buf_path)
    print("Replay buffer saved.")

    print("\n" + "=" * 70)
    print("GRASP TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Best score:     {best_score:.2f}")
    if score_history:
        print(f"Final average:  {np.mean(score_history[-100:]):.2f}")
    if grasp_successes:
        print(f"Final grasp %:  {np.mean(grasp_successes[-100:]) * 100:.1f}%")
    print("=" * 70 + "\n")

    print("Next steps:")
    print("  1. python test_grasp.py --episodes 20")
    print("  2. python collect_grasp_states.py --episodes 1000")
    print("  3. Train the Place model using collected grasp terminal states")

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ENV_NAME = "PickPlace"
    N_ENVS = 2
    N_EPISODES = 10000

    print(f"\n{'=' * 70}")
    print("GRASP SUB-POLICY TRAINING (Decomposed)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME} (Grasp only)")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Network:        64 → 32 MLP")
    print(f"  Replay buffer:  200,000 transitions")
    print(f"{'=' * 70}\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES)
