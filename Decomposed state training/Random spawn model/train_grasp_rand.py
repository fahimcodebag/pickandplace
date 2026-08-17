#!/usr/bin/env python3
# Last updated: 2026-07-05 13:22 +0600
"""
Random-spawn grasp retraining (Stage 1b of the decomposed pipeline).

Fine-tunes the grasp sub-policy to handle randomized object spawns via a
growing spawn-region curriculum (see grasp_spawn_wrapper.py). Design:

  - WARM START: seeds the network weights from the proven fixed-spawn grasp
    model (checkpoints/td3_grasp) — it already knows HOW to grasp; this run
    teaches it WHERE. The original checkpoint is never written to.
  - NEW CHECKPOINT DIR: checkpoints/td3_grasp_rand (fixed-spawn model stays
    pristine; the thesis keeps both artifacts).
  - FRESH BUFFER: fixed-spawn transitions are a needle-thin distribution;
    the warm-started actor immediately generates competent, diverse data.
  - METRIC BEST-SAVING: "best" = mean(spawn_level/2, 50) * mean(success, 50)
    — difficulty-weighted rolling success, copied to <chkpt>/best/ so peaks
    survive later collapses (the lesson the place campaign paid for).
  - Early stop: >=90% success over 100 eps at >=95% of max spawn level.

Reward, termination, network size (64x32), and TD3 hyperparameters are
unchanged from the original grasp training.
"""

import os
import sys
import glob
import shutil

# Must be set before MuJoCo / robosuite are imported to avoid GL segfaults
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Parent ("Decomposed state training") + grandparent (pickandplace root)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

import multiprocessing as mp

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from td3 import Agent
from grasp_spawn_wrapper import make_spawn_grasp_env, SpawnCurriculumGraspWrapper


# ---------------------------------------------------------------------------
# Spawn configuration — FIXED randomization level, NO auto-curriculum.
#
# The self-advancing curriculum was removed after four runs showed its loop
# is the instability source for this stage (park-at-level -> self-similar
# data -> decay; or race-through-levels -> under-consolidation -> decay at
# trivial levels). Grasp — unlike place — has DENSE reward shaping that
# guides the arm from any spawn, so it never needed a curriculum: every run
# cleared moving levels in 1-2 windows. Training directly on the full spawn
# mix gives uniform data diversity forever (nothing to park, no gate to
# tune); best/ still harvests the peak via the metric criterion.
#
#   PHASE A: SPAWN_LEVEL = 1.0  — full position box, rotation fixed at 0
#   PHASE B: SPAWN_LEVEL = 2.0  — + full z-rotation (set manually after
#            phase A converges; reseed from phase A's best/)
# ---------------------------------------------------------------------------
SPAWN_LEVEL = 1.0


# ---------------------------------------------------------------------------
# Subprocess-parallel vectorized environment (same pattern as train_grasp.py)
# ---------------------------------------------------------------------------

def _worker(remote, parent_remote, env_name, seed):
    parent_remote.close()
    env = make_spawn_grasp_env(env_name, seed=seed, curriculum=False,
                               level=SPAWN_LEVEL)
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
    def __init__(self, env_name, n_envs=2):
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
# Best-model preservation (same rationale as train_place.py)
# ---------------------------------------------------------------------------

def _preserve_best_models(chkpt_dir):
    """Copy the just-saved model files into chkpt_dir/best/ so periodic
    checkpoints can never clobber the best policy."""
    best_dir = os.path.join(chkpt_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    for f in glob.glob(os.path.join(chkpt_dir, "*_td3")):
        shutil.copy2(f, os.path.join(best_dir, os.path.basename(f)))


def _warm_start(chkpt_dir, source_dir):
    """Seed td3_grasp_rand with the fixed-spawn grasp weights (read-only on
    the source). Only runs when the new dir has no checkpoint yet."""
    if os.path.exists(os.path.join(chkpt_dir, "actor_td3")):
        return False  # resuming an existing rand run — no seeding
    src_files = glob.glob(os.path.join(source_dir, "*_td3"))
    if not src_files:
        return False
    os.makedirs(chkpt_dir, exist_ok=True)
    for f in src_files:
        shutil.copy2(f, os.path.join(chkpt_dir, os.path.basename(f)))
    return True


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env_name="PickPlace", n_envs=2, n_episodes=10000):
    chkpt_dir = os.path.join(_HERE, "..", "..", "checkpoints", "td3_grasp_rand")
    fixed_grasp_dir = os.path.join(_HERE, "..", "..", "checkpoints", "td3_grasp")
    level_max = SpawnCurriculumGraspWrapper._LEVEL_MAX

    print("=" * 70)
    print(f"RANDOM-SPAWN GRASP RETRAINING: {env_name}")
    print("=" * 70)
    print(f"Environments:   {n_envs} (subprocess-parallel, fork)")
    print(f"Total episodes: {n_episodes}")
    print(f"Network:        64 → 32 (warm-started from fixed-spawn model)")
    print(f"Checkpoints:    {chkpt_dir}")
    print(f"Spawn:          FIXED level {SPAWN_LEVEL} (no auto-curriculum; "
          f"1.0 = full position box, 2.0 = + rotation)")
    print("=" * 70 + "\n")

    vec_env = SubprocVecEnv(env_name, n_envs)

    # --- Hyperparameters (identical to original grasp training) -------------
    agent = Agent(
        alpha=0.0003,
        beta=0.0003,
        input_dims=vec_env.observation_space.shape,
        tau=0.005,
        env=vec_env,
        n_actions=vec_env.action_space.shape[0],
        layer1_size=64,
        layer2_size=32,
        batch_size=512,
        max_size=200000,
        warmup=10000,
        chkpt_dir=chkpt_dir,
    )

    # --- Warm start / resume -------------------------------------------------
    seeded = _warm_start(chkpt_dir, fixed_grasp_dir)
    try:
        agent.load_models()
        if seeded:
            print("Warm-started from fixed-spawn grasp model "
                  f"({fixed_grasp_dir}).")
        else:
            print("Resumed network weights from td3_grasp_rand checkpoint.")
        # Any loaded checkpoint means a competent actor: skip the
        # random-action warmup — random flailing would only fill the buffer
        # with junk and waste hours. (Also correct for reward-change restarts
        # that reseed the dir with previous best weights but no buffer.)
        agent.time_step = agent.warmup
    except Exception:
        print("No checkpoint found anywhere — training from scratch "
              "(expected warm start; check checkpoints/td3_grasp).")

    buf_path = os.path.join(chkpt_dir, "replay_buffer.npz")
    if agent.memory.load(buf_path):
        print(f"Resumed replay buffer "
              f"({min(agent.memory.mem_cntr, agent.memory.mem_size):,} "
              f"transitions).\n")
        if agent.memory.mem_cntr >= agent.warmup:
            agent.time_step = agent.warmup
    else:
        print("Fresh replay buffer (fixed-spawn data intentionally not "
              "carried over).\n")

    # --- Logging -------------------------------------------------------------
    log_dir = os.path.join(_HERE, "..", "..", "logs", f"{env_name}_grasp_rand")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    total_episodes = 0
    best_score = -np.inf     # display only
    best_metric = 0.0        # save criterion: mean(level/max,50)*mean(success,50)
    score_history = []
    grasp_successes = []
    spawn_levels = []

    observations = vec_env.reset()
    print("Starting random-spawn grasp training...\n")

    while total_episodes < n_episodes:
        # choose_action_batch handles the warmup phase internally (uniform
        # random actions while time_step < warmup). With a warm start we set
        # time_step = warmup above, so it acts with the seeded policy + noise
        # from the very first step.
        actions = agent.choose_action_batch(observations)

        next_observations, rewards, dones, infos = vec_env.step(actions)

        for i in range(n_envs):
            episode_scores[i] += rewards[i]
            episode_steps[i] += 1
            agent.remember(observations[i], actions[i], rewards[i],
                           next_observations[i], dones[i])

            if dones[i]:
                score = episode_scores[i]
                steps = int(episode_steps[i])
                grasp_ok = infos[i].get("grasp_success", False)
                level = infos[i].get("spawn_level", 0.0)

                episode_scores[i] = 0
                episode_steps[i] = 0

                score_history.append(score)
                grasp_successes.append(1.0 if grasp_ok else 0.0)
                spawn_levels.append(level)
                avg_score = np.mean(score_history[-100:])
                avg_grasp = np.mean(grasp_successes[-100:]) * 100
                total_episodes += 1

                writer.add_scalar("GraspRand/Score_Episode", score, total_episodes)
                writer.add_scalar("GraspRand/Score_Avg100", avg_score, total_episodes)
                writer.add_scalar("GraspRand/Success_Rate_100", avg_grasp, total_episodes)
                writer.add_scalar("GraspRand/Spawn_Level", level, total_episodes)

                if score > best_score:
                    best_score = score   # display only

                # Difficulty-weighted rolling best (the save criterion)
                best_line = False
                if len(grasp_successes) >= 50:
                    lvl_50 = float(np.mean(spawn_levels[-50:])) / level_max
                    suc_50 = float(np.mean(grasp_successes[-50:]))
                    metric = lvl_50 * suc_50
                    if metric > best_metric + 0.005:
                        best_metric = metric
                        agent.save_models()
                        _preserve_best_models(chkpt_dir)
                        best_line = True
                        print(
                            f"Episode {total_episodes:5d} | ★ BEST POLICY! "
                            f"level50={lvl_50 * level_max:.2f} "
                            f"grasp50={suc_50 * 100:4.1f}% "
                            f"metric={metric:.3f} | Score: {score:7.2f}"
                        )

                if not best_line and total_episodes % 50 == 0:
                    lv = spawn_levels[-50:]
                    print(
                        f"Episode {total_episodes:5d} | Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Grasp%: {avg_grasp:5.1f}% | "
                        f"BestMetric: {best_metric:.3f}"
                    )
                    print(f"    spawn level (50): min={min(lv):.2f} "
                          f"mean={np.mean(lv):.2f} max={max(lv):.2f}")

                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.memory.save(buf_path)
                    print(
                        f"\nCheckpoint at {total_episodes}: Avg={avg_score:.2f} "
                        f"Grasp%={avg_grasp:.1f}% "
                        f"(buffer: {min(agent.memory.mem_cntr, agent.memory.mem_size):,} "
                        f"transitions)\n"
                    )

        # Early stop: mastered the configured spawn level (phase target).
        if (len(grasp_successes) >= 100
                and np.mean(grasp_successes[-100:]) >= 0.85):
            print(
                f"\n🎯 Phase target reached at SPAWN_LEVEL={SPAWN_LEVEL}! "
                f"Success: {np.mean(grasp_successes[-100:]) * 100:.1f}%"
            )
            if SPAWN_LEVEL < level_max:
                print("Next: back up best/, reseed from it, set "
                      f"SPAWN_LEVEL = {level_max} (add rotation), relaunch.")
            break

        agent.learn()
        observations = next_observations

    vec_env.close()
    writer.close()
    agent.save_models()
    agent.memory.save(buf_path)
    print("Replay buffer saved.")

    print("\n" + "=" * 70)
    print("RANDOM-SPAWN GRASP TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Best score:     {best_score:.2f}")
    print(f"Best metric:    {best_metric:.3f} "
          f"(mean level50/max x grasp50; policy in best/)")
    if grasp_successes:
        print(f"Final grasp %:  {np.mean(grasp_successes[-100:]) * 100:.1f}%")
    if spawn_levels:
        print(f"Final level:    {np.mean(spawn_levels[-100:]):.2f}/{level_max}")
    print("=" * 70 + "\n")

    print("Next steps:")
    print("  1. python3 test_grasp_rand.py --episodes 30 --best --random-spawn")
    print("  2. Point the place pipeline's grasp_chkpt_dir at "
          "checkpoints/td3_grasp_rand/best and rerun the end-to-end ladder")
    return agent


if __name__ == "__main__":
    ENV_NAME = "PickPlace"
    N_ENVS = 2
    N_EPISODES = 10000

    print(f"\n{'=' * 70}")
    print("RANDOM-SPAWN GRASP SUB-POLICY TRAINING (Decomposed, Stage 1b)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME} (Grasp only, growing spawn region)")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Network:        64 → 32 MLP (warm-started)")
    print(f"{'=' * 70}\n")

    train(ENV_NAME, N_ENVS, N_EPISODES)
