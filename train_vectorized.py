#!/usr/bin/env python3
"""
Training for PickPlace with subprocess-parallel vectorized environments.

Runs N environments in separate subprocesses via multiprocessing (fork).
On Linux, fork uses copy-on-write so memory overhead is low (~100-200 MB
per env rather than the ~500 MB that 'spawn' would require).
"""

import os
# Must be set before MuJoCo / robosuite are imported to avoid GL segfaults
# on headless machines. Use 'osmesa' if EGL is unavailable.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import multiprocessing as mp

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import torch
from torch.utils.tensorboard import SummaryWriter
from td3 import Agent


# ---------------------------------------------------------------------------
# Curriculum reward wrapper
# ---------------------------------------------------------------------------

class CurriculumRewardWrapper:
    """
    Staged dense reward that wraps the raw robosuite env (before GymWrapper).

    Reward pipeline (each stage provides a dense, bounded signal):
      1. reach:   gripper → object proximity  (0 to W_REACH per step)
      2. grip:    gripper-close bonus when near object (encourages grasping)
      3. grasp:   sustained grasp bonus       (W_GRASP per step while held)
      4. lift:    proportional lift height     (0 to W_LIFT per step)
      5. hover:   object → target bin XY      (0 to W_HOVER per step, grasped)
      6. success: one-time completion bonus    (W_SUCCESS, awarded once)
    """
    # --- reward weights (per step unless noted) ---
    W_REACH       = 1.0
    W_GRIP_CLOSE  = 0.5
    W_GRASP       = 10.0
    W_LIFT        = 3.0
    W_HOVER       = 0.5
    W_SUCCESS     = 50.0   # one-time bonus (not per step)

    # --- penalties ---
    P_IDLE        = -0.4   # small per-step cost to discourage doing nothing
    P_DROP        = -5.0   # dropping a grasped object
    P_AWAY        = -0.7   # moving gripper away from object when not grasped

    # --- distance scales ---
    _REACH_SCALE  = 0.30
    _GRIP_RANGE   = 0.06
    _HOVER_SCALE  = 0.25
    _LIFT_CEIL    = 0.12

    def __init__(self, env):
        self._rs_env = env
        self._init_z = None
        self._success_given = False
        self._target_bin = None
        self._prev_grasped = False
        self._prev_d_reach = None

    def __getattr__(self, name):
        return getattr(self._rs_env, name)

    def reset(self):
        obs_dict = self._rs_env.reset()
        self._success_given = False
        self._target_bin = None
        self._prev_grasped = False
        self._prev_d_reach = None
        try:
            pos = self._get_obj_pos(obs_dict)
            self._init_z = float(pos[2]) if pos is not None else 0.82
        except Exception:
            self._init_z = 0.82
        return obs_dict

    def step(self, action):
        obs_dict, _, done, info = self._rs_env.step(action)
        reward = self._curriculum_reward(obs_dict)
        return obs_dict, reward, done, info

    # --- helpers -----------------------------------------------------------

    def _get_target_obj_name(self):
        try:
            return self._rs_env.obj_to_use
        except AttributeError:
            pass
        for cand in ("Can", "Milk", "Cereal", "Bread"):
            try:
                if self._rs_env.object_to_id.get(cand) is not None:
                    return cand
            except AttributeError:
                pass
        return None

    def _get_obj_pos(self, obs_dict):
        name = self._get_target_obj_name()
        if name is not None:
            key = f"{name}_pos"
            if key in obs_dict:
                return np.array(obs_dict[key])
        for cand in ("Can_pos", "Milk_pos", "Cereal_pos", "Bread_pos"):
            if cand in obs_dict:
                return np.array(obs_dict[cand])
        return None

    def _get_target_bin_pos(self):
        """Return XYZ position of the target bin for the active object."""
        if self._target_bin is not None:
            return self._target_bin
        try:
            name = self._get_target_obj_name()
            # object_to_id uses lowercase keys
            obj_idx = self._rs_env.object_to_id[name.lower()]
            self._target_bin = np.array(
                self._rs_env.target_bin_placements[obj_idx]
            )
            return self._target_bin
        except Exception:
            return None

    def _is_grasped(self):
        try:
            return self._rs_env._check_grasp(
                gripper=self._rs_env.robots[0].gripper,
                object_geoms=self._rs_env.objects[self._rs_env.object_id],
            )
        except Exception:
            return False

    def _gripper_aperture(self, obs_dict):
        """Return normalised gripper opening: 0 = closed, 1 = fully open."""
        try:
            qpos = np.array(obs_dict["robot0_gripper_qpos"])
            # Panda gripper: qpos ≈ [+0.02, -0.02] when open, [0, 0] closed
            return float(np.clip(abs(qpos[0]) / 0.04, 0.0, 1.0))
        except Exception:
            return 0.5

    def _curriculum_reward(self, obs_dict):
        r = 0.0
        eef_pos = np.array(obs_dict["robot0_eef_pos"])
        obj_pos = self._get_obj_pos(obs_dict)
        if obj_pos is None:
            return r

        d_reach = np.linalg.norm(eef_pos - obj_pos)

        # ---- Stage 1: Reach toward object ----------------------------------
        r += self.W_REACH * max(0.0, 1.0 - d_reach / self._REACH_SCALE)

        # ---- Stage 2: Encourage gripper closing when very close ------------
        if d_reach < self._GRIP_RANGE:
            aperture = self._gripper_aperture(obs_dict)
            r += self.W_GRIP_CLOSE * (1.0 - aperture)  # reward closing

        # ---- Grasp check ---------------------------------------------------
        grasped = self._is_grasped()

        # ---- Penalties -----------------------------------------------------
        r += self.P_IDLE

        if self._prev_grasped and not grasped:
            r += self.P_DROP

        if not grasped and self._prev_d_reach is not None:
            if d_reach > self._prev_d_reach + 0.005:
                r += self.P_AWAY

        self._prev_grasped = grasped
        self._prev_d_reach = d_reach

        if grasped:
            # ---- Stage 3: Sustained grasp ----------------------------------
            r += self.W_GRASP

            # ---- Stage 4: Proportional lift --------------------------------
            init_z = self._init_z if self._init_z is not None else 0.82
            lift = max(0.0, obj_pos[2] - init_z)
            r += self.W_LIFT * min(1.0, lift / self._LIFT_CEIL)

            # ---- Stage 5: Move toward target bin (XY) ----------------------
            target = self._get_target_bin_pos()
            if target is not None:
                d_place = np.linalg.norm(obj_pos[:2] - target[:2])
                r += self.W_HOVER * max(0.0, 1.0 - d_place / self._HOVER_SCALE)

        # ---- Stage 6: One-time success bonus -------------------------------
        if not self._success_given:
            try:
                if self._rs_env._check_success():
                    r += self.W_SUCCESS
                    self._success_given = True
            except Exception:
                pass

        return float(r)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_single_env(env_name="PickPlace", seed=None):
    """Create a single robosuite environment instance."""
    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        reward_shaping=False,   # we provide our own curriculum reward
        control_freq=20,
        single_object_mode=2,   # train on one fixed object (simpler)
        object_type="bread",      # lowercase as required by robosuite
    )
    # Fix object spawn to a constant position (must override the method
    # because hard_reset=True re-calls _get_placement_initializer each reset)
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
    env = CurriculumRewardWrapper(env)   # staged rewards before gym flattening
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
    """
    Runs N environments in separate subprocesses communicating via pipes.

    Uses 'fork' on Linux for copy-on-write memory sharing, keeping per-env
    overhead to ~100-200 MB instead of the ~500 MB that 'spawn' would need.
    Each subprocess runs its own MuJoCo instance with true OS-level
    parallelism — no GIL limitation.
    """

    def __init__(self, env_name, n_envs=8):
        self.n_envs = n_envs
        self.waiting = False
        self.closed = False

        # Use fork for COW memory on Linux
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
            work_remote.close()  # parent doesn't use the worker end
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
        """
        Step all environments in parallel with the given actions.

        Args:
            actions: np.ndarray of shape (n_envs, action_dim)

        Returns:
            observations, rewards, dones, infos
        """
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

def train(env_name="PickPlace", n_envs=8, n_episodes=20000):
    """
    Train a TD3 agent using multiple environments (single process).

    Args:
        env_name:    Name of the robosuite task.
        n_envs:      Number of environments (in same process).
        n_episodes:  Total training episodes across all envs.
    """
    print("=" * 70)
    print(f"TRAINING: {env_name}")
    print("=" * 70)
    print(f"Environments:   {n_envs} (subprocess-parallel, fork)")
    print(f"Total episodes: {n_episodes}")
    print("=" * 70 + "\n")

    # --- Create environments ------------------------------------------------
    vec_env = SubprocVecEnv(env_name, n_envs)

    # --- Hyperparameters ----------------------------------------------------
    actor_lr = 0.0005
    critic_lr = 0.0005
    batch_size = 1024
    layer1_size = 512
    layer2_size = 256
    tau = 0.005   # standard TD3 value; 0.05 caused target-network instability

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
        max_size=500000,  # 500K replay buffer
        warmup=25000,     # more random exploration before policy kicks in
    )

    # --- Resume from checkpoint if available --------------------------------
    try:
        agent.load_models()
        print("Resumed network weights from saved checkpoint.")
    except Exception:
        print("No network checkpoint found, starting from scratch.")

    buf_loaded = agent.memory.load()
    if buf_loaded:
        print(f"Resumed replay buffer ({min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions).\n")
        # Skip warmup if buffer already has enough data
        if agent.memory.mem_cntr >= agent.warmup:
            agent.time_step = agent.warmup
    else:
        print("No replay buffer found, starting fresh.\n")

    # --- Logging ------------------------------------------------------------
    log_dir = os.path.join(".", "logs", f"{env_name}_vectorized")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Episode tracking (per-env) ----------------------------------------
    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    total_episodes = 0
    best_score = -np.inf
    score_history = []

    # --- Initial reset ------------------------------------------------------
    observations = vec_env.reset()

    print("Starting training...\n")

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

                # Reset per-env tracking immediately
                episode_scores[i] = 0
                episode_steps[i] = 0

                score_history.append(score)
                avg_score = np.mean(score_history[-100:])
                total_episodes += 1

                # TensorBoard
                writer.add_scalar("Score/Episode", score, total_episodes)
                writer.add_scalar("Score/Average_100", avg_score, total_episodes)
                writer.add_scalar("Steps/Episode", steps, total_episodes)

                # Save best model
                if score > best_score:
                    best_score = score
                    agent.save_models()
                    print(
                        f"Episode {total_episodes:5d} | ★ BEST! Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Steps: {steps:3d}"
                    )
                elif total_episodes % 50 == 0:
                    print(
                        f"Episode {total_episodes:5d} | Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Best: {best_score:7.2f}"
                    )

                # Periodic checkpoint
                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.memory.save()
                    print(f"\nCheckpoint at {total_episodes}: Avg={avg_score:.2f} (buffer: {min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions)\n")

        # ---- Early stopping (checked after processing all envs) ------------
        if len(score_history) >= 100 and np.mean(score_history[-100:]) >= 3000:
            print(f"\n🎯 Target reached! Avg: {np.mean(score_history[-100:]):.2f}")
            break

        # ---- Learn ONCE per timestep (not per env) -------------------------
        agent.learn()

        observations = next_observations

    # --- Cleanup ------------------------------------------------------------
    vec_env.close()
    writer.close()
    agent.save_models()
    agent.memory.save()
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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Configuration
    ENV_NAME = "PickPlace"  # or "Stack", "NutAssembly"
    N_ENVS = 8              # in-process environments
    N_EPISODES = 20000

    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME}")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Replay buffer:  100,000 transitions")
    print(f"{'=' * 70}\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES)

    print("Next steps:")
    print("  1. python convert_float32_tflite.py ./checkpoints/td3/actor_td3 actor_pickplace.tflite")
    print("  2. Check if it fits ESP32 (should be ~230KB)")
    print("  3. If doesn't fit → Use QAT!")