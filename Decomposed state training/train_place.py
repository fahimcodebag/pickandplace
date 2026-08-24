#!/usr/bin/env python3
# Last updated: 2026-07-04 03:30 +0600
"""
Training script for the Place sub-policy (Stage 2 of decomposed RL).

Trains a small 64→32 MLP actor to lift, transport, and place the bread
into the target bin. Each episode starts with the trained grasp model
running until stable grasp, then the place policy takes over.

Uses subprocess-parallel vectorized environments.
Reuses networks.py, td3.py, buffer.py from the parent directory.
"""

import os
import sys

# Must be set before MuJoCo / robosuite are imported
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

# Add parent directory for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import multiprocessing as mp

import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
import torch
from torch.utils.tensorboard import SummaryWriter
from td3 import Agent
# LayerNorm critics + primacy-bias resets, the same remedy that was worth
# +20 to +30 points on the grasp stage. The actor is untouched (still the
# deployed 64->32), so nothing about the ESP32 artifact changes.
from td3_ln import Agent as AgentLN
from place_env_wrapper import PlaceGymWrapper


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_place_env(env_name="PickPlace", seed=None, grasp_chkpt_dir=None,
                   random_spawn=False):
    """Create a single robosuite environment with place-only reward."""
    if grasp_chkpt_dir is None:
        grasp_chkpt_dir = os.path.join(
            os.path.dirname(__file__), "..", "checkpoints", "td3_grasp"
        )

    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        # Raised from 500: one place episode consumes the grasp rollout
        # (<=200) + test-lift (20) + scripted curriculum carry (<=120) +
        # the place phase (PLACE_HORIZON=200) within a SINGLE raw episode, up
        # to ~540 steps. 700 keeps the raw horizon from cutting the place
        # phase short (which would show up as raw_env_horizon terminations).
        horizon=700,
        reward_shaping=False,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Spawn. The original training pinned this to a constant, matching the
    # fixed-spawn grasp model of the time. A transport policy trained that way
    # only ever sees objects held in ONE orientation, so it stalls when handed a
    # rotated grasp -- which is what the lift-certified grasp policy produces.
    # random_spawn leaves robosuite's native sampler alone: full position box
    # plus uniform z-rotation, the same distribution as grasp spawn level 2.0.
    if not random_spawn:
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

    # GymWrapper first (flattens obs_dict → 46-dim vector)
    raw_env = env
    gym_env = GymWrapper(raw_env)
    # PlaceGymWrapper sits on top: runs grasp policy on reset, place rewards on step
    place_env = PlaceGymWrapper(gym_env, raw_env, grasp_chkpt_dir)
    if seed is not None:
        place_env.seed(seed)
    return place_env


# ---------------------------------------------------------------------------
# Subprocess-parallel vectorized environment
# ---------------------------------------------------------------------------

# Globals to pass config to forked workers
_GRASP_CHKPT_DIR = None
_RANDOM_SPAWN = False


def _worker(remote, parent_remote, env_name, seed):
    """Worker loop that runs in a child process."""
    parent_remote.close()
    env = make_place_env(env_name, seed=seed, grasp_chkpt_dir=_GRASP_CHKPT_DIR,
                         random_spawn=_RANDOM_SPAWN)
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
    Each worker loads its own copy of the trained grasp agent for the
    grasp rollout during reset().
    """

    def __init__(self, env_name, n_envs=8):
        self.n_envs = n_envs
        self.closed = False

        ctx = mp.get_context("fork")

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(n_envs)]
        )

        print(f"Spawning {n_envs} subprocess environments (fork)...")
        print("  (Each worker loads a grasp agent for the reset rollout)")
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
        """Reset all environments in parallel."""
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
# Warmup exploration
# ---------------------------------------------------------------------------

def _warmup_action_batch(agent, n_envs, n_actions):
    """Hold-biased random exploration for the place-phase warmup.

    Agent.choose_action_batch()'s default warmup samples full-range uniform
    actions, which is the right call for the grasp stage (starts from a
    neutral, empty-handed pose). The place stage is different: every episode
    *starts already holding the object* (handed off from Stage 1), so
    full-range random actions on every DOF yank it loose almost immediately
    — flooding the replay buffer with degenerate "dropped, then idle for the
    rest of the horizon" transitions instead of useful lift/transport
    coverage.

    Pose dims (position + orientation) get small Gaussian noise so the arm
    drifts around gently instead of jerking violently. The gripper dim is
    biased toward "closed" (env convention: -1 = open, +1 = closed) so the
    grasp survives through most of warmup, letting the critic pretrain on
    informative held-object states before the actor starts acting.
    """
    low, high = agent.min_action, agent.max_action
    pose_dims = n_actions - 1

    pose = np.random.normal(scale=0.15, size=(n_envs, pose_dims))
    pose *= (high[:pose_dims] - low[:pose_dims]) / 2.0

    grip_range = high[-1] - low[-1]
    gripper = np.random.normal(
        loc=low[-1] + 0.9 * grip_range,   # biased near "closed"
        scale=0.15 * grip_range,
        size=(n_envs, 1),
    )

    actions = np.concatenate([pose, gripper], axis=1)
    actions = np.clip(actions, low, high)
    agent.time_step += n_envs
    return actions


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def _reason_tally(reasons, window):
    """Compact 'reason:count' summary over the last `window` episodes.

    Shows the mix of episode termination causes (success / timeout /
    fell_off_table / stalled_holding / grasp_handoff_failed / ...) so the
    dominant failure mode is visible directly instead of inferred from the
    score. Sorted most-frequent first.
    """
    from collections import Counter

    recent = reasons[-window:]
    if not recent:
        return "n/a"
    counts = Counter(recent)
    return "  ".join(
        f"{reason}:{n}" for reason, n in counts.most_common()
    )


def _preserve_best_models(chkpt_dir):
    """Copy the just-saved model files into chkpt_dir/best/.

    save_models() writes to the same files on BOTH new-bests and periodic
    checkpoints, so a periodic checkpoint of a degraded policy overwrites the
    best one (observed: a 17%-success checkpoint clobbered a 70%+ best). This
    snapshots the current (best) weights into an untouched subdir that periodic
    checkpoints never write to, so the best policy is always recoverable.
    """
    import glob
    import shutil

    best_dir = os.path.join(chkpt_dir, "best")
    os.makedirs(best_dir, exist_ok=True)
    for f in glob.glob(os.path.join(chkpt_dir, "*_td3")):
        try:
            shutil.copy2(f, os.path.join(best_dir, os.path.basename(f)))
        except OSError:
            pass


def _drop_summary(diag, window):
    """Summarize WHEN/HOW grasps are lost over the recent drop episodes.

    `diag` is a list of (reason, grasp_lost_step, max_lift_frac) tuples. This
    filters to the drop-type outcomes (timeout_dropped / fell_off_table) and
    reports the median step at which the grasp died and the median peak lift
    reached before losing it. The interpretation:

      - median drop step ~1-5 AND lift ~0  => the handoff grasp was never
        solid; the object is falling out the instant the place policy takes
        over. Fix: firm up the handoff (require a longer stable-grasp hold).
      - median drop step mid-horizon AND lift > 0 => a real grasp is shaken
        loose by transport motion. Fix: gentler translation / slower moves.
    """
    recent = [d for d in diag[-window:]
              if d[0] in ("timeout_dropped", "fell_off_table")]
    if not recent:
        return "no drops"
    steps = sorted(d[1] for d in recent)
    lifts = sorted(d[2] for d in recent)
    mid = len(recent) // 2
    med_step = steps[mid]
    med_lift = lifts[mid]
    return (f"{len(recent)} drops | median grasp_lost_step={med_step} | "
            f"median peak_lift_frac={med_lift:.2f}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(env_name="PickPlace", n_envs=8, n_episodes=10000,
          grasp_chkpt_dir=None, random_spawn=False, place_chkpt_dir=None,
          warm_start_from=None, critic_reset_every=0, layer_norm=False):
    """
    Train the Place sub-policy using TD3 with subprocess-parallel envs.

    Each episode starts with a grasp rollout (using the trained grasp model),
    then the place policy trains on the lift → transport → place phase.
    """
    global _GRASP_CHKPT_DIR, _RANDOM_SPAWN

    place_chkpt_dir = place_chkpt_dir or os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "td3_place"
    )
    grasp_chkpt_dir = grasp_chkpt_dir or os.path.join(
        os.path.dirname(__file__), "..", "checkpoints", "td3_grasp"
    )
    _GRASP_CHKPT_DIR = grasp_chkpt_dir
    _RANDOM_SPAWN = bool(random_spawn)

    print("=" * 70)
    print(f"PLACE MODEL TRAINING: {env_name}")
    print("=" * 70)
    print(f"Environments:   {n_envs} (subprocess-parallel, fork)")
    print(f"Total episodes: {n_episodes}")
    print(f"Network:        64 → 32 (actor & critic)")
    print(f"Grasp model:    {grasp_chkpt_dir}")
    print(f"Spawn:          {'NATIVE random (position + rotation)' if random_spawn else 'fixed'}")
    print(f"Checkpoints:    {place_chkpt_dir}")
    print("=" * 70 + "\n")

    # Verify grasp checkpoint exists
    grasp_actor_path = os.path.join(grasp_chkpt_dir, "actor_td3")
    if not os.path.exists(grasp_actor_path):
        print(f"ERROR: Grasp model checkpoint not found at {grasp_actor_path}")
        print("Train the grasp model first: python train_grasp.py")
        return None

    # --- Create environments ------------------------------------------------
    vec_env = SubprocVecEnv(env_name, n_envs)

    # --- Hyperparameters ----------------------------------------------------
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

    # Gripper (last dim) uses FULL exploration noise. The reduced-noise setting
    # made sense before the wrapper masked the gripper closed during transport;
    # now that the mask force-closes it whenever the object isn't over the bin,
    # throttling the noise no longer protects the grasp — it only starved
    # exploration of the RELEASE action (the first run took ~189 episodes to
    # discover release even at the easiest curriculum level). Over the target
    # the gripper is unmasked, so full noise there is exactly what we want to
    # discover release quickly.
    noise_scale = np.ones(n_actions, dtype=np.float32)

    # --- Agent --------------------------------------------------------------
    _Agent = AgentLN if (layer_norm or critic_reset_every) else Agent
    _extra = {"layer_norm": True} if _Agent is AgentLN else {}
    agent = _Agent(
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
        chkpt_dir=place_chkpt_dir,
        **_extra,
    )

    # --- Resume from checkpoint if available --------------------------------
    # warm_start_from takes precedence and seeds a FRESH run in a new
    # place_chkpt_dir from an existing policy -- retraining transport against a
    # different grasp distribution should not overwrite the artifact it is
    # seeded from, and should not silently resume a half-trained run.
    if warm_start_from:
        import torch as _T
        src = os.path.join(warm_start_from, "actor_td3")
        if not os.path.exists(src):
            raise FileNotFoundError(f"warm start actor not found: {src}")
        sd = _T.load(src, map_location="cpu")
        sd = {k: v for k, v in sd.items() if not k.startswith("log_std")}
        agent.actor.load_state_dict(sd)
        agent.target_actor.load_state_dict(sd)
        print(f"Warm-started actor from {warm_start_from} (critics fresh).")
    else:
        try:
            agent.load_models()
            print("Resumed network weights from saved checkpoint.")
        except Exception:
            print("No network checkpoint found, starting from scratch.")

    buf_path = os.path.join(place_chkpt_dir, "replay_buffer.npz")
    buf_loaded = agent.memory.load(buf_path)
    if buf_loaded:
        print(f"Resumed replay buffer ({min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions).\n")
        if agent.memory.mem_cntr >= agent.warmup:
            agent.time_step = agent.warmup
    else:
        print("No replay buffer found, starting fresh.\n")

    # --- Logging ------------------------------------------------------------
    log_dir = os.path.join(os.path.dirname(__file__), "..", "logs", f"{env_name}_place")
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    # --- Episode tracking (per-env) ----------------------------------------
    episode_scores = np.zeros(n_envs)
    episode_steps = np.zeros(n_envs, dtype=int)
    total_episodes = 0
    best_score = -np.inf   # display only — NOT the save criterion
    # Best-POLICY criterion: rolling difficulty x success. A single-episode
    # score record is meaningless under the curriculum (a success at frac 0.2
    # scores ~the same ~150 as one at frac 1.0), so score-based best froze at
    # an early easy-level episode while the true peak (71% success at frac
    # 1.00, ~ep 4300) went unsaved and was then clobbered by a periodic
    # checkpoint mid-collapse. metric = mean(frac, last 50) * mean(success,
    # last 50): a policy only sets a record by succeeding often AT difficulty.
    best_metric = 0.0
    score_history = []
    place_successes = []
    done_reasons = []
    drop_diag = []   # (reason, grasp_lost_step, max_lift_frac) per episode
    handoff_tries = []   # grasp+test-lift attempts to find a robust handoff
    curric_fracs = []    # per-episode curriculum difficulty (0.2 easy -> 1.0 full)

    # --- Initial reset ------------------------------------------------------
    observations = vec_env.reset()

    print("Starting place training...\n")

    while total_episodes < n_episodes:
        # ---- Batched action selection --------------------------------------
        if agent.time_step < agent.warmup:
            actions = _warmup_action_batch(agent, n_envs, n_actions)
        else:
            actions = agent.choose_action_batch(observations, noise_scale=noise_scale)

        # ---- Step all environments -----------------------------------------
        next_observations, rewards, dones, infos = vec_env.step(actions)

        # ---- Store transitions ---------------------------------------------
        for i in range(n_envs):
            episode_scores[i] += rewards[i]
            episode_steps[i] += 1

            # Store the action the env ACTUALLY executed (the wrapper masks the
            # gripper closed during transport), so the buffer's (s, a, r, s')
            # stays consistent. Falls back to the chosen action if unavailable.
            applied_action = infos[i].get("applied_action", actions[i])

            agent.remember(
                observations[i],
                applied_action,
                rewards[i],
                next_observations[i],
                dones[i],
            )

            # — Handle completed episodes ------------------------------------
            if dones[i]:
                score = episode_scores[i]
                steps = int(episode_steps[i])
                place_ok = infos[i].get("place_success", False)
                reason = infos[i].get("place_done_reason", "unknown")

                # Reset per-env tracking
                episode_scores[i] = 0
                episode_steps[i] = 0

                score_history.append(score)
                place_successes.append(1.0 if place_ok else 0.0)
                done_reasons.append(reason)
                drop_diag.append((
                    reason,
                    infos[i].get("grasp_lost_step", -1),
                    infos[i].get("max_lift_frac", 0.0),
                ))
                handoff_tries.append(infos[i].get("handoff_attempts", 0))
                curric_fracs.append(infos[i].get("curriculum_frac", 1.0))
                avg_score = np.mean(score_history[-100:])
                avg_place = np.mean(place_successes[-100:]) * 100
                total_episodes += 1

                # TensorBoard
                writer.add_scalar("Place/Score_Episode", score, total_episodes)
                writer.add_scalar("Place/Score_Avg100", avg_score, total_episodes)
                writer.add_scalar("Place/Steps_Episode", steps, total_episodes)
                writer.add_scalar("Place/Success_Rate_100", avg_place, total_episodes)

                # Save best model: difficulty-weighted rolling success record.
                # Wait for a full 50-episode window so early noise can't set
                # a fake record; +0.005 margin avoids disk churn on jitter.
                best_line = False
                if score > best_score:
                    best_score = score   # tracked for display only
                if len(place_successes) >= 50:
                    frac_50 = float(np.mean(curric_fracs[-50:]))
                    place_50 = float(np.mean(place_successes[-50:]))
                    metric = frac_50 * place_50
                    if metric > best_metric + 0.005:
                        best_metric = metric
                        agent.save_models()
                        _preserve_best_models(place_chkpt_dir)
                        best_line = True
                        print(
                            f"Episode {total_episodes:5d} | ★ BEST POLICY! "
                            f"frac50={frac_50:.2f} place50={place_50 * 100:4.1f}% "
                            f"metric={metric:.3f} | Score: {score:7.2f}"
                        )
                if not best_line and total_episodes % 50 == 0:
                    print(
                        f"Episode {total_episodes:5d} | Score: {score:7.2f} | "
                        f"Avg: {avg_score:7.2f} | Place%: {avg_place:5.1f}% | "
                        f"BestMetric: {best_metric:.3f}"
                    )
                    print(f"    last 50 outcomes: {_reason_tally(done_reasons, 50)}")
                    print(f"    drop diag (50):   {_drop_summary(drop_diag, 50)}")
                    _recent_tries = handoff_tries[-50:]
                    print(f"    handoff tries (50): avg={np.mean(_recent_tries):.2f} "
                          f"max={max(_recent_tries)}")
                    _recent_fracs = curric_fracs[-50:]
                    print(f"    curriculum frac (50): min={min(_recent_fracs):.2f} "
                          f"mean={np.mean(_recent_fracs):.2f} max={max(_recent_fracs):.2f}")

                # Periodic checkpoint
                if total_episodes % 500 == 0:
                    agent.save_models()
                    agent.memory.save(buf_path)
                    print(
                        f"\nCheckpoint at {total_episodes}: Avg={avg_score:.2f} "
                        f"Place%={avg_place:.1f}% "
                        f"(buffer: {min(agent.memory.mem_cntr, agent.memory.mem_size):,} transitions)"
                    )
                    print(f"  last 100 outcomes: {_reason_tally(done_reasons, 100)}")
                    print(f"  drop diag (100):   {_drop_summary(drop_diag, 100)}\n")

        # ---- Early stopping: 90% place success over last 100 episodes ------
        # Only valid at full difficulty — otherwise the curriculum's easy
        # handoffs would trigger a false victory. Require the recent episodes
        # to be at (near) curriculum_frac=1.0, i.e. the true spawn handoff.
        if (len(place_successes) >= 100
                and np.mean(place_successes[-100:]) >= 0.90
                and np.mean(curric_fracs[-100:]) >= 0.99):
            print(
                f"\n🎯 Place target reached at full difficulty! "
                f"Success rate: {np.mean(place_successes[-100:]) * 100:.1f}%"
            )
            break

        # ---- Learn ONCE per timestep ---------------------------------------
        agent.learn()
        if (critic_reset_every and agent.learn_step_cntr > 0
                and agent.learn_step_cntr % critic_reset_every == 0
                and hasattr(agent, 'reset_critic_heads')):
            agent.reset_critic_heads()
            print(f'  [primacy-bias reset @ {agent.learn_step_cntr} updates]',
                  flush=True)

        observations = next_observations

    # --- Cleanup ------------------------------------------------------------
    vec_env.close()
    writer.close()
    agent.save_models()
    agent.memory.save(buf_path)
    print("Replay buffer saved.")

    print("\n" + "=" * 70)
    print("PLACE TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total episodes: {total_episodes}")
    print(f"Best score:     {best_score:.2f}")
    print(f"Best metric:    {best_metric:.3f} (mean frac50 x place50; policy in best/)")
    if score_history:
        print(f"Final average:  {np.mean(score_history[-100:]):.2f}")
    if place_successes:
        print(f"Final place %:  {np.mean(place_successes[-100:]) * 100:.1f}%")
    print("=" * 70 + "\n")

    print("Next steps:")
    print("  1. python test_place.py --episodes 20")
    print("  2. Convert both models to TFLite for ESP32 deployment")
    print("  3. Implement FSM meta-controller on ESP32")

    return agent


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    _p = argparse.ArgumentParser()
    _p.add_argument("--env-name", default="PickPlace")
    _p.add_argument("--n-envs", type=int, default=2)
    _p.add_argument("--episodes", type=int, default=10000)
    _p.add_argument("--grasp-chkpt-dir", default=None,
                    help="Grasp stage to train transport against. Was "
                         "hardcoded to checkpoints/td3_grasp.")
    _p.add_argument("--random-spawn", action="store_true",
                    help="Native robosuite spawn (position + rotation) instead "
                         "of the fixed pose the original training used.")
    _p.add_argument("--place-chkpt-dir", default=None)
    _p.add_argument("--critic-reset-every", type=int, default=0,
                    help="Reinitialise critic output layers every N updates "
                         "(Nikishin et al.). The place stage shows the same "
                         "decay as the grasp stage: 79%% at ep 1500, 8%% by 2500.")
    _p.add_argument("--layer-norm", action="store_true")
    _p.add_argument("--warm-start-from", default=None,
                    help="Place checkpoint dir to seed the actor from.")
    _a = _p.parse_args()

    ENV_NAME = _a.env_name
    N_ENVS = _a.n_envs
    N_EPISODES = _a.episodes

    print(f"\n{'=' * 70}")
    print("PLACE SUB-POLICY TRAINING (Decomposed)")
    print(f"{'=' * 70}")
    print(f"  Task:           {ENV_NAME} (Place only, grasp handled by Stage 1)")
    print(f"  Environments:   {N_ENVS} (subprocess-parallel)")
    print(f"  Total episodes: {N_EPISODES}")
    print(f"  Network:        64 → 32 MLP")
    print(f"  Replay buffer:  200,000 transitions")
    print(f"{'=' * 70}\n")

    agent = train(ENV_NAME, N_ENVS, N_EPISODES,
                  grasp_chkpt_dir=_a.grasp_chkpt_dir,
                  random_spawn=_a.random_spawn,
                  place_chkpt_dir=_a.place_chkpt_dir,
                  warm_start_from=_a.warm_start_from,
                  critic_reset_every=_a.critic_reset_every,
                  layer_norm=_a.layer_norm)
