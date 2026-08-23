#!/usr/bin/env python3
"""Random-spawn grasp retraining — unified, algorithm-agnostic trainer.

Supersedes `train_grasp_rand.py` for the §9 algorithm comparison. One training
loop drives TD3, TD3+LayerNorm, SAC, and PPO over the identical environment,
reward, spawn distribution, logging, and best-model criterion, so a run
difference is an algorithm difference.

WHY THIS EXISTS — the update-to-data trap
-----------------------------------------
`train_grasp_rand.py` calls `agent.learn()` once per VECTORIZED step. At the
old N_ENVS=2 that is 0.5 gradient steps per environment step. Simply raising
N_ENVS to 20 on the new machine would silently cut the update-to-data ratio
10x — a different algorithm, not a faster run, and it would have quietly
invalidated every comparison against the existing 86%/0.83 baseline. Here the
ratio is explicit: `--updates-per-step` defaults to `n_envs // 2`, exactly
preserving the historical 0.5 updates/env-step. Change it deliberately, and
note (§9.3) that RAISING it worsens the plasticity loss this stage suffers
from unless paired with --layer-norm / --critic-reset-every.

USAGE
  python3 train_rand.py --algo td3    --n-envs 20 --seed 0 --tag phaseA
  python3 train_rand.py --algo td3_ln --n-envs 20 --seed 0 --tag phaseA
  python3 train_rand.py --algo sac    --n-envs 20 --seed 0 --tag phaseA
  python3 train_rand.py --algo ppo    --n-envs 20 --seed 0 --tag phaseA

Checkpoints/logs are keyed by algo+tag+seed, so variants run concurrently
without collision.
"""

import argparse
import csv
import glob
import os
import shutil
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, _ROOT)

import multiprocessing as mp

import numpy as np
import torch as T
from torch.utils.tensorboard import SummaryWriter

from grasp_spawn_wrapper import make_spawn_grasp_env, SpawnCurriculumGraspWrapper
from grasp_diagnostics import GraspDiagnosticsWrapper, DONE_REASONS
from training_probe import probe


# ---------------------------------------------------------------------------
# Vectorized env (spawn level passed explicitly — no module-global)
# ---------------------------------------------------------------------------

def _worker(remote, parent_remote, env_name, seed, spawn_level,
            require_lift=False):
    parent_remote.close()
    env = GraspDiagnosticsWrapper(
        make_spawn_grasp_env(env_name, seed=seed, curriculum=False,
                             require_lift=require_lift,
                             level=spawn_level))
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
            remote.send(env.reset())
        elif cmd == "get_spaces":
            remote.send((env.observation_space, env.action_space))
        elif cmd == "close":
            env.close()
            remote.close()
            break


class SubprocVecEnv:
    def __init__(self, env_name, n_envs, spawn_level, seed0=0,
                 require_lift=False):
        self.n_envs = n_envs
        self.closed = False
        ctx = mp.get_context("fork")
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in range(n_envs)])
        print(f"Spawning {n_envs} subprocess environments (fork)...",
              flush=True)
        self.processes = []
        for i, (wr, r) in enumerate(zip(work_remotes, self.remotes)):
            p = ctx.Process(target=_worker,
                            args=(wr, r, env_name, seed0 + i, spawn_level,
                                  require_lift),
                            daemon=True)
            p.start()
            wr.close()
            self.processes.append(p)
        self.remotes[0].send(("get_spaces", None))
        self.observation_space, self.action_space = self.remotes[0].recv()
        print(f"{n_envs} environments ready.\n", flush=True)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return np.array([r.recv() for r in self.remotes])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        out = [r.recv() for r in self.remotes]
        return (np.array([o[0] for o in out]), np.array([o[1] for o in out]),
                np.array([o[2] for o in out]), [o[3] for o in out])

    def close(self):
        if self.closed:
            return
        for r in self.remotes:
            try:
                r.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join(timeout=5)
        self.closed = True


# ---------------------------------------------------------------------------
# Agent construction
# ---------------------------------------------------------------------------

def build_agent(algo, vec_env, chkpt_dir, args):
    common = dict(alpha=args.lr_actor, beta=args.lr_critic,
                  input_dims=vec_env.observation_space.shape, tau=0.005,
                  env=vec_env, n_actions=vec_env.action_space.shape[0],
                  layer1_size=64, layer2_size=32, batch_size=args.batch_size,
                  chkpt_dir=chkpt_dir)
    if algo == "td3":
        import td3
        return td3.Agent(max_size=args.buffer_size, warmup=args.warmup, **common)
    if algo == "td3_ln":
        import td3_ln
        return td3_ln.Agent(max_size=args.buffer_size, warmup=args.warmup,
                            layer_norm=True, **common)
    if algo == "sac":
        import sac
        return sac.Agent(max_size=args.buffer_size, warmup=args.warmup,
                         layer_norm=args.layer_norm, **common)
    if algo == "ppo":
        import ppo
        return ppo.Agent(rollout_steps=args.rollout_steps,
                         n_envs=vec_env.n_envs, n_epochs=args.ppo_epochs,
                         layer_norm=args.layer_norm, **common)
    raise ValueError(f"unknown algo {algo}")


def warm_start(agent, algo, chkpt_dir, source_dir):
    """Seed the actor trunk from the proven fixed-spawn grasp model.

    TD3 variants load the full checkpoint set. SAC/PPO load the fixed-spawn
    actor's fc1/fc2/output into their own trunk+mean head with strict=False —
    the extra log_std parameters keep their initialization. The source
    checkpoint is never written to.
    """
    if os.path.exists(os.path.join(chkpt_dir, "actor_td3")):
        try:
            agent.load_models()
            return "resumed"
        except Exception:
            pass
    src = os.path.join(source_dir, "actor_td3")
    if not os.path.exists(src):
        return "scratch"
    state = T.load(src, map_location=agent.actor.device)
    if algo in ("td3", "td3_ln"):
        os.makedirs(chkpt_dir, exist_ok=True)
        for f in glob.glob(os.path.join(source_dir, "*_td3")):
            shutil.copy2(f, os.path.join(chkpt_dir, os.path.basename(f)))
        agent.load_models()
    else:
        missing, unexpected = agent.actor.load_state_dict(state, strict=False)
        if unexpected:
            print(f"  (ignored source keys: {list(unexpected)})")
    return "warm-started"


def preserve_best(chkpt_dir):
    best = os.path.join(chkpt_dir, "best")
    os.makedirs(best, exist_ok=True)
    for f in glob.glob(os.path.join(chkpt_dir, "*_td3")):
        shutil.copy2(f, os.path.join(best, os.path.basename(f)))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    run_id = f"{args.algo}_{args.tag}_s{args.seed}"
    chkpt_dir = os.path.join(_ROOT, "checkpoints", f"td3_grasp_rand_{run_id}")
    source_dir = (args.warm_start_from if os.path.isabs(args.warm_start_from)
                  else os.path.join(_ROOT, args.warm_start_from))
    log_dir = os.path.join(_ROOT, "logs", f"grasp_rand_{run_id}")
    os.makedirs(chkpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    updates_per_step = (args.updates_per_step if args.updates_per_step is not None
                        else max(1, args.n_envs // 2))
    level_max = SpawnCurriculumGraspWrapper._LEVEL_MAX

    np.random.seed(args.seed)
    T.manual_seed(args.seed)

    print("=" * 72)
    print(f"RANDOM-SPAWN GRASP RETRAINING  |  algo={args.algo}  run={run_id}")
    print("=" * 72)
    print(f"  Envs                : {args.n_envs} (subprocess-parallel, fork)")
    print(f"  Spawn level         : {args.spawn_level} (1.0 = full position "
          f"box, 2.0 = + z-rotation)")
    if args.algo == "ppo":
        print(f"  Rollout             : {args.rollout_steps} steps/env x "
              f"{args.ppo_epochs} epochs (on-policy; no replay)")
    else:
        print(f"  Updates/vec-step    : {updates_per_step} "
              f"({updates_per_step / args.n_envs:.3f} per env-step; "
              f"historical baseline 0.5)")
        print(f"  Buffer              : {args.buffer_size:,}")
    print(f"  Network             : 64 -> 32 (deployment-fixed)")
    print(f"  Warm start          : {args.warm_start_from or '(scratch)'}")
    print(f"  Checkpoints         : {chkpt_dir}")
    print("=" * 72 + "\n")

    vec_env = SubprocVecEnv("PickPlace", args.n_envs, args.spawn_level,
                            seed0=args.seed * 1000,
                            require_lift=args.require_lift)
    agent = build_agent(args.algo, vec_env, chkpt_dir, args)

    status = warm_start(agent, args.algo, chkpt_dir, source_dir)
    print(f"Init: {status} (source: {source_dir})")
    if status != "scratch" and args.algo in ("td3", "td3_ln", "sac"):
        # A competent actor exists: skip random-action warmup, which would
        # only fill the buffer with junk (rationale from train_grasp_rand.py).
        agent.time_step = agent.warmup

    if args.algo != "ppo":
        buf_path = os.path.join(chkpt_dir, "replay_buffer.npz")
        if agent.memory.load(buf_path):
            n = min(agent.memory.mem_cntr, agent.memory.mem_size)
            print(f"Resumed replay buffer ({n:,} transitions).")
            agent.time_step = agent.warmup
        else:
            print("Fresh replay buffer.")
    print()

    writer = SummaryWriter(log_dir=log_dir)
    csv_path = os.path.join(log_dir, "episodes.csv")
    csv_f = open(csv_path, "a", newline="")
    csv_w = csv.writer(csv_f)
    if csv_f.tell() == 0:
        csv_w.writerow(["episode", "score", "steps", "grasp_success",
                        "done_reason", "spawn_level", "spawn_x", "spawn_y",
                        "spawn_yaw", "grasp_steps", "first_grasp_step",
                        "max_grasp_run", "min_reach_dist",
                        "avg_score_100", "success_100", "best_metric",
                        "env_steps", "grad_steps", "wall_s"])

    # Agent-internal metrics on their own cadence (see training_probe.py).
    probe_path = os.path.join(log_dir, "probe.csv")
    probe_f = open(probe_path, "a", newline="")
    probe_w = csv.writer(probe_f)
    probe_header_written = probe_f.tell() > 0

    # Run manifest so every plot can be labelled and reproduced.
    import json
    with open(os.path.join(log_dir, "manifest.json"), "w") as mf:
        json.dump({**vars(args), "run_id": run_id,
                   "updates_per_step": updates_per_step,
                   "chkpt_dir": chkpt_dir, "level_max": level_max,
                   "done_reasons": list(DONE_REASONS)}, mf, indent=2)

    ep_scores = np.zeros(args.n_envs)
    ep_steps = np.zeros(args.n_envs, dtype=int)
    total_episodes = 0
    env_steps = 0
    best_metric = 0.0
    score_hist, success_hist, level_hist = [], [], []
    reason_hist = []
    t0 = time.time()

    observations = vec_env.reset()
    print("Training...\n", flush=True)

    while total_episodes < args.episodes:
        actions = agent.choose_action_batch(observations)
        next_observations, rewards, dones, infos = vec_env.step(actions)
        env_steps += args.n_envs

        for i in range(args.n_envs):
            ep_scores[i] += rewards[i]
            ep_steps[i] += 1
            if args.algo == "ppo":
                agent.remember(observations[i], actions[i], rewards[i],
                               next_observations[i], dones[i], env_idx=i)
            else:
                agent.remember(observations[i], actions[i], rewards[i],
                               next_observations[i], dones[i])

            if not dones[i]:
                continue

            score, steps = ep_scores[i], int(ep_steps[i])
            ep_scores[i] = ep_steps[i] = 0
            inf = infos[i]
            grasp_ok = bool(inf.get("grasp_success", False))
            level = float(inf.get("spawn_level", args.spawn_level))
            reason = inf.get("done_reason", "unknown")
            reason_hist.append(reason)

            score_hist.append(score)
            success_hist.append(1.0 if grasp_ok else 0.0)
            level_hist.append(level)
            total_episodes += 1
            avg_score = float(np.mean(score_hist[-100:]))
            succ_100 = float(np.mean(success_hist[-100:])) * 100

            writer.add_scalar("GraspRand/Score_Episode", score, total_episodes)
            writer.add_scalar("GraspRand/Score_Avg100", avg_score, total_episodes)
            writer.add_scalar("GraspRand/Success_Rate_100", succ_100, total_episodes)
            writer.add_scalar("GraspRand/Spawn_Level", level, total_episodes)
            # Done-reason tally over the last 50 episodes — the §4 diagnostic
            # that separates failure modes identical scores would conflate.
            if len(reason_hist) >= 50:
                win = reason_hist[-50:]
                for rname in DONE_REASONS:
                    writer.add_scalar(f"DoneReason/{rname}",
                                      win.count(rname) / 50.0, total_episodes)
            if args.algo == "sac":
                writer.add_scalar("GraspRand/SAC_Alpha",
                                  float(agent.alpha), total_episodes)

            # Difficulty-weighted rolling best (§3 Hurdle 9): a policy sets a
            # record only by succeeding often AT difficulty.
            is_best = False
            if len(success_hist) >= 50:
                lvl50 = float(np.mean(level_hist[-50:])) / level_max
                suc50 = float(np.mean(success_hist[-50:]))
                metric = lvl50 * suc50
                writer.add_scalar("GraspRand/Metric", metric, total_episodes)
                if metric > best_metric + 0.005:
                    best_metric = metric
                    agent.save_models()
                    preserve_best(chkpt_dir)
                    is_best = True
                    print(f"Ep {total_episodes:5d} | * BEST  "
                          f"level50={lvl50 * level_max:.2f} "
                          f"grasp50={suc50 * 100:5.1f}% metric={metric:.3f} "
                          f"| score {score:7.2f}", flush=True)

            csv_w.writerow([
                total_episodes, f"{score:.3f}", steps, int(grasp_ok), reason,
                f"{level:.3f}",
                f"{inf.get('spawn_x', float('nan')):.4f}",
                f"{inf.get('spawn_y', float('nan')):.4f}",
                f"{inf.get('spawn_yaw', float('nan')):.4f}",
                inf.get("grasp_steps", -1), inf.get("first_grasp_step", -1),
                inf.get("max_grasp_run", -1),
                f"{inf.get('min_reach_dist', -1.0):.4f}",
                f"{avg_score:.3f}", f"{succ_100:.2f}", f"{best_metric:.4f}",
                env_steps, getattr(agent, "learn_step_cntr", 0),
                f"{time.time() - t0:.1f}"])

            if total_episodes % args.probe_every == 0:
                m = probe(agent, batch=min(args.batch_size, 512))
                m["episode"] = total_episodes
                m["env_steps"] = env_steps
                m["success_100"] = succ_100
                if not probe_header_written:
                    probe_w.writerow(list(m.keys()))
                    probe_header_written = True
                probe_w.writerow([f"{v:.6g}" if isinstance(v, float) else v
                                  for v in m.values()])
                probe_f.flush()
                for k, v in m.items():
                    if k not in ("episode", "env_steps"):
                        writer.add_scalar(f"Probe/{k}", v, total_episodes)

            if not is_best and total_episodes % 50 == 0:
                sps = env_steps / max(1e-9, time.time() - t0)
                win = reason_hist[-50:]
                tally = " ".join(f"{r.replace('timeout_', 't_')}={win.count(r)}"
                                 for r in DONE_REASONS if win.count(r))
                print(f"Ep {total_episodes:5d} | score {score:7.2f} | "
                      f"avg100 {avg_score:7.2f} | grasp100 {succ_100:5.1f}% | "
                      f"best {best_metric:.3f} | {sps:6.0f} st/s", flush=True)
                print(f"    done(50): {tally}", flush=True)
                csv_f.flush()

            if total_episodes % 500 == 0 and args.algo != "ppo":
                agent.save_models()
                agent.memory.save(buf_path)

        # --- gradient updates -------------------------------------------------
        if args.algo == "ppo":
            agent.learn()                    # self-gates on rollout length
        else:
            for _ in range(updates_per_step):
                agent.learn()

        if (args.critic_reset_every and args.algo != "ppo"
                and agent.learn_step_cntr > 0
                and agent.learn_step_cntr % args.critic_reset_every == 0
                and hasattr(agent, "reset_critic_heads")):
            agent.reset_critic_heads()
            print(f"  [primacy-bias reset of critic heads @ "
                  f"{agent.learn_step_cntr} updates]", flush=True)

        observations = next_observations

        if (len(success_hist) >= 100
                and float(np.mean(success_hist[-100:])) >= args.target_success):
            print(f"\nTarget reached: {np.mean(success_hist[-100:]) * 100:.1f}% "
                  f"over 100 episodes at spawn level {args.spawn_level}.")
            break

    vec_env.close()
    agent.save_models()
    if args.algo != "ppo":
        agent.memory.save(buf_path)
    writer.close()
    csv_f.close()
    probe_f.close()

    print("\n" + "=" * 72)
    print(f"DONE  {run_id}")
    print(f"  Episodes     : {total_episodes}")
    print(f"  Env steps    : {env_steps:,}  ({time.time() - t0:.0f}s)")
    print(f"  Best metric  : {best_metric:.3f}  (policy in {chkpt_dir}/best)")
    if success_hist:
        print(f"  Final grasp% : {np.mean(success_hist[-100:]) * 100:.1f}")
    print(f"  CSV          : {csv_path}")
    print(f"  Probe CSV    : {probe_path}")
    print(f"  Manifest     : {os.path.join(log_dir, 'manifest.json')}")
    print("=" * 72 + "\n")
    return best_metric


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algo", choices=["td3", "td3_ln", "sac", "ppo"],
                   default="td3")
    p.add_argument("--n-envs", type=int, default=20)
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="phaseA")
    p.add_argument("--spawn-level", type=float, default=1.0,
                   help="1.0 = full position box (Phase A); 2.0 = + rotation")
    p.add_argument("--updates-per-step", type=int, default=None,
                   help="gradient steps per VECTORIZED step "
                        "(default n_envs//2, preserving the historical "
                        "0.5 updates per env-step)")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--buffer-size", type=int, default=200000)
    p.add_argument("--warmup", type=int, default=10000)
    p.add_argument("--lr-actor", type=float, default=3e-4)
    p.add_argument("--lr-critic", type=float, default=3e-4)
    p.add_argument("--layer-norm", action="store_true",
                   help="LayerNorm critics for sac/ppo (td3_ln has it always)")
    p.add_argument("--critic-reset-every", type=int, default=0,
                   help="primacy-bias reset of critic heads every N updates "
                        "(0 = off)")
    p.add_argument("--rollout-steps", type=int, default=512, help="PPO only")
    p.add_argument("--ppo-epochs", type=int, default=10, help="PPO only")
    p.add_argument("--target-success", type=float, default=0.85)
    p.add_argument("--warm-start-from",
                   default="checkpoints/td3_grasp_rand_best_0355_backup",
                   help="checkpoint dir to seed the actor from. Default is the "
                        "best random-spawn artifact (metric 0.355, 86%% at "
                        "level 0.83) that §9.3 reseeds Phase A from; pass "
                        "checkpoints/td3_grasp for the fixed-spawn 95%% model, "
                        "or '' to train from scratch.")
    p.add_argument("--require-lift", action="store_true",
                   help="Certify grasps with a scripted lift before counting "
                        "them successful (8-step hold + 20-step lift, >=3cm "
                        "rise, still grasped) -- the same bar the place stage "
                        "applies at handoff. Without it the metric accepts "
                        "momentary contact: 86%% of grasps passed the old "
                        "criterion but only 43%% survived handoff.")
    p.add_argument("--probe-every", type=int, default=25,
                   help="episodes between agent-internal probes "
                        "(critic Q stats, action saturation, SAC alpha)")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
