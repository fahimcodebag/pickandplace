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

# Per-term reward columns. The wrapper accumulates these inside
# _grasp_reward itself and emits them in info["reward_terms"] on the
# terminal step, so this cannot drift from the reward under test. A run
# without the instrumented wrapper simply writes blanks.
_RTERMS = ["reach", "grip_close", "grasp_hold", "success_bonus",
           "partial_credit", "align_shape", "dense_align", "builtin",
           "idle", "drop", "away"]


def _rterm_cols(inf):
    t = inf.get("reward_terms") or {}
    out = [f"{t[k]:.2f}" if k in t else "" for k in _RTERMS]
    out += [t.get("n_drops", ""), t.get("n_grasped_steps", "")]
    ga, lr = inf.get("grip_align"), inf.get("lift_rise")
    out += [f"{ga:.4f}" if ga is not None else "",
            f"{lr:.4f}" if lr is not None else ""]
    return out


def _worker(remote, parent_remote, env_name, seed, spawn_level,
            require_lift=False, align_grip=False, reward_v2=False,
            dense_align=False, builtin_reward=False):
    parent_remote.close()
    env = GraspDiagnosticsWrapper(
        make_spawn_grasp_env(env_name, seed=seed, curriculum=False,
                             require_lift=require_lift,
                             align_grip=align_grip,
                             reward_v2=reward_v2,
                             dense_align=dense_align,
                             builtin_reward=builtin_reward,
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
                 require_lift=False, align_grip=False, reward_v2=False,
                 dense_align=False, builtin_reward=False):
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
                                  require_lift, align_grip, reward_v2,
                                  dense_align, builtin_reward),
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
                  layer1_size=args.fc1, layer2_size=args.fc2,
                  batch_size=args.batch_size,
                  chkpt_dir=chkpt_dir)
    # The actor may be wider than the critics (see td3.Agent). Critics never
    # deploy, so widening them buys nothing and would break Net2WiderNet
    # through the td3_ln LayerNorm.
    wide = {}
    if getattr(args, "actor_fc1", None):
        wide = dict(actor_layer1=args.actor_fc1, actor_layer2=args.actor_fc2)
    if algo == "td3":
        import td3
        return td3.Agent(max_size=args.buffer_size, warmup=args.warmup,
                         **wide, **common)
    if algo == "td3_ln":
        import td3_ln
        return td3_ln.Agent(max_size=args.buffer_size, warmup=args.warmup,
                            layer_norm=True, **wide, **common)
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


def warm_start(agent, algo, chkpt_dir, source_dir, actor_only=False):
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
    if actor_only:
        # Critics are a different width from the source, so their checkpoints
        # cannot be loaded. Take the actor (and its target) only and let the
        # critics initialise fresh. Legitimate because critics never deploy
        # (Sec 7) and are rebuilt from an empty replay buffer every run anyway
        # -- no run in this project inherits a buffer.
        agent.actor.load_state_dict(state, strict=False)
        agent.target_actor.load_state_dict(state, strict=False)
        return "warm-started (actor only, critics fresh)"
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
    _a1 = args.actor_fc1 or args.fc1
    _a2 = args.actor_fc2 or args.fc2
    print(f"  Actor               : {_a1} -> {_a2}"
          f"  ({46*_a1 + _a1*_a2 + _a2*7:,} weights)"
          f"{'  [critics %d->%d]' % (args.fc1, args.fc2) if (_a1, _a2) != (args.fc1, args.fc2) else ''}")
    print(f"  Warm start          : {args.warm_start_from or '(scratch)'}")
    print(f"  Checkpoints         : {chkpt_dir}")
    print("=" * 72 + "\n")

    vec_env = SubprocVecEnv("PickPlace", args.n_envs, args.spawn_level,
                            seed0=args.seed * 1000,
                            require_lift=args.require_lift,
                            align_grip=args.align_grip,
                            reward_v2=args.reward_v2,
                            dense_align=args.dense_align,
                            builtin_reward=args.builtin_reward)
    agent = build_agent(args.algo, vec_env, chkpt_dir, args)
    # Weight-range regularisation for per-tensor INT8. See td3.Agent.
    # _clip_actor_weights: max/std IS the quantisation cost under one scale per
    # tensor. 0.0 leaves the update path byte-identical to the bi_s0 baseline.
    if getattr(args, "actor_fakequant", False):
        if not hasattr(agent, "enable_actor_fakequant"):
            raise SystemExit(f"--actor-fakequant unsupported by algo={args.algo}")
        agent.enable_actor_fakequant()
        print("  Actor QAT           : per-tensor INT8 fake-quant, STE grads")
    if getattr(args, "actor_wclip", 0.0):
        if not hasattr(agent, "actor_wclip"):
            raise SystemExit(f"--actor-wclip unsupported by algo={args.algo}")
        agent.actor_wclip = args.actor_wclip
        print(f"  Actor weight clip   : |w| <= {args.actor_wclip} * std(w) "
              f"per Linear tensor (bi_s0 baseline fc1 9.96 / fc2 10.82)")

    status = warm_start(agent, args.algo, chkpt_dir, source_dir,
                        actor_only=args.warm_start_actor_only)
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
                        # Per-term reward accounting (see reward_audit.py).
                        # score alone shows THAT reward moved; these show
                        # which term moved it, which is what a reward
                        # regression actually needs to be diagnosed.
                        "r_reach", "r_grip_close", "r_grasp_hold",
                        "r_success_bonus", "r_partial_credit",
                        "r_align_shape", "r_dense_align", "r_builtin",
                        "r_idle", "r_drop", "r_away",
                        "n_drops", "n_grasped_steps",
                        "grip_align", "lift_rise",
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
            #
            # The window was 50 episodes, which has SE ~6.5 points at p=0.7. A
            # run produces ~40 such windows and this keeps the MAXIMUM, so the
            # saved checkpoint is systematically a lucky window rather than a
            # good policy -- classic max-of-noisy-estimates bias. Measured on
            # the gripfix runs, the window that triggered best/ overstated the
            # checkpoint's true 800-episode rate by +27 / +8 / +4 points across
            # seeds. Worse, a lucky-but-weak window claims the slot and blocks
            # a genuinely better policy later in the run from being saved.
            #
            # --best-window widens it (SE ~3.2 at 200). Selection still uses
            # training data, so it remains noisy; it is just far less biased.
            is_best = False
            bw = args.best_window
            if len(success_hist) >= bw:
                lvl50 = float(np.mean(level_hist[-bw:])) / level_max
                suc50 = float(np.mean(success_hist[-bw:]))
                metric = lvl50 * suc50
                writer.add_scalar("GraspRand/Metric", metric, total_episodes)
                if metric > best_metric + args.best_margin:
                    best_metric = metric
                    agent.save_models()
                    preserve_best(chkpt_dir)
                    is_best = True
                    print(f"Ep {total_episodes:5d} | * BEST  "
                          f"level{bw}={lvl50 * level_max:.2f} "
                          f"grasp{bw}={suc50 * 100:5.1f}% metric={metric:.3f} "
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
                *_rterm_cols(inf),
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
    p.add_argument("--builtin-reward", action="store_true",
                   help="Use robosuite's own staged shaped reward per step "
                        "instead of this stage's custom one. The monolithic v7 "
                        "policy trained on it and certifies 98.7%% of grasps "
                        "against this stage's 79.2%%. Its structural difference "
                        "is a DENSE LIFT TERM -- reward rises continuously with "
                        "object height once grasped -- where this stage has "
                        "none and treats lift as a terminal binary.")
    p.add_argument("--dense-align", action="store_true",
                   help="Per-step alignment penalty during the approach. The "
                        "terminal reward already correlates +0.85 with grip "
                        "alignment, but the CRITIC's Q correlates -0.004 -- so "
                        "the actor gets no gradient toward rotating. This puts "
                        "the signal in every approach transition instead of "
                        "once at termination. Deliberately not potential-"
                        "based. Primary metric is corr(Q, alignment), not "
                        "success rate.")
    p.add_argument("--reward-v2", action="store_true",
                   help="Three reward fixes priced on recorded episodes before "
                        "training (Results/reward_audit/): (A) success bonus "
                        "graded by grip quality, W*(1+q) instead of a flat W "
                        "-- the old bonus was sd 0.000 across 290 successes "
                        "while quality varied 8-fold; (C) hold income capped "
                        "at 12 paid steps, which makes flicker-farming "
                        "impossible rather than merely unprofitable; (D) "
                        "P_IDLE -0.4 -> -1.1 so parking near the object is "
                        "strictly negative. Margin of a certified grasp over "
                        "the best failure mode: 1.62x -> 3.72x.")
    p.add_argument("--align-grip", action="store_true",
                   help="Reward closing the fingers on a flat face of the "
                        "object rather than a corner. The handoff diagnostic "
                        "found 24%% of grasps are corner grips; they pass lift "
                        "certification but cost 20-26 points end-to-end "
                        "(Results/handoff_diagnostic.txt). Adds potential-"
                        "based shaping during the approach plus a success-"
                        "bonus multiplier in [0.5, 1.0].")
    p.add_argument("--require-lift", action="store_true",
                   help="Certify grasps with a scripted lift before counting "
                        "them successful (8-step hold + 20-step lift, >=3cm "
                        "rise, still grasped) -- the same bar the place stage "
                        "applies at handoff. Without it the metric accepts "
                        "momentary contact: 86%% of grasps passed the old "
                        "criterion but only 43%% survived handoff.")
    p.add_argument("--fc1", type=int, default=64,
                   help="Actor hidden 1. 64 is the deployed size (5.2k params, "
                        "8KB INT8). ESP32 headroom is large: 4MB flash, 320KB "
                        "SRAM, 40KB arena, 9.5ms of a 50ms budget -- 256/128 "
                        "(~45k params) fits comfortably.")
    p.add_argument("--fc2", type=int, default=32, help="Hidden 2 (critics).")
    p.add_argument("--actor-fc1", type=int, default=None,
                   help="Actor hidden 1, if the actor is WIDER than the critics. "
                        "Pair with a Net2WiderNet checkpoint from net2wider.py: a "
                        "wider actor cannot inherit the 5-stage warm-start "
                        "curriculum by shape, and cold starts fail at every width.")
    p.add_argument("--actor-fc2", type=int, default=None, help="Actor hidden 2.")
    p.add_argument("--warm-start-actor-only", action="store_true",
                   help="Load only the actor from --warm-start-from and leave the "
                        "critics freshly initialised. Needed when the critics are a "
                        "different width from the source, e.g. a small deployable "
                        "actor paired with large host-side critics.")
    p.add_argument("--actor-fakequant", action="store_true",
                   help="QAT inside the RL loop: per-tensor symmetric INT8 "
                        "fake-quant on actor weights with a straight-through "
                        "estimator, so the RL return is optimised under the "
                        "quantisation that actually ships. Distinct from "
                        "qat_finetune.py, which minimised MSE to an FP32 "
                        "teacher -- an objective measured to be ANTI-correlated "
                        "with deployed success.")
    p.add_argument("--actor-wclip", type=float, default=0.0,
                   help="Clamp each actor Linear weight tensor to |w| <= k*std(w) "
                        "after every actor update, capping max/std -- the per-tensor "
                        "INT8 quantisation cost (Results/int8_deployment.txt Finding "
                        "4). 0 = off. Transport quantises free at 5.7; bi_s0 is 10.0.")
    p.add_argument("--best-window", type=int, default=200,
                   help="Episodes in the rolling window used to select best/. "
                        "Was 50, whose SE (~6.5 points) made checkpoint "
                        "selection pick lucky windows rather than good "
                        "policies. 200 gives SE ~3.2.")
    p.add_argument("--best-margin", type=float, default=0.01,
                   help="Metric improvement required to overwrite best/. "
                        "Larger margins resist noise-driven overwrites.")
    p.add_argument("--probe-every", type=int, default=25,
                   help="episodes between agent-internal probes "
                        "(critic Q stats, action saturation, SAC alpha)")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
