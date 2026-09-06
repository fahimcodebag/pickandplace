#!/usr/bin/env python3
"""Distil the grasp actor into a smaller network.

Pruning alone does not survive: removing units and stopping there leaves mean
|action error| of 0.51-0.78 on a [-1,1] action range (Results/pruning). The
student has to be RETRAINED, and the natural target is the teacher's own
output on states the teacher actually visits -- which is distillation, not
fine-tuning on the RL objective.

States come from teacher rollouts through the real FSM, so the distribution is
the one the student will be deployed on. Training on random states would be
easy and useless: the state space is 46-D and the visited manifold is a
vanishing fraction of it.

Two initialisations are compared, because the difference is the actual
question -- does structured pruning give a better starting point than random
init, or is it just a smaller network trained twice?
"""
import argparse, os, sys
import numpy as np, torch as T, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from networks import ActorNetwork
from prune_actor import load, prune


def collect(teacher, n_states, seed):
    """States the teacher visits, via the deployed FSM."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    import fsm_sim as F
    raw = suite.make("PickPlace", robots="Panda",
                     controller_configs=suite.load_controller_config(
                         default_controller="OSC_POSE"),
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, horizon=700, reward_shaping=True,
                     control_freq=20, single_object_mode=2, object_type="bread")
    env = GymWrapper(raw)
    lo, hi = env.action_space.low, env.action_space.high
    np.random.seed(seed); T.manual_seed(seed)
    S = []
    while len(S) < n_states:
        obs = env.reset()
        for _ in range(220):
            s = np.asarray(obs, dtype=np.float32)
            S.append(s.copy())
            with T.no_grad():
                act = teacher(T.tensor(s).unsqueeze(0)).squeeze(0).numpy()
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            if done or len(S) >= n_states:
                break
    return np.array(S[:n_states], dtype=np.float32)


def fit(student, X, Y, epochs, lr, bs=256, seed=0):
    T.manual_seed(seed)
    opt = T.optim.Adam(student.parameters(), lr=lr)
    sch = T.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt, yt = T.tensor(X), T.tensor(Y)
    n = len(xt)
    for _ in range(epochs):
        idx = T.randperm(n)
        for k in range(0, n, bs):
            b = idx[k:k + bs]
            opt.zero_grad()
            ((student(xt[b]) - yt[b]) ** 2).mean().backward()
            opt.step()
        sch.step()
    student.eval()
    with T.no_grad():
        return float((student(xt) - yt).abs().mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--keep1", type=int, required=True)
    p.add_argument("--keep2", type=int, required=True)
    p.add_argument("--init", choices=["pruned", "random"], default="pruned")
    p.add_argument("--states", type=int, default=60000)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    teacher = load(a.ckpt)
    cache = f"Results/distill/states_{a.states}_{a.seed}.npy"
    os.makedirs("Results/distill", exist_ok=True)
    if os.path.exists(cache):
        X = np.load(cache)
    else:
        X = collect(teacher, a.states, a.seed)
        np.save(cache, X)
    with T.no_grad():
        Y = teacher(T.tensor(X)).numpy()

    if a.init == "pruned":
        student = prune(teacher, a.keep1, a.keep2)
    else:
        student = ActorNetwork(46, a.keep1, a.keep2, 7, chkpt_dir="/tmp").cpu()
    student.train()
    err = fit(student, X, Y, a.epochs, a.lr, seed=a.seed)
    n = sum(x.numel() for x in student.parameters())
    os.makedirs(a.out, exist_ok=True)
    T.save(student.state_dict(), os.path.join(a.out, "actor_td3"))
    print(f"{a.keep1}x{a.keep2} init={a.init}: {n:,} params ({n/1024:.1f} KB INT8), "
          f"mean |action delta| vs teacher {err:.4f}  -> {a.out}")


if __name__ == "__main__":
    main()
