#!/usr/bin/env python3
"""Measure whether the CRITIC values grip alignment.

The actor's policy gradient is dQ/da, so a value function blind to alignment
gives the policy no gradient toward rotating however well the REWARD is shaped.
Measured on rv2_s2: reward correlates +0.85 with alignment, Q correlates
-0.004. This script is the primary metric for the dense-alignment arm.
"""
import argparse, os, sys
import numpy as np, torch as T, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Decomposed state training"))
sys.path.insert(0, os.path.join(HERE, "Decomposed state training", "Random spawn model"))
from grasp_spawn_wrapper import make_spawn_grasp_env
from networks import ActorNetwork


class LNCritic(nn.Module):
    def __init__(self, i, h1, h2):
        super().__init__()
        self.fc1 = nn.Linear(i, h1); self.ln1 = nn.LayerNorm(h1)
        self.fc2 = nn.Linear(h1, h2); self.ln2 = nn.LayerNorm(h2)
        self.q1 = nn.Linear(h2, 1)

    def forward(self, x, a):
        h = T.relu(self.ln1(self.fc1(T.cat([x, a], 1))))
        return self.q1(T.relu(self.ln2(self.fc2(h))))


def yaw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="run dir (holds critic_1_td3 and best/)")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()

    env = make_spawn_grasp_env("PickPlace", seed=a.seed, curriculum=False,
                               level=2.0, require_lift=True)
    act = ActorNetwork(46, 64, 32, 7, chkpt_dir=os.path.join(a.ckpt, "best"))
    act.load_state_dict({k: v for k, v in T.load(
        os.path.join(a.ckpt, "best", "actor_td3"), map_location="cpu").items()
        if not k.startswith("log_std")})
    act.to(T.device("cpu")); act.device = T.device("cpu"); act.eval()
    cr = LNCritic(53, 64, 32)
    cr.load_state_dict(T.load(os.path.join(a.ckpt, "critic_1_td3"), map_location="cpu"))
    cr.eval()

    S = []
    for _ in range(a.episodes):
        o = env.reset(); d = False
        while not d:
            S.append(o.copy())
            with T.no_grad():
                u = act(T.tensor(o, dtype=T.float).unsqueeze(0)).squeeze(0).numpy()
            o, r, d, i = env.step(u)
    S = np.array(S, dtype=np.float32)
    th = yaw(S[:, 10:14])
    dg = np.degrees(np.abs((th + np.pi / 4) % (np.pi / 2) - np.pi / 4))
    with T.no_grad():
        Xs = T.tensor(S); Q = cr(Xs, act(Xs)).squeeze(1).numpy()

    print(f"\n{a.ckpt}   ({len(S)} states)")
    print(f"  corr(Q, alignment quality) = {np.corrcoef(Q, 1 - dg / 45)[0, 1]:+.3f}")
    for lo, hi, l in [(0, 15, "flat 0-15"), (15, 30, "mid 15-30"), (30, 46, "corner 30-45")]:
        m = (dg >= lo) & (dg < hi)
        if m.sum():
            print(f"    {l:<14} n={m.sum():5d}   mean Q = {Q[m].mean():8.2f}")


if __name__ == "__main__":
    main()
