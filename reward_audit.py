#!/usr/bin/env python3
"""Reward audit for the grasp stage.

Rolls out a trained policy and accumulates EVERY reward term separately, then
reports mean accumulated value per outcome class. The question it answers is
not "what does the reward say it wants" but "what does the reward actually
PAY", which is what the agent optimises.

Terms come from info["reward_terms"], accumulated inside _grasp_reward itself,
so this cannot drift from the code under test.
"""
import argparse, collections, os, sys
import numpy as np, torch as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "Decomposed state training"))
sys.path.insert(0, os.path.join(HERE, "Decomposed state training", "Random spawn model"))
from grasp_spawn_wrapper import make_spawn_grasp_env
from networks import ActorNetwork

TERMS = ["reach", "grip_close", "grasp_hold", "success_bonus",
         "partial_credit", "align_shape", "idle", "drop", "away"]


def classify(info, steps):
    if info.get("grasp_success"):
        return "CERTIFIED"
    if info.get("lift_rise") is not None:
        return "slip (gripped, lift failed)"
    if info["reward_terms"]["n_drops"] >= 3:
        return "flicker (>=3 drops, timeout)"
    if info["reward_terms"]["n_grasped_steps"] > 0:
        return "touched (timeout)"
    return "never gripped (timeout)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--level", type=float, default=2.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--reward-v2", action="store_true")
    p.add_argument("--csv", default=None,
                   help="Dump per-episode terms + grip angle + lift rise, so "
                        "candidate rewards can be priced offline.")
    p.add_argument("--explore", type=float, default=0.0,
                   help="Gaussian action noise. 0 = deterministic (what the "
                        "policy converged to); >0 approximates what the agent "
                        "actually SAW during training, which is what shaped it.")
    a = p.parse_args()

    env = make_spawn_grasp_env("PickPlace", seed=a.seed, curriculum=False,
                               level=a.level, require_lift=True,
                               reward_v2=a.reward_v2)
    actor = ActorNetwork(env.observation_space.shape[0], 64, 32,
                         env.action_space.shape[0], chkpt_dir=a.ckpt)
    sd = T.load(os.path.join(a.ckpt, "actor_td3"), map_location="cpu")
    actor.load_state_dict({k: v for k, v in sd.items()
                           if not k.startswith("log_std")})
    actor.to(T.device("cpu")); actor.device = T.device("cpu"); actor.eval()

    buckets = collections.defaultdict(list)
    for i in range(a.episodes):
        obs = env.reset(); done = False; n = 0
        disc = 0.0
        while not done:
            with T.no_grad():
                act = actor(T.tensor(obs, dtype=T.float).unsqueeze(0)).squeeze(0).numpy()
            if a.explore > 0:
                act = np.clip(act + np.random.normal(0, a.explore, act.shape), -1, 1)
            obs, r, done, info = env.step(act)
            disc += (a.gamma ** n) * r
            n += 1
        rt = info["reward_terms"]
        rt["_align"] = info.get("grip_align")
        rt["_rise"] = info.get("lift_rise")
        rt["_certified"] = int(bool(info.get("grasp_success")))
        rt["_steps"] = n
        rt["_total"] = sum(rt[k] for k in TERMS)
        rt["_disc"] = disc
        buckets[classify(info, n)].append(rt)

    if a.csv:
        import csv as _csv
        allrows = [dict(e, outcome=k) for k, v in buckets.items() for e in v]
        with open(a.csv, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
            w.writeheader(); w.writerows(allrows)
        print(f"wrote {a.csv}")

    total_n = sum(len(v) for v in buckets.values())
    print(f"\n{'='*100}")
    print(f"REWARD AUDIT  |  {a.ckpt}")
    print(f"{a.episodes} episodes, spawn level {a.level}, seed {a.seed}, "
          f"action noise {a.explore}")
    print(f"{'='*100}")
    hdr = f"{'outcome':<30}{'n':>5}{'%':>6}" + "".join(f"{t[:9]:>11}" for t in TERMS)
    print(hdr); print("-" * len(hdr))
    order = sorted(buckets, key=lambda k: -len(buckets[k]))
    for k in order:
        v = buckets[k]
        row = f"{k:<30}{len(v):5d}{len(v)/total_n*100:5.1f}%"
        row += "".join(f"{np.mean([e[t] for e in v]):11.1f}" for t in TERMS)
        print(row)
    print("-" * len(hdr))
    print(f"\n{'outcome':<30}{'steps':>8}{'drops':>8}{'held':>8}"
          f"{'TOTAL':>12}{'DISCOUNTED':>13}")
    print("-" * 79)
    for k in order:
        v = buckets[k]
        print(f"{k:<30}{np.mean([e['_steps'] for e in v]):8.0f}"
              f"{np.mean([e['n_drops'] for e in v]):8.1f}"
              f"{np.mean([e['n_grasped_steps'] for e in v]):8.1f}"
              f"{np.mean([e['_total'] for e in v]):12.1f}"
              f"{np.mean([e['_disc'] for e in v]):13.1f}")


if __name__ == "__main__":
    main()
