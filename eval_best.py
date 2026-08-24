#!/usr/bin/env python3
"""Deterministic evaluation of a trained random-spawn grasp actor.

Training-time success is measured under exploration noise and therefore
UNDERSTATES the deployed policy (§3). This runs the noise-free policy, which
is what actually ships to the MCU.

Handles TD3 (networks.ActorNetwork), SAC (GaussianActor -> tanh(mean)) and PPO
(PPOActor -> tanh(mean)); all three share the deployed fc1/fc2/output shape.
"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "Decomposed state training", "Random spawn model"))
import numpy as np, torch
from networks import ActorNetwork
from grasp_spawn_wrapper import make_spawn_grasp_env

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--episodes", type=int, default=30)
p.add_argument("--level", type=float, default=1.0)
p.add_argument("--seed", type=int, default=123)
p.add_argument("--require-lift", action="store_true",
               help="Certify grasps with the scripted lift the place "
                    "stage applies at handoff. Without it the score "
                    "measures momentary contact only.")
a = p.parse_args()

# Every algorithm's deployed actor is 46->64->32->7 with tanh; the trunk/mean
# keys match, so one loader covers all three.
net = ActorNetwork(46, 64, 32, 7, chkpt_dir="/tmp")
sd = torch.load(os.path.join(a.ckpt, "actor_td3"), map_location="cpu")
sd = {k: v for k, v in sd.items() if not k.startswith("log_std")}
net.load_state_dict(sd); net.eval()

env = make_spawn_grasp_env("PickPlace", seed=a.seed, curriculum=False,
                           level=a.level, require_lift=a.require_lift)
succ, steps_all, reasons = 0, [], {}
for ep in range(a.episodes):
    obs = env.reset(); done = False; steps = 0; info = {}
    while not done and steps < 200:
        with torch.no_grad():
            act = net(torch.tensor(obs, dtype=torch.float, device=net.device).unsqueeze(0))
        obs, _, done, info = env.step(act.squeeze(0).cpu().numpy()); steps += 1
    ok = bool(info.get("grasp_success", False)); succ += ok
    steps_all.append((steps, ok))
    r = "success" if ok else "failure"
    reasons[r] = reasons.get(r, 0) + 1
print(f"\n{a.ckpt}")
print(f"  spawn level {a.level} (1.0 = full position box)  |  {a.episodes} episodes, deterministic")
print(f"  SUCCESS: {succ}/{a.episodes} = {100*succ/a.episodes:.1f}%")
# These were one figure before, labelled "mean steps (successes)" but computed
# over ALL episodes: the success mask was built from an empty list, so the
# comprehension was empty and the `or steps_all` fallback averaged everything,
# 200-step timeouts included. Report both, explicitly.
_succ_steps = [s for s, ok in steps_all if ok]
_all_steps = [s for s, _ in steps_all]
print(f"  mean steps (successes only): "
      f"{np.mean(_succ_steps):.0f}" if _succ_steps else "  mean steps (successes only): n/a")
print(f"  mean steps (all episodes):   {np.mean(_all_steps):.0f}")
