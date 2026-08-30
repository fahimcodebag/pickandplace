#!/usr/bin/env python3
"""Where in spawn space does the deployed grasp policy fail?

Runs the best/ checkpoint deterministically (no exploration noise) with lift
certification on, and records spawn pose plus outcome for every episode. The
training CSV cannot answer this: it is written under exploration noise, and its
tail reflects the decayed policy rather than the preserved best/ artifact.
"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
_H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _H)
sys.path.insert(0, os.path.join(_H, "Decomposed state training", "Random spawn model"))
import numpy as np, torch, pandas as pd
from networks import ActorNetwork
from grasp_spawn_wrapper import make_spawn_grasp_env

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
p.add_argument("--episodes", type=int, default=400)
p.add_argument("--level", type=float, default=2.0)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--out", default="Results/failure_map.csv")
a = p.parse_args()

net = ActorNetwork(46, 64, 32, 7, chkpt_dir="/tmp")
sd = torch.load(os.path.join(a.ckpt, "actor_td3"), map_location="cpu")
net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("log_std")})
net.eval()

env = make_spawn_grasp_env("PickPlace", seed=a.seed, curriculum=False,
                           level=a.level, require_lift=True)
def _yaw_from_quat(q):
    """z-yaw from an (x,y,z,w) quaternion."""
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z)))


# Flat 46-D GymWrapper layout: OBJ_POS 0:3, OBJ_QUAT 3:7 (see
# perception_wrapper.py). The spawn_* fields in episodes.csv come from
# GraspDiagnosticsWrapper, which train_rand.py adds and this eval env does not,
# so read the pose off the reset observation instead.
rows = []
for ep in range(a.episodes):
    obs = env.reset(); done = False; steps = 0; info = {}
    obj_pos = np.asarray(obs[0:3], dtype=float)
    obj_yaw = _yaw_from_quat(np.asarray(obs[3:7], dtype=float))
    while not done and steps < 200:
        with torch.no_grad():
            act = net(torch.tensor(obs, dtype=torch.float,
                                   device=net.device).unsqueeze(0))
        obs, _, done, info = env.step(act.squeeze(0).cpu().numpy())
        steps += 1
    # lift_rise is present ONLY when the 8-step hold was reached and the lift
    # probe ran. Its absence therefore means "never established a grip at all",
    # which is a different failure from "gripped, then dropped it on the lift".
    reached_hold = "lift_rise" in info
    rows.append({"episode": ep,
                 "success": int(bool(info.get("grasp_success", False))),
                 "reached_hold": int(reached_hold),
                 "lift_rise": float(info.get("lift_rise", np.nan)),
                 "spawn_x": obj_pos[0],
                 "spawn_y": obj_pos[1],
                 "spawn_yaw": obj_yaw,
                 "steps": steps})
    if (ep + 1) % 50 == 0:
        print(f"  {ep+1}/{a.episodes}  running success "
              f"{100*np.mean([r['success'] for r in rows]):.1f}%", flush=True)

df = pd.DataFrame(rows)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
df.to_csv(a.out, index=False)
print(f"\nwrote {a.out}  |  overall {100*df.success.mean():.1f}%")
