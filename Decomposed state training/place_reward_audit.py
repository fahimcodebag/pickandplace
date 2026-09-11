#!/usr/bin/env python3
"""Price place-stage rewards on recorded episodes BEFORE training (thesis_context 9.10).

One process = one (policy, curriculum frac, seed) cell:

    python place_reward_audit.py --policy actor:../checkpoints/td3_place_X/best \
        --label good --frac 0.3 --episodes 20 --seed 0 --out ../Results/place_reward_audit/x.csv

Rolls out PlaceGymWrapper exactly as training does -- same env factory, grasp
policy, scripted carry and release, termination -- with the curriculum pinned at
--frac, and records per episode the accumulated reward under every mode from ONE
trajectory:
    R_custom    sum of the custom reward (what reward_mode=custom returns)
    R_builtin   sum of robosuite's shaped reward at each transition's end state
    steps       policy steps, so R_builtin_idle(c) = R_builtin - c * steps
and the outcome class (place_done_reason).

POSITIVE CONTROLS, per episode (columns ctrl_*):
  * the rewards the wrapper returns (custom mode) sum to exactly R_custom
  * on every non-release step, the replica r_builtin equals robosuite's own
    reward(): the env is built with reward_shaping=True, so GymWrapper's reward is
    the reference

POLICIES
  actor:DIR  a trained place actor, deterministic
  zero       zero translation: the scaffolding floor
  linger     scripted: hold the object ~0.14 m from the bin centre, outside the
             0.10 m release trigger.  Prices the hover exploit; ends at the
             transport-stall limit.
  creep      scripted: approach the bin 1.05 cm every 45 steps, from the handoff
             distance down to 0.11 m.  Each step counts as progress, so it
             dodges the stall limit and hovers as long as termination allows.
"""
import argparse, csv, os, sys
import numpy as np
import torch as T

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from train_place import make_place_env      # noqa: E402
from networks import ActorNetwork           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--frac", type=float, required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--object-type", default="cereal")
    ap.add_argument("--grasp-ckpt", default=os.path.join(ROOT, "checkpoints", "td3_grasp_rand_td3_ln_cereal_alignwarm_s0", "best"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    np.random.seed(a.seed); T.manual_seed(a.seed)
    env = make_place_env("PickPlace", seed=a.seed * 1000 + 7, grasp_chkpt_dir=a.grasp_ckpt,
                         random_spawn=True, object_type=a.object_type,
                         reward_mode="custom", reward_shaping=True)
    env._curriculum_frac = a.frac
    env._update_curriculum = lambda: None          # pin the curriculum

    actor = None
    if a.policy.startswith("actor:"):
        d = a.policy[len("actor:"):]
        actor = ActorNetwork(46, 64, 32, 7, chkpt_dir=d)
        sd = T.load(os.path.join(d, "actor_td3"), map_location="cpu")
        actor.load_state_dict({k: v for k, v in sd.items() if not k.startswith("log_std")})
        actor.to(T.device("cpu")); actor.device = T.device("cpu"); actor.eval()
    elif a.policy not in ("zero", "linger", "creep"):
        sys.exit(f"unknown policy {a.policy!r}")

    def obj_bin():
        od = env._raw_env._get_observations()
        return env._get_obj_pos(od), env._get_target_bin_pos()

    def scripted(t, z0, d0):
        act = np.zeros(7, dtype=np.float32)
        p, b = obj_bin()
        if p is None or b is None:
            return act
        v = p[:2] - b[:2]; d = float(np.linalg.norm(v))
        u = v / d if d > 1e-6 else np.array([1.0, 0.0])
        r = 0.14 if a.policy == "linger" else max(0.11, d0 - 0.0105 * (t // 45))
        act[0:2] = np.clip(8.0 * (b[:2] + u * r - p[:2]), -1.0, 1.0)
        act[2] = np.clip(5.0 * (z0 - p[2]), -1.0, 1.0)
        return act

    rows = []
    for ep in range(a.episodes):
        obs = env.reset()
        p, b = obj_bin()
        z0 = float(p[2]) if p is not None else 1.0
        d0 = float(np.linalg.norm(p[:2] - b[:2])) if p is not None and b is not None else 0.3
        R_ret = R_c = R_b = 0.0
        steps = released = mism = 0
        maxabs = 0.0
        done, info = False, {}
        while not done:
            if actor is not None:
                with T.no_grad():
                    act = actor(T.tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)).squeeze(0).numpy()
            elif a.policy == "zero":
                act = np.zeros(7, dtype=np.float32)
            else:
                act = scripted(steps, z0, d0)
            obs, r, done, info = env.step(act)
            steps += 1
            R_ret += r; R_c += info["r_custom"]; R_b += info["r_builtin"]
            if info["released"]:
                released += 1
            else:
                dd = abs(info["r_env_raw"] - info["r_builtin"])
                maxabs = max(maxabs, dd); mism += dd > 1e-9
        rows.append(dict(
            label=a.label or a.policy, policy=a.policy, frac=a.frac, seed=a.seed, episode=ep,
            reason=info.get("place_done_reason", ""), success=int(bool(info.get("place_success", False))),
            steps=steps, released=released, handoff_d=round(d0, 4),
            R_custom=round(R_c, 4), R_builtin=round(R_b, 4),
            ctrl_returned_minus_custom=round(R_ret - R_c, 6),
            ctrl_builtin_mismatch_steps=mism, ctrl_builtin_max_abs=maxabs))
        print(f"  ep {ep}: {rows[-1]['reason']:22s} steps {steps:3d}  R_custom {R_c:8.2f}  R_builtin {R_b:7.2f}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    env.close()


if __name__ == "__main__":
    main()
