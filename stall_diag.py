#!/usr/bin/env python3
"""Why does TRANSPORT stall?

A stall is a horizon-out in TRANSPORT: the object never satisfied
  hypot(obj_xy - bin_xy) <= NEAR_TARGET_XY  for RELEASE_TRIG_HOLD steps
within PLACE_HORIZON. That single outcome hides several distinct causes, and
the rule-layer sweep cannot tell them apart. This records the carry trajectory
so each stalled episode can be assigned to one:

  PARKED       policy commands ~nothing; object stops well short of the bin
  BLOCKED      policy commands motion but the object does not move (contact,
               joint limit, OSC saturation)
  NEAR_MISS    object came inside NEAR_TARGET_XY but never for TRIG_HOLD
               consecutive steps -- a trigger-geometry problem, not a policy one
  SHORT        object approaches steadily but runs out of horizon (genuinely slow)
  WANDER       object moves a lot but not toward the bin

Constants and phase logic are imported from fsm_sim so this cannot drift from
the deployed sketch.
"""
import argparse, csv, os, sys
import numpy as np, torch as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fsm_sim as F
from fsm_sim import (FSM, load_actor, OBJ_X, OBJ_Y, OBJ_Z, BIN_X, BIN_Y,
                     GRASP, TRANSPORT, OK, FAIL, NAMES,
                     GRASP_CAP, TL_STEPS, MAX_GRASP_ATTEMPTS)
import robosuite as suite
from robosuite.wrappers import GymWrapper


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grasp-ckpt"); p.add_argument("--place-ckpt")
    p.add_argument("--grasp-tflite"); p.add_argument("--place-tflite")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fixed-spawn", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--tag", default="")
    a = p.parse_args()

    np.random.seed(a.seed); T.manual_seed(a.seed)
    raw = suite.make("PickPlace", robots="Panda",
                     controller_configs=suite.load_controller_config(
                         default_controller="OSC_POSE"),
                     has_renderer=False, has_offscreen_renderer=False,
                     use_camera_obs=False, horizon=700, reward_shaping=True,
                     control_freq=20, single_object_mode=2, object_type="bread")
    if a.fixed_spawn:
        _orig = raw._get_placement_initializer

        def _fixed():
            _orig()
            s = raw.placement_initializer.samplers["CollisionObjectSampler"]
            s.x_range = np.array([0.0, 0.0]); s.y_range = np.array([0.0, 0.0])
            s.rotation = 0.0
            s.ensure_object_boundary_in_range = False
            s.ensure_valid_placement = False
        raw._get_placement_initializer = _fixed
    env = GymWrapper(raw)
    env.seed(a.seed)
    lo, hi = env.action_space.low, env.action_space.high

    if a.int8:
        import tensorflow as tf

        class TFLiteActor:
            def __init__(self, path):
                self.it = tf.lite.Interpreter(model_path=path)
                self.it.allocate_tensors()
                self.inp = self.it.get_input_details()[0]
                self.out = self.it.get_output_details()[0]

            def __call__(self, x):
                v = x.detach().numpy() if hasattr(x, "detach") else np.asarray(x)
                v = v.reshape(1, -1).astype(np.float32)
                sc, zp = self.inp["quantization"]
                if self.inp["dtype"] == np.int8:
                    v = np.clip(np.round(v / sc + zp), -128, 127).astype(np.int8)
                self.it.set_tensor(self.inp["index"], v)
                self.it.invoke()
                o = self.it.get_tensor(self.out["index"])
                sc, zp = self.out["quantization"]
                if self.out["dtype"] == np.int8:
                    o = (o.astype(np.float32) - zp) * sc
                return T.tensor(o.astype(np.float32))
        g_actor = TFLiteActor(a.grasp_tflite) if a.grasp_tflite else load_actor(a.grasp_ckpt)
        p_actor = TFLiteActor(a.place_tflite) if a.place_tflite else load_actor(a.place_ckpt)
    else:
        g_actor, p_actor = load_actor(a.grasp_ckpt), load_actor(a.place_ckpt)

    def flags():
        gr = pl = False
        try:
            gr = bool(raw._check_grasp(gripper=raw.robots[0].gripper,
                                       object_geoms=raw.objects[raw.object_id]))
        except Exception:
            pass
        try:
            pl = bool(raw._check_success())
        except Exception:
            pass
        return gr, pl

    def dist(s):
        return float(np.hypot(s[OBJ_X] - BIN_X, s[OBJ_Y] - BIN_Y))

    rows = []
    for ep in range(a.episodes):
        obs, attempts, reached = None, 0, False
        for attempt in range(1, MAX_GRASP_ATTEMPTS + 1):
            obs = env.reset(); fsm = FSM(); attempts = attempt
            for _ in range(GRASP_CAP + TL_STEPS + 5):
                gr, pl = flags()
                act = fsm.step(np.asarray(obs, dtype=np.float32), gr, pl, g_actor, p_actor)
                obs, _, done, _ = env.step(np.clip(act, lo, hi))
                if fsm.phase == TRANSPORT:
                    reached = True; break
                if fsm.phase in (OK, FAIL) or done:
                    break
            if reached:
                break
        if not reached:
            rows.append(dict(episode=ep, attempts=attempts, success=0,
                             phase="handoff_failed", tag=a.tag))
            continue

        # --- scored portion, instrumenting TRANSPORT ------------------------
        D, P, A, G = [], [], [], []     # dist, obj xyz, translation cmd, grasped
        n = 0
        while n < 700:
            gr, pl = flags()
            s = np.asarray(obs, dtype=np.float32)
            in_tr = (fsm.phase == TRANSPORT)
            act = fsm.step(s, gr, pl, g_actor, p_actor)
            if in_tr:
                D.append(dist(s)); P.append(s[[OBJ_X, OBJ_Y, OBJ_Z]].copy())
                A.append(float(np.linalg.norm(act[0:3]))); G.append(int(gr))
            obs, _, done, _ = env.step(np.clip(act, lo, hi))
            n += 1
            if fsm.phase in (OK, FAIL) or done:
                break

        r = dict(episode=ep, attempts=attempts, success=int(fsm.phase == OK),
                 phase=NAMES[fsm.phase], steps=n, tr_steps=len(D), tag=a.tag)
        if D:
            D = np.asarray(D); P = np.asarray(P); A = np.asarray(A); G = np.asarray(G)
            near = D <= F.NEAR_TARGET_XY
            # longest run of consecutive "near" steps
            best = cur = 0
            for v in near:
                cur = cur + 1 if v else 0
                best = max(best, cur)
            tail = slice(max(0, len(D) - 50), len(D))
            path = float(np.sum(np.linalg.norm(np.diff(P[tail], axis=0), axis=1))) if len(D) > 1 else 0.0
            r.update(
                d_start=float(D[0]), d_min=float(D.min()), d_end=float(D[-1]),
                d_progress=float(D[0] - D.min()),
                n_near=int(near.sum()), max_run_near=int(best),
                act_tail=float(A[tail].mean()), act_all=float(A.mean()),
                path_tail=path,
                net_tail=float(np.linalg.norm(P[-1] - P[tail.start])),
                z_end=float(P[-1, 2]), z_min=float(P[:, 2].min()),
                grasped_end=int(G[-1]), frac_grasped=float(G.mean()),
            )
        rows.append(r)
        if (ep + 1) % 25 == 0:
            s_ = sum(x["success"] for x in rows)
            print(f"  {ep+1}/{a.episodes}  success {s_/len(rows)*100:.1f}%", flush=True)

    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    s_ = sum(r["success"] for r in rows)
    print(f"\n{a.tag} {s_}/{len(rows)} = {s_/len(rows)*100:.1f}%  -> {a.out}")


if __name__ == "__main__":
    main()
