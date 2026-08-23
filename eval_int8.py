#!/usr/bin/env python3
"""Deterministic evaluation of a quantized .tflite grasp actor.

Mirrors eval_best.py exactly -- same env constructor, same spawn level, same
seed, same 200-step cap -- so the INT8 number is directly comparable to the
FP32 one. The only difference is what produces the action.

Run with the converter venv (it has tensorflow AND inherits robosuite):
  /home/fahim/Thesis_fahim/convert_venv/bin/python eval_int8.py --model ...
"""
import argparse, os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "Decomposed state training", "Random spawn model"))
import numpy as np
from grasp_spawn_wrapper import make_spawn_grasp_env

p = argparse.ArgumentParser()
p.add_argument("--model", required=True, help="path to .tflite")
p.add_argument("--episodes", type=int, default=100)
p.add_argument("--level", type=float, default=1.0)
p.add_argument("--seed", type=int, default=123)
a = p.parse_args()

import tensorflow as tf
interp = tf.lite.Interpreter(model_path=a.model)
interp.allocate_tensors()
in_det, out_det = interp.get_input_details()[0], interp.get_output_details()[0]
in_dt, out_dt = np.dtype(in_det["dtype"]), np.dtype(out_det["dtype"])
in_sc, in_zp = in_det.get("quantization", (0.0, 0))
out_sc, out_zp = out_det.get("quantization", (0.0, 0))
print(f"{a.model}\n  input {in_dt} scale={in_sc:.6g} zp={in_zp} | "
      f"output {out_dt} scale={out_sc:.6g} zp={out_zp}")


def act(obs):
    x = np.asarray(obs, dtype=np.float32)[None, :]
    if in_dt == np.int8:
        x = np.clip(np.round(x / in_sc + in_zp), -128, 127).astype(np.int8)
    interp.set_tensor(in_det["index"], x)
    interp.invoke()
    y = interp.get_tensor(out_det["index"])
    if out_dt == np.int8:
        y = (y.astype(np.float32) - out_zp) * out_sc
    return y[0]


env = make_spawn_grasp_env("PickPlace", seed=a.seed, curriculum=False, level=a.level)
succ, steps_all = 0, []
for ep in range(a.episodes):
    obs = env.reset()
    done, steps, info = False, 0, {}
    while not done and steps < 200:
        obs, _, done, info = env.step(act(obs))
        steps += 1
    ok = bool(info.get("grasp_success", False))
    succ += ok
    steps_all.append((steps, ok))
print(f"  spawn level {a.level} | {a.episodes} episodes, deterministic")
print(f"  SUCCESS: {succ}/{a.episodes} = {100.0 * succ / a.episodes:.1f}%")
_succ = [s for s, ok in steps_all if ok]
_all = [s for s, _ in steps_all]
print(f"  mean steps (successes only): {np.mean(_succ):.0f}" if _succ
      else "  mean steps (successes only): n/a")
print(f"  mean steps (all episodes):   {np.mean(_all):.0f}")
