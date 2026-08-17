#!/usr/bin/env python3
# Last updated: 2026-07-21
"""
Dump PLACE-phase calibration states for INT8 quantization.

The place (transport) model only ever sees post-grasp, lifted, over-the-table
observations — a different distribution from the grasp demos. Quantizing it on
grasp-phase states (demos_bread.npz) gives it wrong INT8 scales. This script
runs the real deployment pipeline (grasp model -> place model, curriculum off,
locked FSM params) and records exactly the observations the PLACE model is
asked to act on, so qat_and_convert.py can calibrate on-distribution.

Only the outer-loop observations fed to the place agent are logged. The
scripted release phases (recenter/descend/open/retract) run inside env.step via
P-controllers, not the model, so they are correctly excluded.

Usage (from this folder, in the venv):
    python dump_place_states.py --episodes 60
    # -> writes ../place_states.npz  (key: "states", shape [N, 46])

Then, from the project root:
    python qat_and_convert.py --actor_path checkpoints/td3_place/best/actor_td3 \
        --input_dims 46 --n_actions 7 --fc1 64 --fc2 32 \
        --replay_buffer_path place_states.npz --output_path place_int8.tflite
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, ".."))

from td3 import Agent
from test_place import make_place_eval_env


def main():
    ap = argparse.ArgumentParser(description="Dump place-phase calibration states")
    ap.add_argument("--episodes", type=int, default=60,
                    help="Episodes to roll out (more = better calibration)")
    ap.add_argument("--out", type=str,
                    default=os.path.join(_HERE, "..", "place_states.npz"),
                    help="Output .npz path (default: project-root/place_states.npz)")
    ap.add_argument("--trigger-xy", type=float, default=0.14)
    ap.add_argument("--trigger-hold", type=int, default=3)
    ap.add_argument("--place-horizon", type=int, default=300)
    ap.add_argument("--random-spawn", action="store_true",
                    help="Native full spawn randomization (match if deploying "
                         "the random-spawn system; default is fixed spawn)")
    ap.add_argument("--spawn-range", type=float, default=None)
    args = ap.parse_args()

    grasp_dir = os.path.join(_HERE, "..", "checkpoints", "td3_grasp")
    place_dir = os.path.join(_HERE, "..", "checkpoints", "td3_place", "best")

    env, _ = make_place_eval_env(render=False, grasp_chkpt_dir=grasp_dir,
                                 spawn_range=args.spawn_range,
                                 native_spawn=args.random_spawn)
    env._NEAR_TARGET_XY = args.trigger_xy
    env._RELEASE_TRIGGER_HOLD = args.trigger_hold
    env.PLACE_HORIZON = args.place_horizon

    agent = Agent(alpha=3e-4, beta=3e-4, tau=0.005,
                  input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0],
                  layer1_size=64, layer2_size=32, batch_size=512,
                  chkpt_dir=place_dir)
    agent.load_models()
    print(f"Place model loaded from {place_dir}\n")

    states = []
    placed = 0
    for i in range(args.episodes):
        obs = env.reset()
        states.append(np.asarray(obs, dtype=np.float32))  # handoff observation
        done = False
        n = 0
        info = {}
        while not done:
            action = agent.choose_action(obs, validation=True)
            obs, _, done, info = env.step(action)
            if not done:
                states.append(np.asarray(obs, dtype=np.float32))
            n += 1
        ok = info.get("place_success", False)
        placed += int(ok)
        print(f"  ep {i+1:3d}/{args.episodes} | steps {n:3d} | "
              f"{'PLACED' if ok else info.get('place_done_reason','?'):16s} | "
              f"states so far: {len(states)}")

    arr = np.asarray(states, dtype=np.float32)
    out = os.path.abspath(args.out)
    np.savez(out, states=arr)
    env.close()

    print(f"\n{'='*60}")
    print(f"  Saved {arr.shape[0]} place-phase states -> {out}")
    print(f"  (key 'states', shape {arr.shape})")
    print(f"  Episodes placed: {placed}/{args.episodes}")
    print(f"{'='*60}")
    print("\nNow calibrate the place model with:")
    print("  python qat_and_convert.py --actor_path "
          "checkpoints/td3_place/best/actor_td3 \\")
    print("      --input_dims 46 --n_actions 7 --fc1 64 --fc2 32 \\")
    print(f"      --replay_buffer_path {os.path.basename(out)} "
          "--output_path place_int8.tflite")


if __name__ == "__main__":
    main()
