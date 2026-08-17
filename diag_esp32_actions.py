#!/usr/bin/env python3
"""
Diagnostic: does the ESP32's on-device inference match the same INT8 model
run on the PC, for the SAME observation?

Drives the sim through the ESP32 exactly like hil_main's handoff does, but for
every step also runs grasp_int8.tflite locally on the identical observation and
compares the two 7-dim actions. Localises a mismatch to either the board's
inference/serial path or to the model itself.

Run:  python diag_esp32_actions.py
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

import hil_main
from esp32_bridge import ESP32Bridge
from protocol_float32 import ProtocolFloat32 as Protocol

STEPS = 40
PORT = '/dev/ttyUSB0'


def local_int8(path):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]

    def run(s):
        x = np.asarray(s, dtype=np.float32).reshape(1, -1)
        if ind['dtype'] == np.int8:
            sc, zp = ind['quantization']
            x = np.clip(np.round(x / sc) + zp, -128, 127).astype(np.int8)
        it.set_tensor(ind['index'], x.astype(ind['dtype']))
        it.invoke()
        o = it.get_tensor(outd['index'])[0]
        if outd['dtype'] == np.int8:
            sc, zp = outd['quantization']
            o = (o.astype(np.float32) - zp) * sc
        return np.clip(o, -1.0, 1.0)
    return run


def main():
    grasp_local = local_int8("grasp_int8.tflite")

    raw_env, env = hil_main.make_env(render=False)
    bridge = ESP32Bridge(port=PORT, baudrate=921600, timeout=2.0)
    if not bridge.connect():
        return

    obs = env.reset()
    bridge.send_reset()
    lo, hi = env.action_space.low, env.action_space.high

    print(f"\n{'step':>4} {'phase':<10} {'flags':>5} "
          f"{'maxdiff':>8} {'esp_grip':>9} {'loc_grip':>9}")
    print("-" * 52)

    worst = 0.0
    for t in range(STEPS):
        flags = hil_main.compute_flags(raw_env)
        esp_a, phase = bridge.get_action(obs, flags)
        loc_a = grasp_local(obs)

        diff = float(np.max(np.abs(np.asarray(esp_a) - loc_a)))
        worst = max(worst, diff)
        mark = "  <-- MISMATCH" if diff > 0.05 else ""
        print(f"{t:>4} {Protocol.PHASE_NAMES[phase]:<10} {flags:>5} "
              f"{diff:>8.4f} {esp_a[6]:>9.3f} {loc_a[6]:>9.3f}{mark}")

        obs, _, done, _ = env.step(np.clip(esp_a, lo, hi))
        if done:
            break

    bridge.disconnect()
    print("-" * 52)
    print(f"worst |esp32 - local| over {STEPS} steps: {worst:.4f}")
    if worst < 0.05:
        print("=> ESP32 inference MATCHES the local INT8 model.")
        print("   The models/serial path are fine; the fault is elsewhere.")
    else:
        print("=> ESP32 inference DIVERGES from the same model on the PC.")
        print("   Suspect the on-device interpreter (arena size, op support,")
        print("   or the two-interpreter setup), not the exported model.")


if __name__ == "__main__":
    main()
