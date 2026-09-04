#!/usr/bin/env python3
"""
Prints the EXPECTED outputs of the boot-time self-test in
pick_and_place_INT8_FSM.ino, computed with desktop TFLite on the identical
deterministic input vector s[i] = (i-23)*0.05.

Compare against the ESP32's serial output at boot. Any large disagreement
isolates the fault to on-device inference (arena / op kernels / interpreter
setup) rather than the exported model, the serial protocol or the FSM.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf

STATE_DIM = 46


def run(path, x):
    it = tf.lite.Interpreter(model_path=path)
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    v = x.reshape(1, -1)
    if ind['dtype'] == np.int8:
        sc, zp = ind['quantization']
        v = np.clip(np.round(v / sc) + zp, -128, 127).astype(np.int8)
    it.set_tensor(ind['index'], v.astype(ind['dtype']))
    it.invoke()
    o = it.get_tensor(outd['index'])[0]
    if outd['dtype'] == np.int8:
        sc, zp = outd['quantization']
        o = (o.astype(np.float32) - zp) * sc
    return np.clip(o, -1.0, 1.0)


def main():
    x = ((np.arange(STATE_DIM) - 23) * 0.05).astype(np.float32)
    print("\n--- EXPECTED SELF-TEST OUTPUT (desktop TFLite) ---")
    print("input: s[i]=(i-23)*0.05  ->  "
          f"[{x[0]:+.2f} ... {x[-1]:+.2f}]\n")
    # These MUST be the tflite files the flashed headers were generated from,
    # or the comparison is meaningless. grasp_model.h / place_model.h record
    # their source in the first comment line -- read it rather than assuming.
    import re, sys
    def src_of(header, default):
        try:
            m = re.search(r"Source:\s*(\S+)", open(header).read()[:400])
            return m.group(1) if m else default
        except OSError:
            return default
    cands = {
        "c2m512_s1_int8.tflite": "Results/big2m/c2m512_s1_int8.tflite",
        "place_orig_int8.tflite": "qat_output_bi/place_orig_int8.tflite",
    }
    g = src_of("grasp_model.h", "grasp_int8.tflite")
    pl = src_of("place_model.h", "place_int8.tflite")
    g, pl = cands.get(g, g), cands.get(pl, pl)
    print(f"grasp header source: {g}\nplace header source: {pl}\n")
    for name, path in (("grasp", g), ("place", pl)):
        o = run(path, x)
        print(f"{name}: " + " ".join(f"{v:+.6f}" for v in o))
    print("\nCompare with the ESP32 boot log. Matching (within ~0.01) means")
    print("on-device inference is correct; large differences mean it is not.")


if __name__ == "__main__":
    main()
