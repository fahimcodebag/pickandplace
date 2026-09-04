#!/usr/bin/env python3
"""Verify fsm_sim.py mirrors pick_and_place_INT8_FSM.ino.

The FP32 validation in thesis_context Sec 9.11 is only meaningful while the
host-side replica and the shipped sketch agree. A tuned constant applied to one
and not the other silently reintroduces the harness-vs-artifact gap that
section was written to close. Run this after touching either file.
"""
import importlib.util, re, sys

NAMES = ["NEAR_TARGET_XY", "RELEASE_TRIG_HOLD", "PLACE_HORIZON", "GRASP_HOLD",
         "GRASP_CAP", "TRANSLATE_SCALE", "TL_STEPS", "TL_DZ", "TL_MIN_RISE",
         "CARRY_GAIN", "CARRY_CLIP", "RC_STEPS", "RC_TOL", "DS_STEPS", "DS_DZ",
         "TOUCH_MARGIN", "OP_STEPS", "RT_STEPS", "RT_DZ",
         "BIN_X", "BIN_Y", "BIN_Z",
         # Fix A, mirrored into the sketch once the host measurement adopted it
         "LOST_GRIP_STEPS", "MAX_REGRASP"]
# TABLE_Z is intentionally absent from fsm_sim: the sketch uses it to fail an
# episode early when the object drops below the table, while the replica lets
# the phase machine run on. Both score the episode a failure, so the outcome is
# identical and only the detection point differs.

spec = importlib.util.spec_from_file_location("fsm_sim", "fsm_sim.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except SystemExit:
    pass
ino = open("pick_and_place_INT8_FSM.ino").read()

bad = []
for n in NAMES:
    m = re.search(rf"\b{n}\b\s*=\s*(-?[0-9]*\.?[0-9]+)f?\s*;", ino)
    a = float(m.group(1)) if m else None
    b = getattr(mod, n, None)
    if a is None or b is None or abs(float(a) - float(b)) > 1e-9:
        bad.append((n, a, b))

if bad:
    print("FSM SYNC FAILED")
    for n, a, b in bad:
        print(f"  {n:<20} .ino={a}   fsm_sim={b}")
    sys.exit(1)
print(f"FSM sync OK — {len(NAMES)}/{len(NAMES)} constants agree")
