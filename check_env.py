#!/usr/bin/env python3
"""Report which of this project's dependencies are present in the running
interpreter. Run it with the SPECIFIC python you intend to use -- the answer
differs between venvs, which is the whole point.
"""
import importlib.metadata as md, importlib.util, platform, sys

NEED = [
    ("numpy",       "core",     "everything"),
    ("scipy",       "core",     "paired stats in the analysis scripts"),
    ("torch",       "core",     "FP32 actors (fsm_sim, tag_e2e)"),
    ("mujoco",      "sim",      "physics"),
    ("robosuite",   "sim",      "PickPlace env, OSC_POSE controller"),
    ("gym",         "sim",      "GymWrapper (old gym, not gymnasium)"),
    ("serial",      "HIL",      "pyserial -- talks to the ESP32"),
    ("cv2",         "apriltag", "opencv, tag detection + video"),
    ("tensorflow",  "INT8",     "tflite interpreter (host-side INT8 runs)"),
    ("tensorboard", "optional", "training curves"),
]

print(f"python {platform.python_version()}  ({sys.executable})\n")
print(f"{'module':14} {'group':9} {'status':10} version")
print("-" * 62)
missing = []
for mod, grp, why in NEED:
    ok = importlib.util.find_spec(mod) is not None
    ver = ""
    if ok:
        for dist in (mod, {"cv2": "opencv-python", "serial": "pyserial"}.get(mod, mod)):
            try:
                ver = md.version(dist); break
            except Exception:
                pass
    print(f"{mod:14} {grp:9} {'OK' if ok else 'MISSING':10} {ver or ('-' if ok else why)}")
    if not ok:
        missing.append(mod)
print()
if missing:
    pip = {"cv2": "opencv-python", "serial": "pyserial"}
    print("install:  pip install " + " ".join(pip.get(m, m) for m in missing))
else:
    print("all present")
