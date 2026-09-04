#!/usr/bin/env python3
"""INT8 conversion for the residual corrector, reusing qat_and_convert's
two-stage path (litert-torch float32 -> ai-edge-quantizer static PTQ) but with
the corrector's own architecture: LINEAR output, not the actor's tanh.

Run with convert_venv (needs TensorFlow / ai-edge-quantizer).
"""
import os, sys
import numpy as np, torch as T, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qat_and_convert as Q

net = nn.Sequential(nn.Linear(12, 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
import json
M = json.load(open("assets/tag_residual_small.json"))
mu = np.array(M["mu"], np.float32); sd = np.array(M["sd"], np.float32)
UNFOLDED = os.environ.get("UNFOLDED") == "1"
if UNFOLDED:
    # Feed the model ALREADY-STANDARDISED features and do the subtract/divide
    # on the device (24 flops). Folding it in gave the INT8 input tensor a
    # 2198x dynamic range across features, and per-tensor INT8 sets ONE input
    # scale -- so the small features (det_z sd 0.068) were crushed by the large
    # ones (cx sd 150).
    with T.no_grad():
        net[0].weight.copy_(T.tensor(np.array(M["W"][0], np.float32)))
        net[0].bias.copy_(T.tensor(np.array(M["b"][0], np.float32)))
        net[2].weight.copy_(T.tensor(np.array(M["W"][1], np.float32)))
        net[2].bias.copy_(T.tensor(np.array(M["b"][1], np.float32)))
        net[4].weight.copy_(T.tensor(np.array(M["W"][2], np.float32)))
        net[4].bias.copy_(T.tensor(np.array(M["b"][2], np.float32)))
else:
    net.load_state_dict(T.load("qat_output_corrector/corrector_fp32.pt",
                               map_location="cpu"))
net.eval()
X = np.load("qat_output_corrector/calib_X.npy")
if UNFOLDED:
    X = ((X - mu) / sd).astype(np.float32)
ref = np.load("qat_output_corrector/ref_Y.npy")

out = "qat_output_corrector/corrector_int8%s.tflite" % ("_unfolded" if UNFOLDED else "")
Q.convert_to_tflite(net, X, out, input_dims=12, per_tensor=True)

import tensorflow as tf
it = tf.lite.Interpreter(model_path=out); it.allocate_tensors()
inp, outd = it.get_input_details()[0], it.get_output_details()[0]
n = [len(d["quantization_parameters"]["scales"]) for d in it.get_tensor_details()
     if len(d["quantization_parameters"]["scales"])]
print(f"\ngranularity: {'PER-TENSOR' if set(n) == {1} else 'PER-CHANNEL!'}   "
      f"size {os.path.getsize(out)} B")

pred = np.zeros_like(ref)
for i in range(len(X)):
    v = X[i:i+1].astype(np.float32)
    sc, zp = inp["quantization"]
    if inp["dtype"] == np.int8:
        v = np.clip(np.round(v / sc + zp), -128, 127).astype(np.int8)
    it.set_tensor(inp["index"], v); it.invoke()
    o = it.get_tensor(outd["index"]); sc, zp = outd["quantization"]
    if outd["dtype"] == np.int8:
        o = (o.astype(np.float32) - zp) * sc
    pred[i] = o.reshape(-1)

Y = np.array([ref[i] for i in range(len(ref))])
d = np.linalg.norm(pred - ref, axis=1) * 1000
print(f"INT8 vs FP32 corrector output: median {np.median(d):.3f} mm  p95 {np.percentile(d,95):.3f}")
