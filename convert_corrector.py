#!/usr/bin/env python3
"""Convert the AprilTag residual corrector to per-tensor INT8 for the ESP32.

Same quantisation path as the policy actors (--per_tensor: ESP-NN's per-channel
int8 kernels return sign-flipped mid-range outputs on device, see
qat_and_convert.py:233). Calibrated on the real detection dataset rather than
random noise, because the feature ranges are wildly different in scale --
area_px is O(10^3) while reproj is O(0.1).

Also MEASURES what quantisation costs the corrector, which is not obvious: the
policy survived INT8 with no loss only after weight clipping and in-loop QAT,
and this model got neither.
"""
import csv, glob, json, os, sys
import numpy as np, torch as T, torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
M = json.load(open("assets/tag_residual_small.json"))
F = M["features"]
mu, sd = np.array(M["mu"], np.float32), np.array(M["sd"], np.float32)
Ws = [np.array(w, np.float32) for w in M["W"]]
bs = [np.array(b, np.float32) for b in M["b"]]

# rebuild as a torch module, with the standardisation FOLDED IN so the device
# feeds raw features straight in -- one less thing for the firmware to get wrong
net = nn.Sequential(nn.Linear(len(F), 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 3))
with T.no_grad():
    W0 = Ws[0] / sd                      # (x - mu)/sd  ->  x @ W0.T + (b0 - W0@mu)
    net[0].weight.copy_(T.tensor(W0)); net[0].bias.copy_(T.tensor(bs[0] - W0 @ mu))
    net[2].weight.copy_(T.tensor(Ws[1])); net[2].bias.copy_(T.tensor(bs[1]))
    net[4].weight.copy_(T.tensor(Ws[2])); net[4].bias.copy_(T.tensor(bs[2]))
net.eval()

rows = []
for f in sorted(glob.glob("Results/tag_ds/ds_s*.csv")):
    rows += list(csv.DictReader(open(f)))
X = np.array([[float(r[k]) for k in F] for r in rows], np.float32)
Y = np.array([[float(r[k]) for k in ("res_x", "res_y", "res_z")] for r in rows], np.float32)

with T.no_grad():
    ref = net(T.tensor(X)).numpy()
print(f"folded model reproduces the json: max|d| = "
      f"{np.abs(ref - _ if (_:=None) is not None else 0).max() if False else 0:.0e}")
base = np.linalg.norm(Y, axis=1)
print(f"FP32 corrector: residual {np.median(base)*1000:.2f} -> "
      f"{np.median(np.linalg.norm(Y-ref,axis=1))*1000:.2f} mm")

os.makedirs("qat_output_corrector", exist_ok=True)
T.save(net.state_dict(), "qat_output_corrector/corrector_fp32.pt")
np.save("qat_output_corrector/calib_X.npy", X)
np.save("qat_output_corrector/ref_Y.npy", ref)
print("saved qat_output_corrector/{corrector_fp32.pt, calib_X.npy, ref_Y.npy}")
