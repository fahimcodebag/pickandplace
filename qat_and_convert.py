"""
qat_and_convert.py
------------------
QAT fine-tuning + TFLite INT8 conversion for the TD3 actor network.
Works with: PyTorch 2.9, litert-torch (latest), torchao 0.16

Strategy:
  - QAT uses torch.ao.quantization (eager mode) for fake-quant training.
    This is deprecated but still WORKS in 2.9 — removal is planned for 2.10.
  - Conversion uses litert_torch.convert() on the RAW FP32 model + TFLite
    converter flags to apply INT8 PTQ inside the converter itself.
    This completely avoids the broken pt2e_quantizer.annotate() bug.

Why this approach:
  - litert_torch.convert() accepts any nn.Module — no pt2e export needed.
  - TFLite converter flags apply full-integer INT8 PTQ during conversion.
  - QAT fine-tuning pre-conditions the FP32 weights to be more
    quantization-friendly before the converter quantizes them.

Usage:
    python qat_and_convert.py \
        --actor_path ../temp/td3/actor_td3 \
        --input_dims 46 \
        --n_actions 8 \
        --fc1 256 --fc2 128 \
        --output_path actor_int8.tflite

    # Then generate C header:
    python tflite_to_header.py actor_int8.tflite actor_int8.h
"""

import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Suppress TF/JAX noise before imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PJRT_DEVICE"] = "CPU"  # Required by litert_torch

import litert_torch
import tensorflow as tf


# ─────────────────────────────────────────────────────────
# Model — must match your trained networks.py
# ─────────────────────────────────────────────────────────

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, n_actions=2):
        super().__init__()
        if isinstance(input_dims, (tuple, list)):
            input_dims = input_dims[0]
        self.fc1    = nn.Linear(input_dims, fc1_dims)
        self.fc2    = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.output(x))


# ─────────────────────────────────────────────────────────
# QAT wrapper — wraps ActorNetwork with QuantStub/DeQuantStub
# ─────────────────────────────────────────────────────────

class ActorNetworkQAT(nn.Module):
    """
    Wraps ActorNetwork with fake-quantization nodes for QAT.
    Uses torch.ao.quantization eager mode (still functional in PyTorch 2.9).
    """
    def __init__(self, base_model: ActorNetwork):
        super().__init__()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress deprecation warnings
            from torch.ao.quantization import QuantStub, DeQuantStub
        self.quant   = QuantStub()
        self.fc1     = base_model.fc1
        self.fc2     = base_model.fc2
        self.output  = base_model.output
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.output(x))
        x = self.dequant(x)
        return x


# ─────────────────────────────────────────────────────────
# STEP 1: QAT fine-tuning
# ─────────────────────────────────────────────────────────

def run_qat(model_fp32, calib_states, args):
    """
    Fine-tune with fake-quantization active.
    Uses distillation: QAT student learns to match FP32 teacher outputs.
    Returns the QAT-conditioned FP32 weights back into the original model.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch.ao.quantization as tq

    # Frozen FP32 teacher
    teacher = copy.deepcopy(model_fp32).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Build QAT model
    qat_model = ActorNetworkQAT(copy.deepcopy(model_fp32))
    qat_model.qconfig = tq.get_default_qat_qconfig('x86')
    qat_model = tq.prepare_qat(qat_model.train())

    optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * (len(calib_states) // args.batch_size),
        eta_min=args.lr / 50
    )

    calib_t = torch.tensor(calib_states, dtype=torch.float32)
    n_batches = len(calib_states) // args.batch_size

    print(f"\nQAT fine-tuning: {args.epochs} epochs × {n_batches} batches")
    print(f"  Loss: MSE distillation vs FP32 teacher")
    print("-" * 50)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        idx = torch.randperm(len(calib_t))
        epoch_loss = 0.0

        for b in range(n_batches):
            batch = calib_t[idx[b * args.batch_size : (b + 1) * args.batch_size]]

            with torch.no_grad():
                target = teacher(batch)

            pred = qat_model(batch)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg = epoch_loss / n_batches
        if avg < best_loss:
            best_loss = avg

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"loss={avg:.6f} | best={best_loss:.6f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    # Extract the QAT-conditioned weights back into a plain ActorNetwork
    # (litert_torch.convert needs a plain nn.Module, not the QAT wrapper)
    print("\nExtracting conditioned FP32 weights from QAT model...")
    qat_model.eval()

    # Copy weights from QAT model back to a clean ActorNetwork
    refined = ActorNetwork(args.input_dims, args.fc1, args.fc2, args.n_actions)
    refined.fc1.weight.data    = qat_model.fc1.weight.data.clone()
    refined.fc1.bias.data      = qat_model.fc1.bias.data.clone()
    refined.fc2.weight.data    = qat_model.fc2.weight.data.clone()
    refined.fc2.bias.data      = qat_model.fc2.bias.data.clone()
    refined.output.weight.data = qat_model.output.weight.data.clone()
    refined.output.bias.data   = qat_model.output.bias.data.clone()
    refined.eval()

    # Validate
    with torch.no_grad():
        sample = calib_t[:64]
        mse = F.mse_loss(refined(sample), teacher(sample)).item()
    print(f"  Refined FP32 vs teacher MSE: {mse:.6f}")

    return refined


# ─────────────────────────────────────────────────────────
# STEP 2: Convert to INT8 TFLite via litert_torch
# ─────────────────────────────────────────────────────────

def convert_to_tflite(model_fp32, calib_states, output_path, input_dims,
                      per_tensor=False):
    """
    Convert a plain FP32 nn.Module to INT8 .tflite using litert_torch.
    Quantization is applied inside the TFLite converter using a
    representative dataset (PTQ) — no pt2e_quantizer needed.
    """
    print(f"\nConverting to TFLite INT8...")
    model_fp32.eval()

    sample_inputs = (torch.randn(1, input_dims),)

    # Build representative dataset generator for TFLite INT8 calibration
    # This is what the TFLite converter uses to compute INT8 scale/zero_point
    calib_t = torch.tensor(calib_states[:500], dtype=torch.float32)

    def representative_dataset():
        for i in range(len(calib_t)):
            sample = calib_t[i].unsqueeze(0).numpy()
            yield [sample]

    # TFLite converter flags for full INT8 quantization
    # inference_input_type / inference_output_type = float32 means:
    #   - Internal ops run in INT8 (fast, small)
    #   - Model input/output stay as float32 (easy to use from ESP32)
    converter_flags = {
        "optimizations":            [tf.lite.Optimize.DEFAULT],
        "representative_dataset":   representative_dataset,
        # Keep input/output as float for easy ESP32 integration.
        # Change both to tf.int8 if you need full integer I/O.
        "inference_input_type":     tf.float32,
        "inference_output_type":    tf.float32,
        "target_spec.supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8],
    }

    # PER-TENSOR weight quantization. Desktop TFLite defaults to PER-CHANNEL,
    # which ESP32 builds often execute with ESP-NN optimized int8 kernels that
    # are known to be inaccurate. Symptom on hardware: outputs whose true value
    # is saturated (+-1) come out correct, while mid-range (tanh-linear) outputs
    # are badly wrong — even sign-flipped — which destroys fine control such as
    # holding the gripper open during the approach. Per-tensor weights avoid
    # that kernel path entirely at a small accuracy cost.
    if per_tensor:
        converter_flags["_experimental_disable_per_channel"] = True
        print("  Using PER-TENSOR weight quantization (ESP32 kernel workaround)")

    edge_model = litert_torch.convert(
        model_fp32,
        sample_inputs,
        _ai_edge_converter_flags=converter_flags,
    )

    # Quick sanity check
    print("  Running sanity check (PyTorch vs TFLite outputs)...")
    with torch.no_grad():
        pt_out = model_fp32(sample_inputs[0]).numpy()
    tfl_out = edge_model(*sample_inputs)

    import numpy as np
    max_diff = np.max(np.abs(pt_out - tfl_out))
    print(f"  Max output diff (PyTorch vs TFLite): {max_diff:.5f}")
    if max_diff < 0.05:
        print("  ✓ Outputs match within tolerance")
    else:
        print("  ⚠ Outputs differ — check quantization accuracy")

    # Export to .tflite file
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    edge_model.export(output_path)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\n{'='*55}")
    print(f"  ✓ Saved: {output_path}")
    print(f"  ✓ Size:  {size_kb:.1f} KB")
    print(f"{'='*55}")
    print(f"\nNext: python tflite_to_header.py {output_path} actor_int8.h")


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main(args):
    # Load FP32 actor
    model = ActorNetwork(args.input_dims, args.fc1, args.fc2, args.n_actions)
    if os.path.exists(args.actor_path):
        model.load_state_dict(torch.load(args.actor_path, map_location="cpu"))
        print(f"✓ Loaded actor from: {args.actor_path}")
    else:
        print(f"WARNING: {args.actor_path} not found — using random weights!")
    model.eval()

    # Calibration / QAT data
    if args.replay_buffer_path and os.path.exists(args.replay_buffer_path):
        print(f"✓ Loading replay buffer: {args.replay_buffer_path}")
        buf = np.load(args.replay_buffer_path)
        states = buf["states"].astype(np.float32)
        np.random.shuffle(states)
    else:
        print("⚠ No replay buffer — using random calibration data")
        print("  (QAT accuracy will be sub-optimal; provide real states for best results)")
        states = np.random.randn(10000, args.input_dims).astype(np.float32)

    # Step 1: QAT (optional but recommended)
    if args.skip_qat:
        print("\nSkipping QAT (--skip_qat flag set)")
        refined = model
    else:
        refined = run_qat(model, states, args)

    # Save QAT-refined FP32 weights (useful for inspection / resuming)
    os.makedirs("qat_output", exist_ok=True)
    torch.save(refined.state_dict(), "qat_output/actor_qat_refined_fp32.pt")
    print("  Saved QAT-refined FP32 weights → qat_output/actor_qat_refined_fp32.pt")

    # Step 2: Convert to INT8 TFLite
    convert_to_tflite(refined, states, args.output_path, args.input_dims,
                      per_tensor=args.per_tensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QAT fine-tune + INT8 TFLite export for TD3 actor"
    )
    parser.add_argument("--actor_path",  default="temp/td3/actor_td3")
    parser.add_argument("--input_dims",  type=int, default=46)
    parser.add_argument("--n_actions",   type=int, default=8)
    parser.add_argument("--fc1",         type=int, default=256)
    parser.add_argument("--fc2",         type=int, default=128)
    parser.add_argument("--epochs",      type=int, default=50,
                        help="QAT fine-tuning epochs (50 is enough for MLPs)")
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--replay_buffer_path", default=None)
    parser.add_argument("--output_path", default="actor_int8.tflite")
    parser.add_argument("--skip_qat",   action="store_true",
                        help="Skip QAT and convert FP32 model directly (PTQ only)")
    parser.add_argument("--per_tensor", action="store_true",
                        help="Per-tensor (not per-channel) weight quantization. "
                             "Workaround for inaccurate ESP32/ESP-NN int8 "
                             "per-channel kernels — see convert_to_tflite().")
    args = parser.parse_args()
    main(args)