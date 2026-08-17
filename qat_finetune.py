"""
qat_finetune.py
---------------
Step 1 of 3: Quantization-Aware Training (QAT) fine-tuning of the TD3 Actor.

Uses PyTorch's pt2e (PyTorch 2 Export) QAT flow, which is the modern API
compatible with litert-torch / ai_edge_torch for TFLite conversion.

Only the ACTOR is quantized — critics are training-only and never deployed.

Usage:
    python qat_finetune.py \
        --actor_path temp/td3/actor_td3 \
        --input_dims 46\
        --n_actions 8 \
        --fc1 512 --fc2 256 \
        --epochs 100 \
        --replay_buffer_path replay_buffer.npz  # optional, uses random if absent
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e, convert_pt2e

# ── Modern pt2e quantizer (works with litert-torch)
try:
    from ai_edge_torch.quantize import pt2e_quantizer
    USING_AI_EDGE = True
    print("Using ai_edge_torch PT2E quantizer")
except ImportError:
    try:
        from litert_torch.quantize import pt2e_quantizer
        USING_AI_EDGE = True
        print("Using litert_torch PT2E quantizer")
    except ImportError:
        # Fallback: use torch's own XNNPack quantizer (good for ARM/ESP32)
        from torch.ao.quantization.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
        USING_AI_EDGE = False
        print("Using XNNPACKQuantizer (fallback — litert-torch not installed)")


# ─────────────────────────────────────────────────────────
# Model definition (must match your trained networks.py)
# ─────────────────────────────────────────────────────────

class ActorNetwork(nn.Module):
    """
    Identical architecture to networks.py ActorNetwork.
    Defined here standalone so QAT doesn't depend on your full codebase.
    """
    def __init__(self, input_dims, fc1_dims=256, fc2_dims=128, n_actions=2):
        super().__init__()
        if isinstance(input_dims, (tuple, list)):
            input_dims = input_dims[0]
        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.output(x))


# ─────────────────────────────────────────────────────────
# Calibration / fine-tune data generator
# ─────────────────────────────────────────────────────────

def make_dataloader(replay_buffer_path, input_dims, batch_size=256, n_batches=200):
    """
    Returns a list of (state_batch,) tuples for calibration/QAT.
    Loads from a saved replay buffer .npz if available, else uses random data.
    """
    if replay_buffer_path and os.path.exists(replay_buffer_path):
        print(f"Loading replay buffer from {replay_buffer_path}")
        buf = np.load(replay_buffer_path)
        states = buf["states"]                          # shape (N, input_dims)
        # Shuffle
        idx = np.random.permutation(len(states))
        states = states[idx]
    else:
        print("No replay buffer found — using random calibration data")
        print("WARNING: QAT accuracy will be sub-optimal. Provide real states if possible.")
        n_samples = batch_size * n_batches
        states = np.random.randn(n_samples, input_dims).astype(np.float32)

    batches = []
    for i in range(0, min(len(states), batch_size * n_batches), batch_size):
        chunk = states[i : i + batch_size]
        if len(chunk) < batch_size:
            break
        batches.append(torch.tensor(chunk, dtype=torch.float32))
    return batches


# ─────────────────────────────────────────────────────────
# Main QAT routine
# ─────────────────────────────────────────────────────────

def run_qat(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load pretrained FP32 actor
    model = ActorNetwork(args.input_dims, args.fc1, args.fc2, args.n_actions)
    if os.path.exists(args.actor_path):
        state = torch.load(args.actor_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded pretrained actor from: {args.actor_path}")
    else:
        print(f"WARNING: {args.actor_path} not found — using random weights!")
    model.eval()

    # Keep a frozen FP32 teacher copy for distillation loss
    teacher = ActorNetwork(args.input_dims, args.fc1, args.fc2, args.n_actions)
    teacher.load_state_dict(model.state_dict())
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # 2. Build calibration data
    calib_data = make_dataloader(
        args.replay_buffer_path,
        args.input_dims,
        batch_size=args.batch_size,
        n_batches=200,
    )
    sample_inputs = (calib_data[0],)  # used for export/capture

    # 3. Export model to pt2e (torch.export) format
    print("\nExporting model to pt2e format...")
    exported_model = torch.export.export_for_training(
        model, sample_inputs
    ).module()

    # 4. Set up quantizer
    if USING_AI_EDGE:
        qconfig = pt2e_quantizer.get_symmetric_quantization_config(
            is_per_channel=True,
            is_qat=True,            # QAT mode — inserts fake-quant ops
            is_dynamic=False,       # Static quantization (needed for TFLite Micro)
        )
        quantizer = pt2e_quantizer.PT2EQuantizer().set_global(qconfig)
    else:
        quantizer = XNNPACKQuantizer().set_global(
            get_symmetric_quantization_config(is_qat=True)
        )

    # 5. Prepare model for QAT (inserts fake-quantization nodes)
    qat_model = prepare_qat_pt2e(exported_model, quantizer)
    qat_model.train()

    # 6. QAT fine-tuning loop
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=args.lr)

    # Cosine LR schedule: start at lr, decay to lr/100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(calib_data), eta_min=args.lr / 100
    )

    print(f"\nStarting QAT fine-tuning for {args.epochs} epochs...")
    print(f"  Batches/epoch : {len(calib_data)}")
    print(f"  Batch size    : {args.batch_size}")
    print(f"  LR            : {args.lr}")
    print(f"  Loss          : MSE distillation (QAT output vs FP32 teacher)")
    print("-" * 60)

    best_loss = float("inf")
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        np.random.shuffle(calib_data)  # shuffle batch order each epoch

        for batch in calib_data:
            with torch.no_grad():
                teacher_out = teacher(batch)

            qat_out = qat_model(batch)
            loss = F.mse_loss(qat_out, teacher_out)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(calib_data)

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save intermediate QAT checkpoint
            torch.save(qat_model.state_dict(),
                       os.path.join(args.output_dir, "actor_qat_best.pt"))

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch:4d}/{args.epochs} | "
                  f"Loss: {avg_loss:.6f} | Best: {best_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    print("\nQAT fine-tuning complete!")

    # 7. Convert fake-quant → real INT8 quantization
    print("Converting to INT8...")
    qat_model.eval()
    int8_model = convert_pt2e(qat_model)

    # 8. Validate: check that INT8 output is close to FP32 teacher
    print("\nValidation: comparing INT8 vs FP32 outputs on 10 random batches...")
    max_err = 0.0
    with torch.no_grad():
        for batch in calib_data[:10]:
            fp32_out = teacher(batch)
            int8_out = int8_model(batch)
            err = F.mse_loss(int8_out, fp32_out).item()
            max_err = max(max_err, err)
    print(f"  Max MSE (INT8 vs FP32): {max_err:.6f}")
    if max_err < 0.01:
        print("  ✓ Quantization looks good (MSE < 0.01)")
    else:
        print("  ⚠ MSE is high — consider more QAT epochs or smaller network")

    # 9. Save the final INT8 exported program
    out_path = os.path.join(args.output_dir, "actor_int8_pt2e.pt2")
    torch.export.save(int8_model, out_path)
    # Also save as regular state dict for inspection
    torch.save(int8_model.state_dict(),
               os.path.join(args.output_dir, "actor_int8_state_dict.pt"))

    print(f"\nSaved INT8 model to: {out_path}")
    print("→ Next step: run  python convert_to_tflite.py")

    return int8_model, sample_inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor_path",         default="temp/td3/actor_td3")
    parser.add_argument("--input_dims",  type=int, default=46,
                        help="Observation space dim (check env.observation_space)")
    parser.add_argument("--n_actions",   type=int, default=8,
                        help="Action space dim (check env.action_space)")
    parser.add_argument("--fc1",         type=int, default=256)
    parser.add_argument("--fc2",         type=int, default=128)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch_size",  type=int, default=256)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--replay_buffer_path", default=None,
                        help="Path to replay_buffer.npz with saved states")
    parser.add_argument("--output_dir",  default="qat_output")
    args = parser.parse_args()

    run_qat(args)
