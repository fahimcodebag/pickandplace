# Project Context — Staged RL for MCU Deployment (Panda Pick-and-Place)

> **Purpose of this file:** Paste this into a new conversation so the assistant has full project context without re-exploration. Last updated: 2026-06-30.

---

## 1. Thesis Goal

Deploy complex RL control models for a **7-DOF Panda robot** on **ultra-low-cost MCUs (ESP32)**.

- **Environment:** Robosuite `PickPlace` task with `Panda` robot, `OSC_POSE` controller, `bread` object, `single_object_mode=2`.
- **Algorithm:** TD3 (Twin Delayed DDPG) with Prioritized Experience Replay (PER).
- **Observation space:** 46-dimensional (joint states, gripper, object pose, etc.).
- **Action space:** 7-dimensional continuous (OSC_POSE: 3 position + 3 orientation + 1 gripper).
- **Current model:** 512→256 MLP actor (~630 KB as float32 TFLite). Critic: 512→256.
- **Deployment target:** ESP32 via TFLite Micro, serial HIL (Hardware-in-the-Loop) with robosuite sim on PC.

### The Core Problem

A monolithic TD3 policy is too large and fails to generalize:
- **Fixed bread spawn:** 1 MB model learns full pick-and-place in ~500 episodes.
- **Random bread spawn:** Even an 8 MB model fails after 60,000 episodes — it learns to grasp but **stutters post-grasp and cannot place**. This failure pattern persists even with HER + PER + dense reward shaping.

### The Proposed Solution: Staged/Hierarchical RL

Train **separate small RL models** for each stage of the task:

| Stage | Behavior | Transition Trigger (FSM) |
|-------|----------|--------------------------|
| 1. Reach | Move gripper toward object | `d(eef, obj) < threshold` |
| 2. Grasp | Close gripper on object | `_check_grasp() == True` |
| 3. Lift | Lift object above table | `obj_z - table_z > lift_height` |
| 4. Transport | Move held object to target bin XY | `d(obj_xy, target_xy) < threshold` |
| 5. Place | Lower and release at target | `_check_success() == True` or release done |

Each sub-policy uses a **much smaller network** (e.g., 64→32 or 32→16 MLP) since it only needs to approximate one narrow behavior.

The meta-controller is a **rule-based FSM** (not learned) — cheap and deterministic on ESP32.

### Key Thesis Claims to Validate

1. **Decomposition solves training failure:** Staged training overcomes the exploration/credit-assignment collapse seen in monolithic training under randomized initial conditions.
2. **Smaller total model size:** Sum of all quantized stage models < size of a monolithic model that achieves equivalent success (requires finding a successful monolithic baseline, or reframing as "monolithic scaling failed; decomposition made deployment-viable sizes achievable at all").
3. **MCU-viable deployment:** Each stage model fits ESP32 constraints; FSM switches models at stage boundaries.
4. **Post-quantization viability:** QAT or post-training quantization preserves per-stage success rates.

### Critical Implementation Detail: Handoff State Sampling

When training stage N, initial states must be sampled from the **actual terminal-state distribution of stage N-1** (run stage N-1 policy, collect real exit states including noise/failures), NOT from idealized resets. This is the #1 failure mode of naive stage decomposition — distribution shift at handoffs.

---

## 2. Project File Structure

All code lives in: `/home/fahim__/RL_implementaion_and_simlation_manipulator/pickandplace/`

### Core Training & Algorithm

| File | Purpose | Key Details |
|------|---------|-------------|
| `networks.py` | Actor (512→256→7, tanh) and Critic (512→256→1) networks | PyTorch `nn.Module`, parameterized `fc1_dims`/`fc2_dims`, handles tuple/int `input_dims` |
| `td3.py` | TD3 Agent with PER | `choose_action()`, `choose_action_batch()`, `learn()` with IS-weighted critic loss, target network soft update (tau), warmup period, batch support for vectorized envs |
| `buffer.py` | PER replay buffer with SumTree | `store_transition()`, `sample_buffer_per()` returns `(s, a, r, s', done, tree_idx, is_weights)`, `update_priorities()`, save/load support |
| `train_vectorized.py` | Main training script | Subprocess-parallel vectorized envs (fork), `CurriculumRewardWrapper` with 6-stage dense reward, fixed bread spawn via placement initializer override, TensorBoard logging, periodic checkpointing |
| `utils_rl.py` | RL utilities | `OUNoise`, `NoiseScheduler`, `RewardNormalizer`, `compute_reward()` (sparse goal-conditioned), `potential_reward_shaping()`, `compute_staged_reward()` |
| `buffer_her.py` | HER-enabled replay buffer | Hindsight Experience Replay variant |

### Reward Structure (CurriculumRewardWrapper in train_vectorized.py)

Already decomposed into stages — directly maps to proposed sub-policies:

```
Stage 1 - REACH:   W_REACH=1.0, scale=0.30 (gripper→object proximity)
Stage 2 - GRIP:    W_GRIP_CLOSE=0.5, range=0.06 (close gripper when near)
Stage 3 - GRASP:   W_GRASP=10.0 (sustained grasp bonus)
Stage 4 - LIFT:    W_LIFT=3.0, ceil=0.12 (proportional lift height)
Stage 5 - HOVER:   W_HOVER=0.5, scale=0.25 (object→target bin XY)
Stage 6 - SUCCESS: W_SUCCESS=50.0 (one-time completion bonus)
Penalties: P_IDLE=-0.4, P_DROP=-5.0, P_AWAY=-0.7
```

### Deployment Pipeline

| File | Purpose | Key Details |
|------|---------|-------------|
| `convert_float32_tflite.py` | PyTorch→Keras→TFLite FP32 conversion | Copies weights layer-by-layer, verifies accuracy, benchmarks inference. Hardcoded `STATE_DIM=46`, `ACTION_DIM=7`, `fc1=512`, `fc2=256` |
| `tflite_to_header.py` | TFLite→C header array | For embedding model in ESP32 firmware |
| `pick_and_place_FP32.ino` | ESP32 Arduino firmware | TFLite Micro inference, 100KB tensor arena, sync-pattern serial protocol, PSRAM support, memory diagnostics |
| `protocol_float32.py` | Python-side serial protocol | `ProtocolFloat32.encode_state()` / `decode_action()`, 4-byte sync pattern `0xAA55AA55`, checksum, also has legacy INT8 `Protocol` class |
| `esp32_bridge.py` | Serial bridge to ESP32 | `ESP32Bridge.get_action(state)` with retry logic, stats tracking, 921600 baud |
| `hil_main.py` | Hardware-in-the-Loop testing | Runs robosuite episodes with ESP32 inference, logs per-episode JSON data, fixed bread spawn matching training |

### Model Artifacts

| File | Size | Description |
|------|------|-------------|
| `actor_float32.tflite` | 631 KB | Current monolithic FP32 TFLite model |
| `actor_model_float32.h` | 3.9 MB | C header version of above |
| `demos_bread.npz` | 16.6 MB | Collected demonstration data |
| `checkpoints/td3/` | — | TD3 checkpoint directory |
| `checkpoints/td3_builtin/` | — | Alternative checkpoint directory |

### Other Training Script Variants

Multiple iterations exist: `train_v2.py` through `train_v8.py`, `train_vectorized_bigger.py`, `train_vectorized_builtin.py`, `train_vision.py`, `networks_bigger.py`, `networks_v2.py`, `networks_vision.py`, `td3_bigger.py`, `td3_v2.py`, `td3_vision.py` — these represent earlier experiments with different architectures/approaches.

### Test/Evaluation Scripts

`test.py` (main, with inline CurriculumRewardWrapper), `test_bigger.py`, `test_builtin.py`, `test_v2.py` through `test_v8.py`, `test_vision.py`, `test_protocol.py`, `diagnose_vectorized.py`.

---

## 3. Environment Configuration

```python
env = suite.make(
    "PickPlace",
    robots="Panda",
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon=500,
    reward_shaping=False,   # custom CurriculumRewardWrapper used instead
    control_freq=20,
    single_object_mode=2,   # single fixed object
    object_type="bread",
)
```

**Fixed spawn override** (used in training and testing):
```python
_orig_gpi = env._get_placement_initializer
def _fixed_placement():
    _orig_gpi()
    s = env.placement_initializer.samplers["CollisionObjectSampler"]
    s.x_range = np.array([0.0, 0.0])
    s.y_range = np.array([0.0, 0.0])
    s.rotation = 0.0
    s.ensure_object_boundary_in_range = False
    s.ensure_valid_placement = False
env._get_placement_initializer = _fixed_placement
```

**Key observation dict keys available from robosuite:**
- `robot0_eef_pos` — end-effector position (3D)
- `robot0_gripper_qpos` — gripper joint positions
- `Bread_pos` — bread position (3D)
- Object detection: `_check_grasp(gripper, object_geoms)`, `_check_success()`
- Target bin: `env.target_bin_placements[obj_idx]`

### Training Hyperparameters (current monolithic)

```python
actor_lr = 0.0005
critic_lr = 0.0005
batch_size = 1024
layer1_size = 512
layer2_size = 256
tau = 0.005
warmup = 25000
max_buffer_size = 500000
n_envs = 8  # subprocess-parallel, fork
gamma = 0.99
noise = 0.1
update_actor_interval = 2
```

---

## 4. Key Observations from Robosuite Environment

These are relevant for defining stage transitions in the FSM:

- **Grasp detection:** `env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.objects[env.object_id])`
- **Success detection:** `env._check_success()`
- **Object position:** `obs_dict["Bread_pos"]` (3D numpy array)
- **End-effector position:** `obs_dict["robot0_eef_pos"]` (3D)
- **Gripper aperture:** `obs_dict["robot0_gripper_qpos"]` — qpos approx [+0.02, -0.02] when open, [0, 0] when closed
- **Table height:** ~0.82–0.845 (bread resting z)
- **Target bin position:** `env.target_bin_placements[env.object_to_id[name.lower()]]`

---

## 5. What Needs to Be Built

### 5.1 Per-Stage Environment Wrappers
- Each stage needs its own wrapper with:
  - **Stage-specific reward** (isolated from the existing CurriculumRewardWrapper stages)
  - **Stage-specific reset** (e.g., "lift" stage resets with object already grasped in gripper)
  - **Stage-specific termination** (success = reached transition condition; failure = timeout or drop)
  - **Possibly pruned observation space** (each stage may not need all 46 dims)

### 5.2 Handoff State Collection & Sampling
- Run trained stage N-1 policy -> collect terminal states -> use as initial state distribution for stage N training
- Must include failure cases and noisy outcomes, not just idealized successes

### 5.3 Per-Stage Training Scripts
- Same TD3 algorithm, but with smaller networks per stage (e.g., 64→32)
- Per-stage checkpointing, logging, evaluation

### 5.4 FSM Meta-Controller
- Rule-based (not learned)
- Transition conditions based on sensor readings (distances, grasp state, heights)
- Runs on ESP32 alongside the active stage model

### 5.5 Multi-Model ESP32 Firmware
- Store all stage TFLite models in flash
- Load only the active stage's model into RAM
- FSM determines which model to use and when to swap

### 5.6 Updated Conversion Pipeline
- `convert_float32_tflite.py` needs to handle variable STATE_DIM, ACTION_DIM, layer sizes per stage
- Separate `.tflite` and `.h` files per stage

### 5.7 End-to-End Chained Evaluation
- Run full pipeline: stage 1 -> handoff -> stage 2 -> ... -> stage 5
- Report per-stage AND end-to-end success rates
- Critical: 95% per-stage x 4 stages = ~81% end-to-end — must report honestly

### 5.8 Ablation: Place-Only Sub-Task (Recommended First)
- Train "place" alone with scripted grasp + randomized target
- If hard to learn -> confirms exploration/horizon is the bottleneck, not parameter interference
- If learns fast -> confirms parameter interference is real
- Either result strengthens thesis

---

## 6. Design Decisions & Theoretical Framing (from analysis)

### What this approach IS:
- **Hierarchical/Modular RL** (skill chaining / options framework)
- Multiple single-agent policies, each responsible for one temporal stage
- One agent (Panda arm) using different policies **sequentially over time**

### What this approach is NOT:
- **NOT MARL** (Multi-Agent RL) — that's multiple agents acting concurrently
- **NOT Model-Based RL** — still model-free TD3, just decomposed

### RL's Genuine Advantage (for thesis defense):
- Strongest for **contact-rich stages** (grasp, place) where dynamics are hard to model analytically
- For free-space stages (reach, transport), RL's advantage is weaker unless you inject **randomized disturbances/obstacles** during training
- Defensible framing: "RL learns recovery strategies from data, avoiding the need for a human to enumerate and hand-code responses to every disturbance variation"
- **Don't overstate:** Classical control isn't "helpless" — it's that RL shifts the generalization burden from human engineer to learning algorithm

### Decomposition Benefits (two independent axes):
1. **Training:** Fixes exploration/credit-assignment collapse under randomization (each stage has narrow, specifiable success condition)
2. **Deployment:** Each stage model is small enough for MCU (this is a separately-measurable benefit)

### Residual RL / IL+RL Consideration:
- Was discussed but NOT planned for initial implementation
- If pure RL per stage proves too sample-hungry for grasp/place, could add imitation learning base + small RL correction as a future extension
- Key caveat: residual RL's size advantage only holds when the base is non-learned (spline/trajectory) and corrections are small

---

## 7. ESP32 Hardware Constraints

- **Flash:** Typically 4 MB (some variants 8/16 MB) — all stage models stored here (additive)
- **RAM (SRAM):** ~320 KB — only one stage model needs to be resident at a time
- **PSRAM:** Some boards have 4–8 MB PSRAM (slower, but usable for tensor arena)
- **Current tensor arena:** 100 KB allocated for inference
- **Current inference time:** ~30 ms per step (FP32), ~15 ms (INT8)
- **Serial baud rate:** 921600
- **Control frequency:** 20 Hz (50 ms per step budget — inference must complete within this)

---

## 8. Dependencies

- Python: PyTorch, NumPy, robosuite, gymnasium (gym), TensorFlow (for TFLite conversion), TensorBoard, pyserial
- ESP32: ArduTFLite, TensorFlow Lite Micro
- Training hardware: CUDA GPU (if available), Linux (fork-based multiprocessing)

---

## 9. Summary: Current State -> Next Steps

**Current state:** Monolithic TD3 training pipeline is complete and working for fixed-spawn bread pick-and-place. ESP32 HIL deployment pipeline is functional end-to-end. Random-spawn training has failed with monolithic approach even at 8 MB model size.

**Immediate next step:** Implement the staged/hierarchical RL decomposition — this is the main remaining thesis contribution. All infrastructure (TD3, PER buffer, vectorized training, TFLite conversion, ESP32 firmware, HIL testing) exists and needs adaptation, not rewriting.
