# Repository Map — `pickandplace`

**What this file is:** the index. Every directory, what lives in it, and which
script or results file answers which question. Read this first; it tells you
where to go next.

**What this file is NOT:** the narrative (that is `thesis_context.md`) or the
evidence (that is `Results/*.txt`). Numbers are not duplicated here — only
pointers, so this file cannot go stale by disagreeing with a measurement.

Last surveyed: 2026-09-08 · 118 commits · branch `main`

---

## 0. The three documents, and which one you want

| File | Role | When to read it |
|---|---|---|
| `REPOSITORY_MAP.md` (this) | **Index.** Where everything is. | "Where is X?" / new to the repo |
| `thesis_context.md` (1310 ln) | **Narrative.** What was tried, what failed, what superseded what. §9.16 is the fresh-session entry point, §9.17 the newest finding; §10 is the claim-status table. | "Why is it like this?" / resuming work |
| `Results/*.txt` (39 files) | **Evidence.** Measured tables at protocol, with method and caveats. | "What is the number, and how solid?" |

`PROJECT_CONTEXT.md` (281 ln, last updated 2026-06-30) is an **older** context
primer from the monolithic era and is superseded by `thesis_context.md` for
everything after Stage-2 decomposition. Keep for history; do not cite.

**Rule that earned itself:** findings documented only in a results file get
re-derived. The joint-5 joint-limit stall was recorded in both
`fsm_sim.py` and `Results/transport_stall_diagnosis.txt` and was still
rediscovered from scratch months later. A finding must be reachable from
`thesis_context.md` §9.16 or from this file, or it is effectively lost.

**Recently revised:** the **random-spawn** headline cells (`thesis_context.md`
§9.17) — FP32 90.67 → **93.58%**, INT8 78.33 → **83.58%**, quantization cost
12.33 → **10.00**. Fixed-spawn cells are unchanged and un-re-measured.

**Harness note:** measure through `fsm_sim.py` itself, not a reimplementation.
`main()` runs a respawn-and-retry handoff loop that a custom driver will miss,
and it is worth ~2 points.

---

## 1. Top level — the pipeline that ships

### Simulation harness / evaluation
| File | What it does |
|---|---|
| `fsm_sim.py` | **The main harness.** 9-phase FSM meta-controller over the two actors. Hosts BOTH the original grasp+place pipeline (place actor drives transport) and the scripted transport variants (`--waypoint-transport`, `--staged-transport`, `--scripted-transport`). All rule-layer constants are module-level and overridable by CLI. |
| `eval_best.py`, `eval_int8.py` | Single-stage policy evaluation, FP32 and INT8. |
| `check_env.py` | Environment / observation-layout sanity checks. |
| `check_fsm_sync.py` | Asserts the Python FSM constants match `pick_and_place_INT8_FSM.ino`. Host-only constants are excluded by design until mirrored. |
| `failure_map.py`, `phase2_diag.py`, `handoff_diag.py`, `handoff_carry.py`, `grasp_diagnostics.py` | Failure-class instrumentation feeding the diagnosis results files. |

### Training (current, decomposed)
| File | What it does |
|---|---|
| `Decomposed state training/train_grasp.py` | Stage-1 grasp actor, fixed spawn. |
| `Decomposed state training/train_place.py` | Stage-2 place/transport actor. |
| `Decomposed state training/Random spawn model/train_rand.py` | Random-spawn grasp training. `--object-type`, `--cold-start`, `--grasp-horizon`, warm-start, QAT-in-loop, critic resets. |
| `Decomposed state training/grasp_env_wrapper.py` | Grasp reward/termination, object-aware grip gate, `truncated` flag. |
| `Decomposed state training/place_env_wrapper.py` | Place-stage wrapper. |
| `Decomposed state training/Random spawn model/grasp_spawn_wrapper.py` | Spawn-randomisation ladder (level 1.0 = fixed yaw, 2.0 = full circle). |
| `networks.py` / `networks_v2.py` / `networks_bigger.py` / `networks_sac.py` | Actor/critic definitions. The shipped actor is 64→32. |
| `buffer.py`, `buffer_her.py`, `utils_rl.py` | Replay buffers and helpers. |

### Quantization / deployment chain
`train_*` → checkpoint → `convert_float32_tflite.py` → `.tflite` →
`tflite_to_header.py` → `*_model.h` → `.ino`

| File | What it does |
|---|---|
| `convert_float32_tflite.py` | Torch → TFLite (FP32 and INT8 paths). |
| `tflite_to_header.py` | `.tflite` → C header byte array. |
| `analyze_qat.py`, `analyze_wclip.py`, `analyze_widen.py` | QAT / weight-clipping / width sweeps. |
| `prune_actor.py`, `distill_actor.py` | Pruning and distillation of FP32 actors. |
| `pick_and_place_INT8_FSM.ino` | **The deployed firmware.** FSM + both INT8 actors + on-device AprilTag. |
| `pick_and_place_FP32.ino` | FP32 reference firmware. |
| `grasp_model.h`, `place_model.h`, `actor_model_float32.h`, `corrector_model.h` | Generated model headers. |

### Perception (AprilTag)
| File | What it does |
|---|---|
| `esp32_apriltag/` | Vendored AprilTag (BSD-2), Arduino-library layout, memory-reduced for ESP32. `src/`, `common/`, `host/` shim, `libat32.so`. |
| `at32_perception.h` | On-device detection + pose, mirrors `TagDetector.detect()` line for line. ROI, intrinsics, calibration constants live here. |
| `at32.py` | ctypes bridge to `libat32.so` for host-side parity testing. |
| `apriltag_sim.py`, `perception_wrapper.py`, `validate_tag_perception.py` | Simulated tag perception and validation. |
| `fit_wrist_corrector.py`, `fit_wrist16_corrector.py`, `fit_wrist32_corrector.py`, `convert_corrector*.py`, `tune_residual.py`, `wrist_dataset.py` | Wrist-camera pose corrector: fit, quantize, tune. |

### Hardware-in-the-loop
| File | What it does |
|---|---|
| `hil_main.py` | HIL driver. Sim on host, policy on ESP32 over serial. `--on-device-perception` sends image ROIs. |
| `esp32_bridge.py` | Serial transport, framing, debug-line capture, per-episode perception summary. |
|  `protocol_float32.py` | Wire format incl. `IMG_MSG`, ROI crop, 15-float pose payload. |
| `diag_esp32_actions.py` | Action-level host/device diff. |

### Legacy — monolithic era, kept for history
`train_v2..v8.py`, `test_v4..v8.py`, `train_vectorized*.py`, `train_vision.py`,
`test_vision.py`, `demo_collector.py`, `mono_grip_test.py`, `net2wider.py`.
These predate the decomposition; `train_v8.py` is the last monolithic loop
(two-phase reward). Superseded by `Decomposed state training/`.

---

## 2. Directories

| Directory | Contents |
|---|---|
| `Results/` | 39 `.txt` evidence files, ~85 experiment subdirs of raw CSVs, `figures/`. See §4. |
| `checkpoints/` | 161 run directories. Naming convention in §3. Each holds `best/` and periodic saves. |
| `logs/` | 174 training logs, one per run, named to match its checkpoint family. |
| `qat_output*/` | Quantization artifacts per campaign: `_bi` (the shipped pair), `_fix`, `_phaseB`, `_reset`, `_corrector`. Each holds `.tflite`, the `_float32.tflite` reference, and the refined `.pt`. |
| `Decomposed state training/` | Current training code + `Grasp_training_history.txt`. |
| `esp32_apriltag/` | Vendored detector (see §1). |
| `assets/` | Tag images, textures. |
| `Important Texts/` | Reference material. |

**Shipped artifact pair:** `qat_output_fix/grasp_bi_noqat_int8.tflite` +
`qat_output_bi/place_orig_int8.tflite`. The FP32 equivalents are
`checkpoints/td3_grasp_rand_td3_ln_bi_s0/best` + `checkpoints/td3_place/best`.
`.tflite` files in the repo **root** are stale (`thesis_context.md` §9.4).

---

## 3. Checkpoint naming

```
td3_grasp_rand_td3_ln_<experiment>_s<seed>
 |    |     |     |   |      |         └── seed
 |    |     |     |   |      └── experiment tag
 |    |     |     |   └── LayerNorm critics
 |    |     |     └── algorithm actually trained (td3 / sac / ppo)
 |    |     └── random spawn (absent = fixed spawn)
 |    └── stage: grasp | place
 └── base algorithm
```

Experiment tags in use: `phaseA` / `phaseB` / `phaseBr` (spawn ladder),
`resetA` (critic resets), `bi` (**the shipped pair**), `liftcert_*` (lift
certification), `gripfix*`, `rv2` (reward v2), `align` (yaw alignment),
`qat8` / `wclip*` / `wide128` / `smallc64` / `bigc512` / `c2m*` (capacity and
quantization sweeps), `buf200k/500k/1000k` (buffer size), `long60k`,
`cereal_{base,align}[warm]` (per-object, cereal).

`td3_place*` variants: `_bi_s*` (paired with the shipped grasp), `_rs_reset_*`
vs `_rs_control_*` (critic-reset ablation), `*_backup` (manual saves).

---

## 4. Results index — which file answers what

| File | Question it settles |
|---|---|
| `phaseA_algorithm_comparison.txt`, `phaseA_eval100.txt` | TD3 vs SAC vs PPO on the grasp stage. |
| `phaseB_rotation.txt` | Full yaw randomisation; critic resets replicate; capability is specialised in both directions. |
| `reset_ablation.txt`, `quantization_resetpolicy.txt` | Critic resets / plasticity loss. |
| `lift_certification.txt`, `gripfix_and_selection.txt`, `grip_robustness.txt` | Stage-1 certification and grip quality. |
| `handoff_diagnostic.txt` (superseded by `monolithic_control.txt`); raw data in `Results/handoff_carry/` | The stage interface; grip pose predicts carry survival. |
| `builtin_reward_results.txt`, `reward_v2_results.txt` | Reward shaping; robosuite's own shaped reward. |
| `capacity_experiments.txt`, `critic_capacity_2m.txt`, `buffer_size.txt`, `weight_range_regularisation.txt`, `qat_in_training.txt` | Capacity, buffer, weight-clipping, in-loop QAT. |
| `fsm_fp32_validation.txt`, `fsm_rule_layer_sweep.txt` | The FSM replica matches the eval harness; rule-layer tuning. |
| `transport_stall_diagnosis.txt` | **Central.** Three failure causes; Fix A/C adopted, Fix B/D rejected. Part 3's "blocked class is unfixable from the rule layer" is correct *as scoped*, and carries a forward pointer — superseded in **disposition**, not analysis, by `orientation_anchor.txt`. |
| `transport_retrain_negative.txt`, `wrist_alignment_negative.txt` | Settled negatives — do not retry. `KEEP_ROTATION=1` (−52 pts) is also settled, but do **not** read it as "rotation: closed" — see `thesis_context.md` §9.13. |
| `orientation_anchor.txt` | **Newest** (`thesis_context.md` §9.17). `a[3:6]=0` freezes `goal_ori` in robosuite's OSC → joint 5 saturates against its one-sided range. `ROT_ANCHOR_EPS=1e-6` recovers the blocked class: cereal 57.4→89.2%, bread main pipeline 88.7→91.4%. Also documents the cereal scoring artifact (physical 99.7% vs scored 89.7%) that confounds bread-vs-cereal comparisons. |
| `object_generalisation.txt`, `cereal_per_object.txt`, `orientation_ablation.txt`, `yaw_alignment.txt` | Beyond bread; per-object policies; ablations. |
| `pruning_distillation.txt` | Parameter reduction vs accuracy. |
| `int8_deployment.txt`, `validated_90_model.txt` | INT8 conversion and the validated model. |
| `apriltag_perception.txt`, `wrist_camera_route.txt`, `corrector_on_device.txt`, `tag16h5_deployment*.txt`, `on_device_perception.txt` | Perception, from sweep to on-device detection. |
| `hil_hardware_validation.txt`, `int8_FSM_HIL_performance.txt` | Hardware-in-the-loop. |
| `transport_diagnosis.txt` | Cereal transport failure analysis (pre-`orientation_anchor`). |

---

## 5. Operational notes

* **Evaluation protocol:** ≥12 eval seeds × 100 episodes, paired on seed.
  Seeds: `7 31 47 89 101 123 211 307 401 503 555 2024`. 100-episode evaluations
  ran ~10 points optimistic four separate times.
* **Parallelism:** MuJoCo will not parallelise a single environment, but
  (arm, seed) pairs are independent. Shard ~28-wide with `xargs -P 28` and pin
  one core per worker (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`).
  ~3400 episodes then finish in the wall time of one 100-episode job.
* **Two virtualenvs:** `../venv` (torch + robosuite), `../convert_venv` (TF).
* **Quantization acceptance is behavioural**, never output diff — the two were
  anti-correlated (`thesis_context.md` §9.12).
* **When a failure class survives every rule-layer remedy, suspect the layer
  below** (`thesis_context.md` §9.17).

### What is actually in git (checked 2026-09-08)

The working tree is ~46 GB; the repository is not. Do not assume a clone has
what this map describes.

| | on disk | in git |
|---|---|---|
| `checkpoints/` run dirs | 161 | 132 — **29 uncommitted** (recent sweeps: `bigc512`, `wclip*`, `cereal_*`, `qat8`, `wide128`, `buf*`, `c2m*`, `smallc64`, `cont64`, `long60k`) |
| `logs/` | 174 | **0 — `logs/` is in `.gitignore` entirely** |
| `Results/*.txt` | 39 | all tracked |

`.gitignore` excludes `checkpoints/**/replay_buffer.npz` (70–702 MB each,
regenerable, over GitHub's 100 MB limit) and all of `logs/`. Model weights
themselves *are* tracked. So: training logs exist only on this machine, and a
third of the checkpoint sweeps are local-only until committed. If the record is
meant to survive this host, those are the two gaps to close first.

### Known stale / open
* `.tflite` in repo root — stale (§9.4).
* `PROJECT_CONTEXT.md` — June, monolithic era.
* `ROT_ANCHOR_EPS` not mirrored to `.ino`; `check_fsm_sync.py` must gain it when it is.
* INT8 arm untested against `orientation_anchor.txt`.
* `hil_main.py` fixed-spawn hard-code (§9.16 item 2).
* `train_place.py` 50-episode best-window (§9.7).
