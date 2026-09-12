# Repository Map — `pickandplace`

**What this file is:** the index. Every directory, what lives in it, and which
script or results file answers which question. Read this first; it tells you
where to go next.

**What this file is NOT:** the narrative (that is `thesis_context.md`) or the
evidence (that is `Results/*.txt`). Numbers are not duplicated here — only
pointers, so this file cannot go stale by disagreeing with a measurement.

Last surveyed: 2026-09-08 · artifact and results index revised 2026-09-11 · branch `orientation-anchor-fix`

---

## 0. The three documents, and which one you want

| File | Role | When to read it |
|---|---|---|
| `REPOSITORY_MAP.md` (this) | **Index.** Where everything is. | "Where is X?" / new to the repo |
| `thesis_context.md` (~1,600 ln) | **Narrative.** What was tried, what failed, what superseded what. The header carries the current headline; §9.16 is the fresh-session entry point, §9.18 the newest finding; §10 is the claim-status table. | "Why is it like this?" / resuming work |
| `Results/*.txt` (43 files) | **Evidence.** Measured tables at protocol, with method and caveats. | "What is the number, and how solid?" |

`PROJECT_CONTEXT.md` (281 ln, last updated 2026-06-30) is an **older** context
primer from the monolithic era and is superseded by `thesis_context.md` for
everything after Stage-2 decomposition. Keep for history; do not cite.

**Rule that earned itself:** findings documented only in a results file get
re-derived. The joint-5 joint-limit stall was recorded in both
`fsm_sim.py` and `Results/transport_stall_diagnosis.txt` and was still
rediscovered from scratch months later. A finding must be reachable from
`thesis_context.md` §9.16 or from this file, or it is effectively lost.

**Recently revised:** the deployed grasp is **`c2m512_s1`** (`thesis_context.md`
§9.15.1), replacing `bi_s0`. §9.17's `ROT_ANCHOR_EPS` helped `bi_s0` but **hurts
`c2m512_s1`** on random spawn (§9.17.1). `fsm_sim.py` now defaults to 0.0 (anchor off); place policies
trained with the anchor are evaluated with `--rot-anchor-eps 1e-6`.

**Resume point (2026-09-11, evening):** `thesis_context.md` §9.18 status block. Curriculum pair and cereal from-scratch arm finished and evaluated end-to-end (`Results/curriculum_pair/correlation_report.txt`, `Results/cereal_scratch/report.txt`): place training rises then degrades in every configuration.

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
| `pick_and_place_INT8_FSM.ino` | **The deployed firmware.** FSM + both INT8 actors + on-device AprilTag + FP32 corrector. |
| `pick_and_place_FP32.ino` | FP32 reference firmware. |
| `grasp_model.h`, `place_model.h`, `actor_model_float32.h`, `corrector_model.h` | Generated model headers. Each names its source file in its first lines. |

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
| `hil_main.py` | HIL driver. Sim on host, policy on ESP32 over serial. `--random-spawn`; `--on-device-perception` sends image ROIs. |
| `esp32_bridge.py` | Serial transport, framing, debug-line capture, per-episode perception summary. |
|  `protocol_float32.py` | Wire format incl. `IMG_MSG`, ROI crop, 15-float pose payload. |
| `diag_esp32_actions.py` | Action-level host/device diff. |

### Legacy — monolithic era, kept for history
`train_v2..v8.py`, `test_v4..v8.py`, `train_vectorized*.py`, `train_vision.py`,
`test_vision.py`, `demo_collector.py`, `mono_grip_test.py`, `net2wider.py`.
These predate the decomposition; `train_v8.py` is the last monolithic loop
(two-phase reward). Superseded by `Decomposed state training/`.

---

### Place-training diagnostics (§9.18)
| File | What it does |
|---|---|
| `run_curriculum_pair.sh` | Controlled fixed-vs-random place pair, bread, from scratch, snapshots. |
| `run_cereal_scratch.sh` | Cereal random-spawn place from scratch. |
| `run_place_big.sh`, `run_place_smallbuf.sh`, `run_place_pairfix.sh` | Recipe and warm-start arms. |
| `eval_place_snapshots.sh` + `eval_place_snapshot_job.sh` | End-to-end evaluation of every snapshot (resumable). |
| `analyze_place_snapshots.py` | Training metric vs end-to-end correlation, keyed on `time_step`. |
| `finish_curriculum_pair.sh` | Unattended: wait for trainers, final eval, write the report. |
| `eval_cereal_scratch.sh` + `eval_cereal_scratch_job.sh` | Cereal end-to-end evaluation: every `cerealscratch` snapshot plus the final actor of each warm-started cereal run; records robosuite success and physical rest (`--settle-steps 60`). Resumable. |
| `analyze_cereal_scratch.py` | Cereal from scratch vs warm-started cereal (and bread `curR`), keyed on transitions collected. |
| `stop_runs.sh` | Stop runs by pattern, filtering on `/proc` comm so it cannot kill its caller. |
| `osc_anchor.py` | `ROT_ANCHOR_EPS = 1e-6`, the TRAINING value (place wrapper). `fsm_sim.py` defaults to 0.0 since 2026-09-11 (§9.17.1). |
| `run_anchor_c2m512.sh` + `analyze_anchor_c2m512.py` | The anchor on the deployed artifact (§9.17.1): `baseline` (eps 0, reproduces the recorded CSVs), `treatment` (eps 1e-6, refuses to run without `PREDICTION.txt`), `mechanism` (tilt probe). |
| `anchor_probe.py` | Passive probe around `fsm_sim.main()`: blocked/pinned carries, gripper tilt, release point. Positive control on `bi_s0`. |
| `run_cereal_pairing.sh` + `analyze_cereal_pairing.py` | Cereal: trained place policy vs scripted transport vs transfer, paired 12×100, with reproduction controls (§9.18, §10). |
| `run_place_es_batch.sh` | Place training batches on cereal (A: custom reward + early stopping; B: potential-based built-in reward; C: B with shaping annealed toward sparse), 3 warm + 3 scratch each. `Results/place_training_batches.txt`. |
| `es_watch.py` | Early stopping on periodic end-to-end evaluation: scores each snapshot through `fsm_sim.py`, writes `<run>/STOP` after 4 unbeaten snapshots; `--no-stop` scores only. |
| `Decomposed state training/place_reward_audit.py` + `run_place_reward_audit.sh` + `analyze_place_reward_audit.py` | Prices every place reward mode on the same recorded episodes before training (§9.10 method), with exact accounting controls. |
| `run_place_selection.sh` + `analyze_place_selection.py` | Place-checkpoint selection on end-to-end success, re-scored on held-out seeds (§9.18). |

---

## 2. Directories

| Directory | Contents |
|---|---|
| `Results/` | 43 `.txt` evidence files, ~85 experiment subdirs of raw CSVs, `figures/`. See §4. `Results/big2m/` holds the `c2m512_s*` INT8 artifacts. |
| `checkpoints/` | 161 run directories. Naming convention in §3. Each holds `best/` and periodic saves. |
| `logs/` | 174 training logs, one per run, named to match its checkpoint family. |
| `qat_output*/` | Quantization artifacts per campaign: `_bi` (holds the deployed `place_orig_int8`; its `bi_s0` grasp is superseded), `_fix`, `_phaseB`, `_reset`, `_corrector`. Each holds `.tflite`, the `_float32.tflite` reference, and the refined `.pt`. |
| `Decomposed state training/` | Current training code + `Grasp_training_history.txt`. |
| `esp32_apriltag/` | Vendored detector (see §1). |
| `assets/` | Tag images, textures, and the deployed corrector `tag_residual_wrist32.json`. |
| `Important Texts/` | Reference material. |

**Deployed artifact** (confirmed by the firmware headers): grasp
`Results/big2m/c2m512_s1_int8.tflite` (source `checkpoints/td3_grasp_rand_td3_ln_c2m512_s1/best`)
+ place `qat_output_bi/place_orig_int8.tflite` (source `checkpoints/td3_place/best`, the
original fixed-spawn policy, never retrained) + FP32 AprilTag corrector
`assets/tag_residual_wrist32.json` (1,571 params). Superseded grasp:
`qat_output_fix/grasp_bi_noqat_int8.tflite` (`checkpoints/td3_grasp_rand_td3_ln_bi_s0/best`).
`.tflite` files in the repo **root** are stale and match neither deployed model
(`thesis_context.md` §9.4).

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
`resetA` (critic resets), `bi` (the previous shipped grasp, superseded), `liftcert_*` (lift
certification), `gripfix*`, `rv2` (reward v2), `align` (yaw alignment),
`qat8` / `wclip*` / `wide128` / `smallc64` / `bigc512` / `c2m*` (capacity and
quantization sweeps; **`c2m512_s1` is the deployed grasp**), `buf200k/500k/1000k` (buffer size), `long60k`,
`cereal_{base,align}[warm]` (per-object, cereal), `can_warm` (per-object, can).

`td3_place*` random-spawn investigation (§9.18): `_cereal_s*` (default recipe), `_cerealbig_s*` / `_cerealsmall_s*` / `_breadbig_s*` (recipe arms), `_pairfix_s*` (critics warm-started), `_curF_s*` / `_curR_s*` (controlled fixed vs random pair, with `snapshots/`), `_cerealscratch_s*` (cereal from scratch, with `snapshots/`), `_cerES_*` / `_cerBP_*` / `_cerBPann_*` (`scr`/`warm`, seeds 10–12: early stopping, potential-based built-in reward, annealed toward sparse; `Results/place_training_batches.txt`).

`td3_place*` variants: `_bi_s*` (paired with the superseded `bi_s0` grasp), `_rs_reset_*`
vs `_rs_control_*` (critic-reset ablation), `*_backup` (manual saves).

---

## 4. Results index — which file answers what

Section numbers are `thesis_context.md` sections.

| File | Question it settles |
|---|---|
| `phaseA_algorithm_comparison.txt`, `phaseA_eval100.txt` | TD3 vs SAC vs PPO on the grasp stage. |
| `phaseB_rotation.txt` | Full yaw randomisation; critic resets replicate; capability is specialised in both directions. |
| `reset_ablation.txt`, `quantization_resetpolicy.txt` | Critic resets / plasticity loss (§9.6). |
| `lift_certification.txt` | Lift-certifying the grasp criterion aligns the stage interface (§9.5). |
| `gripfix_and_selection.txt` | Grip-quality reward and the checkpoint-selection fix (§9.7). |
| `grip_robustness.txt` | Grip quality, not jerk, drops objects; the monolithic grip is better (§9.15.1). |
| `handoff_diagnostic.txt` (superseded by `monolithic_control.txt`); raw data in `Results/handoff_carry/` | The stage interface; grip pose predicts carry survival (§9.8). |
| `builtin_reward_results.txt`, `reward_v2_results.txt` | Reward shaping; robosuite's own shaped reward (§9.9, §9.10). |
| `int8_deployment.txt` | INT8 conversion of the superseded `bi_s0`; distillation-loss QAT hurt (§9.12, revised by §9.15.1). |
| `weight_range_regularisation.txt` | Weight-range clipping for INT8; removes catastrophic seeds (§9.15.1). |
| `qat_in_training.txt` | QAT inside the RL loop helps INT8; overturns §9.12's verdict (§9.15.1). |
| `capacity_experiments.txt` | Wider actor via net2wider: negative, confounded by a small critic (§9.15.1). |
| `validated_90_model.txt` | `smallc64_s5`, the first held-out INT8 model above 90% (§9.15.1). |
| `buffer_size.txt` | Replay buffer size: the largest INT8 lever; seed variance collapses (§9.15.1). |
| `critic_capacity_2m.txt` | Large critic given a 2M buffer: the recipe behind `c2m512_s1` (§9.15.1). |
| `fsm_fp32_validation.txt`, `fsm_rule_layer_sweep.txt` | The FSM replica matches the eval harness (§9.11); rule-layer tuning (§9.15). |
| `transport_stall_diagnosis.txt` | **Central.** Three failure causes; Fix A/C adopted, Fix B/D rejected. Part 3's "blocked class is unfixable from the rule layer" is correct *as scoped*, and carries a forward pointer — superseded in **disposition**, not analysis, by `orientation_anchor.txt` (§9.17). |
| `transport_retrain_negative.txt`, `wrist_alignment_negative.txt` | Settled negatives — do not retry. `KEEP_ROTATION=1` (−52 pts) is also settled, but do **not** read it as "rotation: closed" — see `thesis_context.md` §9.13. |
| `orientation_anchor.txt` | §9.17, **measured on the superseded `bi_s0`** (on `c2m512_s1` it hurts — see the next row). `a[3:6]=0` freezes `goal_ori` in robosuite's OSC → joint 5 saturates against its one-sided range; `ROT_ANCHOR_EPS=1e-6` recovers the blocked class. Also documents the cereal scoring artifact (physical vs scored placement) that confounds bread-vs-cereal comparisons. |
| `orientation_anchor_c2m512.txt`; raw data in `anchor_c2m512/` | §9.17.1. The anchor on the deployed artifact: baseline reproduced to the episode, prediction committed first, −2.75 FP32 / −2.50 INT8 on random spawn; the loss is in steeply tilted grasps. |
| `place_random_spawn_investigation.txt` | **Newest** (§9.18, final). Place training rises then degrades in every configuration — from scratch and warm-started, bread and cereal, fixed and random spawn — and final weights score far below `best/`. Not random spawn, not the object, not warm-starting: late-training instability is the open problem. Also not buffer, critic warm start, forgetting, freezing, drops or collision. |
| `curriculum_pair/` | Bread fixed vs random spawn from scratch; every snapshot scored end-to-end; `correlation_report.txt` = training metric vs end-to-end (§9.18). |
| `cereal_pairing.txt`; raw data in `cereal_pairing/` | Cereal, trained place policy vs scripted transport vs transfer, paired 12×100: scripted 89.17% vs trained 88.00%, not significant; the anchor helps this pipeline (§9.18, §10). |
| `place_training_batches.txt`; raw data in `place_reward_audit/`, `place_reward_audit2/`, `place_es/`, `place_es_B/`, `place_es_C/`, `place_batch_holdout/` | The three cereal place-training batches (early stopping; potential-based built-in reward; annealed toward sparse): design, the pricing gates, and the held-out result — none fixes the late decay. |
| `place_batch_holdout/` | Held-out scoring of all 18 batch runs at peak snapshot / `best/` / final weights, 8 unused seeds × 100, deduplicated by actor md5 (§9.18). |
| `place_selection/` | Place-checkpoint selection on end-to-end success, re-scored on 8 held-out seeds × 100: peak vs `best/` vs final weights vs reference (§9.18). |
| `cereal_scratch/` | Cereal from scratch vs warm-started cereal on transitions: `report.txt`; `eval.tsv` (4×50 per checkpoint), `protocol.tsv` (12×50), `control.tsv` (positive control) (§9.18). |
| `object_generalisation.txt` | Zero-shot to unseen objects fails, ordered by shape rather than size (§9.15.2). |
| `orientation_ablation.txt` | World-frame orientation input is nearly free; gripper-frame is essential (§9.15.2). |
| `yaw_alignment.txt` | The grasp policy does not align jaw yaw (§9.15.2). |
| `cereal_per_object.txt` | Cereal per-object grasp; its grasp metric is anti-correlated with task success (§9.15.2). |
| `transport_diagnosis.txt` | Cereal transport under the bread place actor; scripted waypoint transport (§9.15.2; pre-`orientation_anchor`). |
| `pruning_distillation.txt` | Parameter reduction vs accuracy; pruned initialisation beats random (§9.15.2). |
| `apriltag_perception.txt` | AprilTag end-to-end under the real FSM: latch at grasp; the learned residual corrector (§9.14.1, §9.14.2). |
| `corrector_on_device.txt` | The residual corrector must run FP32 on the MCU; INT8 destroys it (§9.14.2). |
| `wrist_camera_route.txt` | Perception via the wrist camera, one detection per episode, on a plain ESP32 (§9.14.3). |
| `tag16h5_deployment.txt`, `tag16h5_deployment_path.txt` | tag16h5 as the deployment tag family, and why (§9.14.3). |
| `on_device_perception.txt` | Detector and corrector running on the ESP32 itself, no PSRAM (§9.14.3). |
| `hil_hardware_validation.txt` | HIL with the deployed `c2m512_s1` INT8 under random spawn (§8.4, §9.15.1). |
| `int8_FSM_HIL_performance.txt` | The earlier 10-episode HIL run: fixed spawn, older model, raw terminal log (§8.4; superseded). |

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
* `.tflite` in repo root — stale, matches neither deployed model (§9.4).
* `PROJECT_CONTEXT.md` — June, monolithic era.
* Joint-5-gated anchor untested (§9.16).
* `ROT_ANCHOR_EPS` is deliberately not in the `.ino`: it hurts the deployed `c2m512_s1` (§9.17.1).
* Place training degrades late in every configuration; select place checkpoints on end-to-end evaluation of snapshots, not the training metric (§9.18).
* The anchor's effect is configuration-dependent: off for the deployed `c2m512_s1` on bread, on for the
  cereal place-actor pipeline. Pass `--rot-anchor-eps` explicitly (§9.17.1, §9.18).
* `train_place.py` 50-episode best-window (§9.7).
