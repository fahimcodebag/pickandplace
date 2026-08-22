# Thesis Context — Deploying Decomposed Deep Reinforcement Learning for Robotic Pick-and-Place on Microcontrollers

> Context file for thesis text generation. Covers: system architecture, the
> complete training campaign (hurdles → diagnoses → solutions, with the key
> training-log evidence retained), final results, INT8 quantization-aware
> fine-tuning, hardware-in-the-loop (HIL) validation on ESP32, and the
> in-progress generalization to randomized object spawns.
>
> Last updated: 2026-08-22 01:04 +06

---

## 1. Thesis Goal and System Overview

**Goal:** demonstrate that a robotic pick-and-place skill normally associated
with large deep-RL models can be delivered as **tiny per-stage neural policies
(64→32 MLPs, a few KB each) running on a bare ESP32 microcontroller**, by
decomposing the task into single-purpose learned sub-policies glued together
by a deterministic rule-based finite state machine (FSM).

- **Simulator:** robosuite `PickPlace` (MuJoCo), Panda 7-DOF arm, bread
  object, `single_object_mode=2`, OSC_POSE controller at 20 Hz.
- **Action space (7-D):** `[dx, dy, dz, droll, dpitch, dyaw, gripper]`;
  gripper +1 = close, −1 = open.
- **Observation:** 46-D flat vector (robot proprioception + object pose via
  robosuite `GymWrapper`). No cameras, no CNNs, no recurrence — full-state
  vector observations make the problem an MDP, so plain feed-forward MLPs
  suffice (a deliberate design choice enabling MCU deployment).
- **RL algorithm:** TD3 (twin critics, delayed actor updates, target policy
  smoothing) with Prioritized Experience Replay (SumTree PER); γ = 0.99;
  replay buffer 200,000 transitions; actor/critic LR 3e-4; batch 512;
  τ = 0.005.
- **Networks:** actor and critics 64→32 hidden units (ReLU, tanh output).
  Only actors are deployed; critics are training-only.
- **Object spawn:** Sections 2–8 use a **fixed** object spawn (position range
  patched to zero, rotation 0) — the same regime as the monolithic baseline,
  which keeps the comparison fair and isolates the learning and deployment
  questions. Generalization to randomized spawns is the current work
  (Section 9).
- **Training hardware constraint:** consumer laptop (WSL2 Ubuntu, 8 GB RAM,
  i5-1135G7, no discrete GPU), only 2 parallel simulation environments
  (subprocess-forked). Every experiment costs hours — this motivated a strict
  experimental methodology (Section 4).

### Decomposed architecture (the FSM as meta-controller)

| FSM state | Controller | Exit condition |
|---|---|---|
| 1. GRASP | learned grasp policy (64→32) | 8 consecutive grasp checks pass |
| 2. TEST-LIFT | scripted +z probe | object rises ≥ 3 cm and stays gripped; else retry GRASP (≤ 8 attempts) |
| 3. TRANSPORT | learned transport policy (64→32) | object held within 0.14 m of bin center for 3 consecutive steps |
| 4. RELEASE | scripted P-controller (recenter → lower → open → retract) | placement success check → DONE |

Theoretical framing: the architecture instantiates the **options framework**
(Sutton, Precup & Singh, 1999) with hand-defined options — learned
intra-option policies and fixed termination conditions — while replacing the
learned policy-over-options with a deterministic FSM. This is a deliberate
simplification for MCU deployment: the task order is known, and determinism,
verifiability, and minimal memory are more valuable than meta-level learning.
The closest classical ancestor is sequential composition of controllers
("funnels", Burridge, Rizzi & Koditschek, 1999); related: hierarchical RL and
skill chaining (Konidaris & Barto).

---

## 2. Stage 1 — Grasp Sub-Policy

The grasp stage was trained first with a **dense, staged reward**
(`grasp_env_wrapper.py`): reach shaping (distance-scaled), a gripper-closing
bonus when near the object, a per-step sustained-grasp reward, a one-time
stable-grasp bonus, and small penalties for idling, dropping, and retreating.
Episodes terminate early on 5 consecutive grasp-confirmed steps (success) or
at a 200-step horizon.

**Result: ≈ 95% grasp success**, reached with early stopping at 95% over the
last 100 episodes. Because the reward is dense, this stage trained without
curricula or special stabilization — an instructive contrast with Stage 2.

---

## 3. Stage 2 — Place/Transport Sub-Policy: Hurdles and Solutions

This stage consumed the bulk of the research effort and produced the thesis's
main methodological findings. Training wraps the environment so that each
episode begins by running the frozen grasp policy to a stable grasp
(`place_env_wrapper.py`), then hands off to the learning policy. Success =
robosuite's `_check_success()` (object inside the target bin). Diagnostic
instrumentation was added to the training logs and drove every decision:
per-episode **done-reason tallies** (success / timeout_held / timeout_dropped
/ fell_off_table / stalled_holding / grasp_handoff_failed / release_miss …),
**drop diagnostics** (`grasp_lost_step`, `max_lift_frac` medians), **handoff
attempt counts**, and **curriculum difficulty** statistics per 50-episode
window.

### Hurdle 1 — Reward farming ("hover exploit")

**Symptom:** with per-step state rewards (hover/lift/hold bonuses), the policy
learned to hover the grasped object above the bin indefinitely, collecting
~7/step forever instead of releasing.
**Solution:** replaced all per-step state rewards with **potential-based
shaping** `r = γΦ(s′) − Φ(s)` (policy-invariant; Ng, Harada & Russell, 1999).
Φ is a bounded cumulative-milestone potential over the *object* pose:
lift → approach (XY to bin) → place (3D to bin), weights 2/4/6, each milestone
gating the next. Lift credit uses a per-episode **ratchet** (max lift so far)
so that lowering the object into the bin cannot erase lift credit — necessary
because the bin floor (z = 0.80) is *below* the object's initial height.
Lingering now pays ≈ 0 by construction; shaping telescopes to
Φ(end) − Φ(start) regardless of path length.

### Hurdle 2 — Grip instability during motion (~95% of episodes dropped)

**Log evidence:** done-reason tally `timeout_dropped ≈ 36/50`; drop
diagnostics median `grasp_lost_step = 11–17` with real lift achieved —
i.e., the object was shaken loose mid-motion, not released by the gripper
action and not lost at handoff.
**Solution — three action shims** (scripted transport invariants, mirroring
the FSM design):
1. **Orientation freeze** (action dims 3:6 → 0): rotations shear the object
   out of a closed gripper; placement needs no reorientation.
2. **Gripper masked closed** during transport: exploration noise cannot pop
   the grasp open mid-carry.
3. **Translation gentling** (dims 0:3 × 0.5): OSC_POSE actions are per-step
   pose deltas; halving them caps commanded acceleration so real grasps
   survive inertial load. (Place-phase horizon raised 150 → 200 to
   compensate for slower transport.)
The masked action actually executed is stored in the replay buffer
(`info["applied_action"]`) to keep (s, a, r, s′) consistent.

### Hurdle 3 — Marginal handoff grasps

**Symptom:** even with gentling, a fraction of handoffs slipped in the first
seconds — the grasp model was rewarded for *achieving* grasps, never for
*holding them under motion*, so some handoffs were physically too weak.
**Solution:** a scripted **test-lift robustness filter** in `reset()`: after
a confirmed grasp, command a gentle straight-up lift for 20 steps; accept the
handoff only if the object rises ≥ 3 cm and stays gripped; otherwise retry
(≤ 8 attempts). **Log evidence after the fix:** drop diagnostics read
"no drops" for long stretches; handoff attempts averaged ~2.5–3 — proving
robust grasps existed and merely needed *selection*, so the grasp model did
not need retraining. First genuine placements appeared immediately after
(episodes ~155–170).

### Hurdle 4 — The "safe-hold" local optimum (sparse-reward exploration wall)

**Symptom (reproduced across 3+ runs):** the policy discovers placement
(score ~+150 episodes appear), then collapses to holding the lifted object
motionless until timeout — score pinned at ≈ −14 (idle bleed), tallies
dominated by `stalled_holding`/`timeout_held` 43–48/50. Reward rebalancing
(drop penalty −25 → −8; approach/place potentials 3→4/5→6) did **not** break
it. Root cause: ending the episode still gripping was a bounded,
zero-variance outcome; risk-averse TD3 collapses into it.
**Solution:** a one-time **terminal hold penalty** `W_TIMEOUT_HELD = −20`
applied when an episode ends still gripping — making never-releasing strictly
worse than a failed attempt, while keeping the penalty below the shaping gain
for approaching the bin (so the least-bad path is "carry to bin and release",
not "dump the object").

**Negative result worth reporting (penalty-shape A/B/C sweep):** a
distance-shaped variant (−8 floor + −18 × remaining-distance fraction) was
hypothesized to add a transport gradient. Outcome: the deep (−42 worst-case)
high-variance terminal target destabilized the critic — the policy
*thrashed* between dumping the object (drop diagnostics up to 17/50) and
safe-holding, and forgot mastered difficulty levels (89% → 14%). A softened
−8/−10 variant eliminated dumping but its weak floor removed at-bin release
pressure and safe-hold returned. **Conclusion: the flat penalty's uniform
release pressure is the working mechanism**; shaping the *penalty* trades one
failure mode for another. (The transport gradient belonged elsewhere — see
Hurdle 6.)

### Hurdle 5 — Success too sparse to bootstrap: reverse curriculum

**Symptom:** even with correct incentives, full-distance transport produced
1–5 successes per 50 episodes — too sparse to stabilize a value function.
**Solution:** a **reverse curriculum**: after the test-lift, a scripted
P-controller carries the object toward the bin until it is within
`frac × full_distance` of the target, then hands off. `frac` starts at 0.2
(success ≈ just release; frequent reward) and ratchets toward 1.0 (policy
does the whole transport). Each environment tracks its own success window —
no inter-process communication. Key refinements learned the hard way:
- **Advance only on sustained mastery** (≥ 75% over a full 40-episode
  window); an early lenient rule (50%/15 episodes) let a hot streak advance
  prematurely and strand a half-baked policy.
- **Reversible**: regress a level if the window falls below 15% — the
  curriculum self-corrects to the hardest level the policy can hold.
- **Finer steps in the hard region** (0.05 above frac 0.3, vs 0.1 below):
  0.1 jumps at higher difficulty caused overshoot failures
  (`fell_off_table` with the grasp never lost — the object driven off the
  table while still gripped).
- Raw environment horizon raised 500 → 700 so grasp + test-lift + carry +
  place fit in one episode.

### Hurdle 6 — The zero-gradient "desert" in the approach potential

**The single most predictive bug of the campaign.** The approach potential
used `xy_frac = max(0, 1 − d/0.25)` — **exactly zero beyond 25 cm from the
bin**, while full transport is ≈ 0.53 m. Any curriculum handoff beyond
frac ≈ 0.47 therefore started in a region with *no directional reward signal
at all*, crossable only by exploration noise — and inside a zero-gradient
region, holding still is genuinely locally optimal. This single fact
retro-predicted every observed wall: every run ground at frac 0.40–0.45
(inside the field) and collapsed into safe-hold at 0.50–0.65 (desert opens);
one run reached frac 1.0 transiently on noise-crossings alone and was
unstable there.
**Solution:** `_HOVER_SCALE` 0.25 → 0.60 — the approach gradient now covers
the entire transport with margin (slope ~6.7/m everywhere instead of 16/m
near-only). **Log evidence:** the next run ratcheted 0.2 → **0.85 at 84%**
within ~1,600 episodes (previous best-ever difficulty ≈ 0.65), later
0.96 at 88%.

### Hurdle 7 — Replay-composition pathologies at the difficulty frontier

Two related mechanisms were identified and separated experimentally:
- **PER failure-amplification:** during a dip, large-TD failure terminals are
  preferentially replayed, dragging transport values down globally →
  more failures → stronger priorities. Mitigation: PER priority exponent
  α 0.6 → 0.4. (Controlled comparison showed this alone did *not* remove
  collapses — documented as a falsified single-cause hypothesis.)
- **Failure-length data flooding (the decisive mechanism here):** a failed
  "held" episode contributes ~200 transitions; a success ~80. One bad
  50-episode window feeds ~9,000 "hold → small negative" transitions vs a
  few hundred success transitions (≈ 96% of intake), starving the critic of
  transport data regardless of sampling weights.
  **Solution:** **transport-stall early termination** — if the object sets no
  new best distance-to-bin (by ≥ 1 cm) for 50 consecutive gripped steps, the
  episode ends immediately (`transport_stall`, same penalty as timeout).
  Poison per failure drops ~4×; episode turnover during recovery
  quadruples. **Log evidence:** `transport_stall` replaced `timeout_held` in
  the tallies, and post-peak dips became contested recoveries rather than
  monotonic crashes.

### Hurdle 8 — The transport-vs-release conflict → architectural decomposition

**Insight:** one policy asked to both (a) carry the object a long distance
and (b) commit to a precise release has intrinsically conflicting objectives
— penalizing holding encourages dumping; penalizing dropping restores
safe-holding. After the penalty sweep (Hurdle 4) proved no scalar could
reconcile them, the conflict was **dissolved architecturally**:
- The learned policy became **transport-only**: translation control with the
  gripper scripted closed every step. It never opens the gripper.
- **Release became a scripted FSM routine** (`_scripted_lower_and_release`),
  triggered when the object has been held within the release radius for a
  hold count: **Phase 0** recenter (P-control XY to within 3 cm of bin
  center — the trigger admits up to 14 cm offset, and dropping from offset
  hits the bin rim); **Phase 1** descend (−z at 0.12/step to within 2 cm of
  the bin floor or until height stalls — a touchdown proxy, since there is
  no force sensing); **Phase 2** open (8 steps, no arm motion); **Phase 3**
  retract upward.
- The elegant part: **the safe-hold pathology inverts into the success
  condition** — "object held stably over the bin", formerly the dominant
  failure mode, is now the transport policy's goal state that triggers the
  scripted release. The failure mode became the stage boundary.
- On gentleness: with no tactile/force feedback, a *learned* release cannot
  outperform lower-then-open — "gentle" requires contact information neither
  has; the scripted descent-before-open already minimizes drop height.

**Implementation gotcha (documented for reproducibility):** robosuite's
`_check_success()` requires the gripper site to be ≥ ~4.2 cm away from the
object (`r_reach = 1 − tanh(10·d) < 0.6`) *in addition to* the object being
in the bin. A release routine that opens and immediately checks success
scores genuinely-placed objects as misses (~70% phantom-miss rate was
observed and traced to this). Hence Phase 3: retract the empty gripper and
accept the first success seen while clearing.

### Hurdle 9 — Checkpoint governance (an operational lesson)

Two bugs cost real peaks before being fixed:
1. `save_models()` wrote the same files on new-best **and** periodic
   checkpoints, so a later, degraded periodic save could clobber the best
   policy. Fix: on every new best, snapshot all model files into an
   untouchable `best/` subdirectory.
2. Under a curriculum, **single-episode score is meaningless as a "best"
   criterion** — a success scores ≈ +150 at *any* difficulty, so the "best"
   froze on an early easy-level episode while a far stronger policy (88% at
   frac 0.96) went unsaved. Fix: best-policy criterion =
   `mean(difficulty, 50) × mean(success, 50)` — a difficulty-weighted rolling
   success metric. A policy sets a record only by succeeding often *at
   difficulty*.

### Stage-2 outcome (training)

With all fixes active, training ratcheted to **frac 0.96 at 88% rolling
success (metric 0.849)** in ~1,370 episodes, with self-recovering dips —
compared to earlier configurations that plateaued at frac 0.4 or collapsed
at 0.65. Residual post-peak oscillation at maximum difficulty was identified
as TD3 convergence instability amplified by constant exploration noise
(training-time success *understates* the deterministic policy), which
motivated the evaluation-first protocol below.

---

## 4. Experimental Methodology (worth a thesis section of its own)

- **One variable per run.** Every training run changed exactly one thing, so
  each log was a clean signal. Falsified hypotheses (distance-shaped
  penalties, PER-α-as-single-cause, and others) are reported as negative
  results with the evidence that killed them.
- **Log-driven diagnosis.** Aggregate metrics (score, success%) were never
  trusted alone; the done-reason tally and drop diagnostics distinguished
  failure modes that identical scores would conflate (e.g., "never
  transported" vs "dropped mid-carry" vs "released and missed"), and each
  failure mode mapped to a different lever.
- **Buffer/checkpoint hygiene.** After any change to rewards or action
  semantics, replay buffer and models were wiped (stored transitions carry
  the old reward scale); for logging or curriculum-dynamics changes, training
  resumed. Resuming from a mid-collapse checkpoint was shown (twice) to
  poison recovery — fresh restarts with all accumulated fixes consistently
  outperformed resumed degraded runs.
- **Pre-committed decision criteria.** Before each run, the outcomes that
  would confirm or refute the change were written down, preventing
  post-hoc rationalization across expensive multi-hour experiments.

---

## 5. End-to-End Evaluation and the Rule-Layer Ablation

Deterministic evaluation (`test_place.py`: no exploration noise, full
pipeline grasp → test-lift → transport → scripted release, full transport
distance, 50 episodes):

| Configuration | Success | Notes |
|---|---|---|
| Best policy, default FSM params (trigger 0.10 m / hold 5 / horizon 200) | **78%** (39/50) | failures: 7 slow transports (horizon), 3 hovering just outside trigger, 1 drop |
| Same policy, tuned FSM params (trigger **0.14 m** / hold **3** / horizon **300**) | **92%** (46/50) | all 4 failures = slow-creeper horizon-outs; **zero drops, zero release misses**; successes ~58 steps, scores 146 ± 1 |

The 78% → 92% delta comes **entirely from rule-layer (FSM) parameters** —
the trained network is identical. This is a direct demonstration of the
architecture's thesis: the learned policy supplies robust transport
behavior; the deterministic rule layer turns it into a reliable system. The
widened trigger works *because* the scripted release recenters before
descending — division of labor between learning (gross transport) and
control (precise terminal placement).

**Failure-anatomy method:** failure causes were read directly from terminal
scores (e.g., −24 ≈ held penalty + idle over full horizon with partial
progress; −12 ≈ same but near-bin; −10 at 31 steps ≈ early drop), a technique
used throughout the campaign to classify failures without video inspection.

**Robustness to spawn randomization** was measured on this same stack and is
reported in Section 9.1: the system holds ~80% out to ±5 cm of spawn noise
and collapses to 3% under full robosuite randomization, with the failure
localized entirely to the grasp stage.

---

## 6. Distilled Design Lessons

1. **Potential-based shaping beats per-step state rewards** wherever an agent
   can linger (policy-invariance kills farming exploits by construction).
2. **Dense- vs sparse-reward stages need different machinery:** the grasp
   stage (dense staged reward) trained plainly; the place stage (sparse
   success) needed shaping + reverse curriculum + termination design.
   Curricula are a tool for sparse-reward exploration, not a default.
3. **Terminal penalties need uniform pressure, not shape:** shaping the
   *penalty* created new exploits; shaping the *potential* (the approach
   field) provided the gradient safely.
4. **Check the support of every shaping term:** a bounded potential that is
   zero outside its scale radius creates invisible zero-gradient regions
   where do-nothing is locally optimal. Match the field's range to the
   task's diameter.
5. **Watch replay composition, not just sampling weights:** with asymmetric
   episode lengths, failures can dominate buffer intake by sheer length;
   early termination of non-progressing episodes fixes what priority
   exponents cannot.
6. **Conflicting objectives are architecture problems, not reward problems:**
   the transport-vs-release tension resisted every scalar trade-off and
   dissolved instantly under decomposition (and turned the worst failure
   mode into a stage-transition condition).
7. **Best-model governance under curricula:** difficulty-weighted rolling
   metrics, write-protected best snapshots, and deterministic re-evaluation
   (training success under exploration noise understates the policy).
8. **Scripted micro-skills are first-class citizens:** the test-lift filter,
   the carry controller, and the release routine are each a few lines of
   P-control, cost nothing at deployment, and repeatedly outperformed
   attempts to learn the same function.
9. **Do not transplant machinery between stages without re-checking its
   premise:** the reverse curriculum that rescued the sparse-reward place
   stage became the *sole source* of pathology in the dense-reward grasp
   stage (Section 9.3). Match the mechanism to the reward structure, not to
   what worked last time.
10. **Deployment numerics can fail behaviorally while looking correct:**
    quantization error concentrated in unsaturated outputs broke fine
    gripper control while saturated outputs matched bit-for-bit
    (Section 8.3). Validate on a fixed input vector with PC-computed
    expected outputs, not on aggregate task success.
11. **Audit which predicates actually exist in the observation:** the FSM's
    transition conditions were MuJoCo contact queries absent from the 46-D
    state; every observation-only proxy failed (0–2 successes out of 8).
    Interface boundaries must be checked for *information*, not just for
    data format (Section 8.2).

---

## 7. INT8 Quantization-Aware Fine-Tuning (QAT)

Both deployed actors — the **grasp** policy and the **place/transport**
policy (each 46-D input → 64 → 32 → 7-D tanh output) — were **QAT
fine-tuned and converted to full-integer INT8** for MCU inference. Only
actors are quantized; critics never leave the training host.

**Working pipeline (`qat_and_convert.py` → `tflite_to_header.py`):**
1. **QAT fine-tuning** of the trained FP32 actor with fake-quantization,
   distilling against the FP32 teacher so the weights are pre-conditioned
   for integer arithmetic before conversion.
2. **Conversion** via `litert_torch.convert()` with TFLite converter flags
   for **full-integer INT8** weights and activations (float32 I/O at the
   tensor boundary), driven by a representative dataset.
3. **C-array generation** (`tflite_to_header.py`): the `.tflite` flatbuffer
   is embedded as a header (`grasp_model.h`, `place_model.h`) for
   compilation into the ESP32 firmware.

*(A second script, `qat_finetune.py`, implements the same idea through the
`ai_edge_torch` PT2E quantizer. It is retained for reference but was not
used for the deployed models — that quantizer was not installable on the
development machine. `qat_and_convert.py` is the path that produced every
deployed artifact.)*

**Calibration data — on-distribution states matter.** A quantizer calibrated
on the wrong state distribution mis-sizes its activation ranges, so each
stage is calibrated on states it actually visits: the grasp actor on
`demos_bread.npz` (grasp-phase states), and the transport actor on
`place_states.npz`, generated by `Decomposed state training/dump_place_states.py`,
which runs the **real** grasp → test-lift → transport pipeline and dumps the
states the transport policy encounters in deployment.

**Quantization mode — per-tensor, not per-channel (see §8.3).** Desktop
TFLite defaults to *per-channel* weight quantization; the ESP32's optimized
ESP-NN kernels compute it incorrectly. All deployed models are therefore
converted with `--per_tensor` (`_experimental_disable_per_channel`). The
fidelity cost is small and behaviorally irrelevant (grasp actor: output
correlation 0.994 → 0.986, mean absolute error 0.026 → 0.036; still 10/10
grasps in simulation), and on-device outputs then match desktop TFLite
exactly.

**Resulting footprint:**

| Artifact | Network | Size |
|---|---|---|
| `actor_float32.tflite` (monolithic baseline, `td3_builtin`) | 512 → 256 | 616 KB |
| `grasp_int8.tflite` | 64 → 32 | **8.4 KB** |
| `place_int8.tflite` | 64 → 32 | **8.4 KB** |

Both decomposed policies together occupy **16.8 KB** — roughly **37× smaller
than the single float32 baseline actor** — with a 40 KB tensor arena per
interpreter. Decomposition and quantization compound: smaller *per-stage*
networks, then 4× again from integer weights.

---

## 8. Hardware-in-the-Loop (HIL) Validation on ESP32

The final system was validated **hardware-in-the-loop**: the physics
simulation runs on the PC, while the **ESP32 runs both INT8 policies *and*
the entire finite state machine**. The MCU decides which policy or scripted
phase is active, performs integer inference, and returns the action; the PC
only steps physics. This exercises the deployed artifact — integer
arithmetic, memory behavior, control-flow, and serial latency — exactly as
it would run on real hardware.

### 8.1 Functional block architecture

```
┌─────────────────────────────  PC (WSL2)  ─────────────────────────────┐
│  hil_main.py (orchestrator + physics)                                 │
│    robosuite PickPlace env                                            │
│      ├─ 46-D observation ─────────────────┐                           │
│      ├─ _check_grasp / _check_success ────┤ → flags byte              │
│      └─ handoff retry loop (respawn ≤ 8×) │                           │
│         ▲                                 ▼                           │
│    env.step(action) ◄─ 7-D action ─ ESP32Bridge (esp32_bridge.py)     │
│                                       seq numbers, retries, stats     │
│                                       Protocol: framing + checksum    │
└──────────────────────────────┬────────────────────────────────────────┘
                               │  USB serial, 921,600 baud
┌──────────────────────────────▼────────────────────────────────────────┐
│  ESP32 firmware — pick_and_place_INT8_FSM.ino                         │
│    frame parser (sync 0xAA55AA55, header, CRC)                        │
│    ┌───────────────────────────────────────────────────────────────┐  │
│    │ FSM (on-device): GRASP → TEST_LIFT → TRANSPORT → RECENTER     │  │
│    │                  → DESCEND → OPEN → RETRACT → DONE_OK/FAIL    │  │
│    │   trigger 0.14 m │ hold 3 │ place horizon 300 │ bin hardcoded │  │
│    └───────────────────────────────────────────────────────────────┘  │
│    two TFLite-Micro interpreters, INT8 grasp + place actors           │
│    two 40 KB tensor arenas   →   action frame + phase status byte     │
└───────────────────────────────────────────────────────────────────────┘
```

**Protocol (v2).** Every message is
`[SYNC 0xAA55AA55][type:1][seq:1][len:2][payload][crc:2]` with a 16-bit
checksum; the decoder scans the stream for the sync pattern, so the link
self-resynchronizes after noise. The PC→MCU frame carries the 46 floats
**plus a flags byte**; the MCU→PC frame carries the 7 action floats **plus a
phase status byte**, which lets the orchestrator observe FSM transitions and
episode termination without inferring them.

### 8.2 What runs where — and why the split is where it is

Two design decisions define the PC/MCU boundary, and both were forced by
measurement rather than convenience:

- **Grasp/success predicates are supplied by the PC.** `_check_grasp()` and
  `_check_success()` are MuJoCo *contact* queries; neither is present in the
  46-D observation. Observation-only proxies were implemented and scored
  against the simulator: a pure gripper–object proximity test achieved
  **0/8** placements (it fires while the fingers are still closing, tripping
  TEST_LIFT early), and a finger-closure variant **1–2/8**, versus **93%**
  with the true predicates. The contact information is genuinely not
  recoverable from proprioception, so the PC — which owns physics — ships
  both predicates in the flags byte. On a real robot these correspond to
  gripper force/current sensing and a placement sensor: cheap, standard
  hardware, but *sensors*, not inference.
- **Handoff retries stay on the PC.** `place_env_wrapper.reset()` respawns
  the object and retries a failed grasp or test-lift up to 8 times before an
  episode is scored, and the 92% baseline is conditional on that loop. Only
  the simulator can respawn, so the MCU instead reports its phase and accepts
  a RESET frame to resync its FSM between attempts.
- **The bin position is hardcoded on the MCU** (`[0.1975, 0.1575, 0.80]`) —
  bins do not move, so this is a legitimate deployment constant.

### 8.3 The decisive deployment bug — per-channel INT8 kernels on ESP-NN

**Symptom:** on hardware, the arm reached the object but the gripper never
closed; `FLAG_GRASPED` never asserted, so the FSM never left GRASP. Model
files, header bytes, protocol, and PC-side flags all checked out.

**Signature (from the sketch's boot self-test):** action dimensions whose
true value is **saturated** (±0.992, −1.0) matched desktop TFLite exactly,
while **mid-range** (tanh-linear) dimensions were badly wrong — some
sign-flipped. Saturation hides numerical error; mid-range exposes it. Fine
control — holding the gripper open during approach — lives precisely in the
mid-range, which is why the failure looked like a *behavioral* bug rather
than a *numerical* one.

**Root cause:** desktop TFLite quantizes weights **per output channel** by
default; ESP32 builds route those kernels through ESP-NN's optimized integer
implementations, which compute per-channel scaling incorrectly. **Fix:**
convert with `--per_tensor` (§7). On-device output then matched desktop
bit-for-bit.

**Debug methodology (the transferable part).** Isolate layers outward from
the model until one fails: (1) INT8 model vs FP32 model on the PC → passed;
(2) generated header bytes vs the `.tflite` file (md5) → passed; (3) serial
protocol loopback (`test_protocol.py`) → passed; (4) PC-side flag emission
(`DEBUG_HANDOFF` in `hil_main.py`) → passed; (5) **on-device boot self-test
on a fixed input vector, compared against `selftest_expected.py` computed on
the PC → failed.** Everything but the last passed, which localized the fault
to on-device inference itself and excluded serial, FSM, simulator, and
export. `diag_esp32_actions.py` then diffed ESP32 actions against the same
INT8 model run locally on identical observations. **Lesson: a deployment
self-test on a fixed input, with PC-computed expected outputs, is the single
highest-value piece of embedded-ML debug tooling** — it cleanly separates
numerics from integration.

### 8.4 HIL results

Ten end-to-end episodes with the ESP32 as the sole source of control
actions and FSM state (`Results/int8_FSM_HIL_performance.txt`):

| Metric | Result |
|---|---|
| **Placements** | **9/10 (90%)** |
| Average score | 64.65 ± 21.57 |
| Mean handoff attempts | 1.00 (no retries needed) |
| Communication cycles | **1,839 / 1,839 successful (100%)** — 0 timeouts, 0 checksum errors |
| Mean round-trip cycle time | **9.49 ms** (inference + framing + serial) |
| Control period available | 50 ms (20 Hz) |

**Interpretation.**
- **90% on hardware vs 92% in the Python baseline** — the deployment loses
  essentially nothing. The single failure exhausted the 300-step transport
  horizon, which is the *same residual failure mode* as the simulation
  baseline (all 4 of its failures were slow-creeper horizon-outs), not a
  quantization or deployment artifact.
- **9.49 ms round-trip leaves ~5× headroom** inside the 20 Hz control
  period; the measured cycle rate (~105 Hz) is well above the control rate,
  and this figure includes serial transport, so pure on-device inference is
  faster still.
- **A perfect 1,839/1,839 communication record** validates the framed,
  checksummed, self-resynchronizing protocol at 921,600 baud.

This closes the thesis's deployment claim: the decomposed policies are not
merely small on paper — **both policies and the full control FSM run in real
time on a $5-class microcontroller**, in closed loop with physics, at the
same success rate as the desktop pipeline.

### 8.5 Operational note — host toolchain memory, not device memory

Builds failed with `cc1plus.exe: out of memory` — a **host compiler** RAM
starvation on the 8 GB development machine (it fails inside TFLite-Micro
library `.cpp` files, identically for one or two models, so shrinking the
network does not help). It is not an ESP32 or sketch issue. Mitigations:
`wsl --shutdown` before compiling (largest effect, since WSL2 holds host
RAM), a larger Windows page file (8–16 GB), capping WSL via `.wslconfig`
`memory=4GB`, and moving the sketch off cloud-synced folders. Recorded
because it consumed real project time and is a predictable hazard when
cross-compiling TFLite-Micro on a memory-constrained laptop.

---

## 9. Generalization to Randomized Object Spawns (current work)

Everything above — grasp training, place training, all evaluation, and the
HIL validation — was carried out with a **fixed object spawn** (position
range patched to zero, rotation 0). That was deliberate: it isolates the
learning and deployment questions. It also matches the monolithic baseline's
regime, which makes the head-to-head comparison fair — but it means the
decomposition's *strongest* claim is not yet demonstrated. The claim is:
**a decomposed system should absorb spawn randomness in the grasp stage**,
because the transport stage only ever sees "object in gripper, bin over
there", regardless of where the object started.

### 9.1 Spawn-randomization ladder (measured)

The deployed stack (best transport policy, locked FSM parameters) evaluated
under graded spawn noise via `test_place.py --spawn-range R` / `--random-spawn`:

| Spawn regime | End-to-end success | Dominant failure |
|---|---|---|
| Fixed (training pose) | **92%** | slow-creeper horizon-outs |
| Uniform ±2 cm | 77% | mixed |
| Uniform ±5 cm | 80% | drops rise to ~15% |
| Native robosuite (full bin box + rotation) | **3%** | **28/30 `grasp_handoff_failed`** |

**The hypothesis is supported and the bottleneck is identified.** Transport
degrades gracefully (~80% out to ±5 cm) and, on the one native-randomization
episode where the grasp stage delivered an object at all, placed it in 45
steps. **The grasp stage is the binding constraint** — it was trained on a
single pose and does not generalize to arbitrary positions and yaw. This is
the one place where retraining is justified by a *change in task scope*
rather than by patching a failure.

### 9.2 Random-spawn grasp retraining — five runs, four falsified hypotheses

`Decomposed state training/Random spawn model/` implements the retraining:
`grasp_spawn_wrapper.py` (spawn control via a patched placement initializer),
`train_grasp_rand.py` (separate `checkpoints/td3_grasp_rand`, warm-started
from the fixed-spawn actor), `test_grasp_rand.py` (deterministic eval). The
fixed-spawn grasp model and base wrapper are left untouched throughout.

The campaign began with a spawn **curriculum** (level 0.1 → 2.0: position box
scaling, then added z-rotation), transplanted from the successful place-stage
design. Every run reached ~level 0.9 and then decayed. The hypotheses, in
order, and what killed each:

1. **Reward farming at hard poses.** At difficult corner spawns, success
   *decoupled* from score: success fell 65% → 26–43% while scores held at
   85–100+, with failures scoring 180+. Cause: a per-step grasp reward
   (`W_GRASP=10`) combined with a *terminating* success bonus made a
   flickering marginal grip (grip, slip, regrasp, ×200 steps) out-earn a
   clean success, which stops the income. Invisible at fixed spawn, where
   stable grasps are trivially fast. **Fix:** `W_GRASP_SUCCESS` 20 → 150
   (override in the spawn wrapper only). Score/success re-coupled — success
   episodes now score ~250 — **but the decay recurred**, falsifying farming
   as the engine.
2. **Failure-length data flooding** (the mechanism that *was* decisive in the
   place stage, §3 Hurdle 7). Predicted fix: cap failure episodes,
   `GRASP_HORIZON` 200 → 120. Outcome: **worse** (peak 76% at level 0.80 vs
   86% at 0.83; earlier decay), and it likely clipped slow far-corner grasps
   that legitimately need the full horizon. **Reverted.** An important
   negative result: a mechanism verified in one stage was wrongly assumed to
   generalize to another.
3. **Buffer saturation.** Falsified by timing — one run decayed at 131k of
   200k transitions.
4. **Plateau duration (the surviving correlate).** Across all runs, decay
   onset tracked *time parked at one difficulty level* (~400–600 episodes),
   not incentives, not buffer fullness, not episode-length asymmetry. Model:
   a parked curriculum feeds self-similar data → TD3 overfits the narrow
   distribution → performance drifts. Loosening the advance gate (0.8 → 0.7)
   to keep the curriculum moving produced the *sickest* run yet: it raced
   through levels without consolidating, then decayed anyway at trivial ones.

### 9.3 Conclusion: the curriculum was the disease, not the cure

After four failed interventions, the meta-observation is that **every
grasp-stage pathology — farming, racing, parking, decay — was a pathology of
the curriculum loop itself**, and the curriculum had been inherited from the
place stage without re-examining its premise. **The place stage needed a
reverse curriculum because its reward was sparse. The grasp reward is dense**
(reach + grip + hold shaping guide the policy from *any* spawn), so nothing
about the grasp stage requires staged difficulty.

**Current configuration (`train_grasp_rand.py`): auto-curriculum removed.**
`SPAWN_LEVEL = 1.0` is a fixed module constant; every episode samples from
the full position box; the environment is constructed with `curriculum=False`.
There is no level to park at, race through, or tune — the data distribution
is uniformly diverse from the first episode. Early stopping is plain
convergence (≥ 85% success over 100 episodes). **Phase B**, set manually
after Phase A converges, raises `SPAWN_LEVEL` to 2.0 to add full z-rotation.

**Status:** Phase A is configured and pending; the best random-spawn artifact
so far is metric 0.355 (86% success at spawn level 0.83), preserved as
`checkpoints/td3_grasp_rand_best_0355_backup` and used to reseed the run.
Expected signature of a healthy run: success starts *low* (~40–60%, since the
full box is presented immediately) and climbs steadily **without** the decay.
If success still decays after peaking on uniformly diverse data, the
remaining explanation is plain TD3 overtraining / plasticity loss, and the
response is to harvest `best/` and stop long runs early.

Once Phase A/B converge, the plan is to point the place pipeline's grasp
checkpoint at `td3_grasp_rand/best`, re-run the end-to-end ladder, and
fine-tune the transport policy **only if** off-corridor starts measurably
degrade it (transport has so far seen only the single fixed spawn→bin
corridor, though its observation includes object coordinates and its shaping
field spans 0.6 m — see §3 Hurdle 6).

---

## 10. Thesis Claim Status

| Claim | Status | Evidence |
|---|---|---|
| Decomposition makes each sub-problem learnable with tiny networks | **Demonstrated** | 95% grasp, 88% transport at frac 0.96, both 64→32 |
| A deterministic rule layer converts a good policy into a reliable system | **Demonstrated** | 78% → 92% from FSM parameters alone, identical weights (§5) |
| Sub-policies fit and run on a commodity MCU in real time | **Demonstrated** | 16.8 KB total, 9.49 ms/cycle vs 50 ms budget (§7–8) |
| Quantized deployment preserves task behavior | **Demonstrated** | 90% on hardware vs 92% in simulation, same residual failure mode (§8.4) |
| Decomposition confines spawn randomness to the grasp stage | **Partially shown — open work** | Transport holds ~80% at ±5 cm; grasp is the measured bottleneck at full randomization (§9.1); random-spawn grasp retraining in progress (§9.2–9.3) |
