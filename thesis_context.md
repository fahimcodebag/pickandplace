# Thesis Context — Deploying Decomposed Deep Reinforcement Learning for Robotic Pick-and-Place on Microcontrollers

> Context file for thesis text generation. Covers: system architecture, the
> complete training campaign (hurdles → diagnoses → solutions, with the key
> training-log evidence retained), final results, INT8 quantization-aware
> fine-tuning, hardware-in-the-loop (HIL) validation on ESP32, and the
> generalization to randomized object spawns, and the current deployed
> INT8 pipeline.
>
> **START HERE if resuming work: the headline below, §9.15.1 (the deployed artifact),
> §9.16 (what is open and the methodological rules), §9.17 (a controller-interface
> defect, measured only on the superseded `bi_s0`) and §9.18 (place-training
> instability).** Sections 7, 8.4 and 9.1–9.3 record
> earlier states of the project and are explicitly superseded where §9.4 onward
> says so.
>
> **For "where is X?" read `REPOSITORY_MAP.md`** — an index of every directory,
> script, checkpoint family and results file, with the checkpoint naming
> convention decoded. This file answers "why is it like this"; that one answers
> "where is it"; `Results/*.txt` answers "what is the number".
>
> Current headline — deployed artifact `c2m512_s1` grasp INT8 + `place_orig_int8` + FP32
> corrector `tag_residual_wrist32.json` (§9.15.1, §9.14.2):
> **random spawn INT8 95.33% / FP32 94.92%, fixed spawn INT8 99.83% / FP32 100.0%**
> (held-out, 12 seeds × 100). **Hardware: 97/100 on the ESP32 under random spawn.**
> **With AprilTag perception running on the ESP32 itself** (no PSRAM): **89.67%**
> against a 94% ground-truth ceiling (§9.14.3).
> These are anchor-off figures, which is also what the firmware runs. §9.17's `ROT_ANCHOR_EPS`
> helped the older `bi_s0` (INT8 78.33 → 83.58%, FP32 90.67 → 93.58%) but **costs `c2m512_s1`
> 2.50–2.75 points on random spawn** (§9.17.1): leave it off for the deployed artifact.
>
> Last updated: 2026-09-11

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

**Later hardware validation (supersedes the 10-episode figure above):** 97/100 under
random spawn with `c2m512_s1` INT8, on-device inference 0.26 ms (§9.15.1,
`Results/hil_hardware_validation.txt`); and full AprilTag perception on the board
itself, 18/20 episodes on its own pose (§9.14.3).

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

### 9.4 STATUS UPDATE — §9.1–9.3 above are superseded

Everything from §9.1 to §9.3 is preserved as an accurate record of the earlier
campaign, but its numbers and its conclusions have been overtaken. The
random-spawn ladder's "3% at native randomization" and the curriculum diagnosis
describe a pipeline that no longer exists. Read §9.4 onward as current.

> **Superseded (2026-09-11):** this table and the artifact and checkpoint paths below
> describe `bi_s0`. Current: grasp `Results/big2m/c2m512_s1_int8.tflite` (run
> `checkpoints/td3_grasp_rand_td3_ln_c2m512_s1`), the same transport, and the FP32
> corrector `assets/tag_residual_wrist32.json` (§9.14.2, §9.15.1). Headline at the top.

**Current headline (all at spawn level 2.0 = full position box + full
z-rotation, i.e. robosuite's native sampler):**

| Configuration | Fixed spawn | Random spawn |
|---|---|---|
| FP32, deployed FSM | **100.0%** (200 ep) | **89.6%** (1200 ep) |
| INT8, deployed FSM | **96.6%** (1200 ep) | **76.1%** (1200 ep) |

All random-spawn figures use the **tuned** rule layer (§9.15). Earlier drafts of
this file quoted 84.0% FP32 / 70.0% INT8 random from 4-to-7 eval seeds; those
were **low by 3-6 points** because per-seed sd is 3.3 points and the seed draw
was unlucky. See §9.15.
| *thesis §8.4 FP32 Python baseline* | *92%* | — |
| *earlier INT8 HIL on hardware* | *90% (10 ep)* | — |

**Deployed artifacts:** grasp `qat_output_fix/grasp_bi_noqat_int8.tflite`
(7.9 KB), transport `qat_output_bi/place_orig_int8.tflite` (7.9 KB). The repo's
`grasp_int8.tflite` / `place_int8.tflite` date from commit `cbd4535` (Aug 17)
and are **not** these; they were deliberately left in place.

**Source checkpoints:** grasp
`checkpoints/td3_grasp_rand_td3_ln_bi_s0/best` (built-in reward, seed 0);
transport `checkpoints/td3_place/best` — the ORIGINAL fixed-spawn policy,
never retrained.

### 9.5 The single largest win: the two stages disagreed on what a grasp is

Stage 1 certified a grasp as `_check_grasp()` true for 5 consecutive steps.
Stage 2's handoff required an 8-step hold **plus** a scripted 20-step lift with
≥3 cm rise. Measured: 86% of episodes satisfied the stage-1 criterion, only 43%
survived the stage-2 handoff. Stage 1 was rewarding fast snatches that failed
the moment the object left the table.

`GraspRewardWrapper.require_lift=True` makes the stage-1 criterion **identical**
to the stage-2 probe. End-to-end went **13% → 82%** with no change to the
transport policy, the FSM, or the architecture. See
`Results/lift_certification.txt`.

*Lesson for the thesis: in a decomposed system, the interface contract between
stages is a first-class design object. A mismatch there cost more than every
reward and algorithm intervention in this project combined.*

### 9.6 Plasticity loss is real, and stage-specific

Decay was established as **plasticity loss**, not task contradiction: PPO decayed
with no replay buffer at all (falsifying stale replay), and primacy-bias critic
resets (Nikishin et al.) recovered performance the controls never reached.
`td3_ln` is bit-reproducible (numpy-seeded exploration), which made the ablation
causally decisive.

| Stage | Critic resets |
|---|---|
| Grasp, level 1.0 | **+20.3** pts, 3/3 seeds, t=5.17 |
| Grasp, level 2.0 | **+30.0** pts, 3/3 seeds, t=4.91 |
| **Place/transport** | **−30.0** pts (mean 55.0 → 25.0), one seed collapsing to 1/100 |

The sign **flips** between stages. Transport does ~6.7 gradient steps per
episode against grasp's ~48, so after each reset it has far fewer updates to
recover. Plasticity-loss resets are **not** a general remedy for this pipeline.
Sizing a schedule in gradient steps does not transfer between stages with
different episode lengths — the interval was first copied verbatim (20k) and
fired ~1 reset instead of ~7. See `Results/reset_ablation.txt`,
`Results/transport_retrain_negative.txt`.

### 9.7 Checkpoint selection was silently costing points

`best` was the argmax over a **50-episode** success window. At p=0.75 that
window has SE ≈ 6.1 points, a run generates ~40 of them, and keeping the
maximum is max-of-noisy-estimates bias. Measured overestimates: **+27 / +8 / +4**
points for seeds 0/1/2.

Widening to a 200-episode window (`--best-window`, `--best-margin`) is a pure
**measurement** change — re-running the three seeds stopped at *identical*
episodes — and was worth **74.0% → 79.2%** mean certified grasp for **zero**
additional training. Seed spread collapsed from 23.3 points to 2.1: most of
what looked like seed variance lived in the selection rule, not the policies.

`train_place.py` still selects on a 50-episode window; that fix has **not** been
applied there.

### 9.8 Four failed interventions on grip angle, and the control that explained them

A handoff diagnostic (`handoff_diag.py`) measured object pose in the gripper
frame at handoff. Corner grips — fingers closing on a corner of the bread rather
than a flat face, `min(|rel_yaw|, 90−|rel_yaw|)` near 45° — correlated
monotonically with failure across three independent policies (92.6% → 70.2%
flat→corner for the then-best policy, p < 5e-3), at 24% prevalence.

Four interventions were then aimed at it, and **all four failed**:

1. **Alignment multiplier on the success bonus** — measurably *hurt*
   (`Results/wrist_alignment_negative.txt`).
2. **Graded success bonus** (`--reward-v2`) — inert on alignment.
3. **Dense per-step alignment penalty** (`--dense-align`) — the policy cut the
   penalty ~30% by *shortening its approach* (65 → 47 steps) while alignment
   fell in 2/3 seeds. Reward hacking.
4. Network capacity — falsified directly: a supervised probe showed the
   **existing 5.2k network computes grip alignment to 1.8° median error** from
   the raw 46-D observation (chance 22.5°). Wider nets buy 0.7° and plateau.

Two mechanisms were measured along the way and both stand as facts:

* **The critic is blind to alignment.** Terminal reward correlates **+0.85**
  with alignment; the critic's Q(s,π(s)) correlates **−0.004**, non-monotonic,
  across **six** independently trained critics. The actor's gradient is ∂Q/∂a,
  so no reward shaping could reach it.
* Object orientation *is* observable (`obs[3:7]`, `obs[10:14]`) and the wrist
  *is* controllable (73° per 20 steps), so neither observability nor authority
  was the constraint.

**Then the control that should have been run first.** The user's earlier
**monolithic** policy (`train_v7.py`/`train_v8.py`, 2.2 M actor params, 30 M
timesteps, checkpoints at `/home/fahim/Thesis_fahim/checkpoints/td3_v7`) was
evaluated under criteria identical to the decomposed stage:

| Model | Lift-certified | Corner prevalence | Corner-grip success |
|---|---|---|---|
| monolithic v7 (final) | **98.7% / 98.0%** (2 seeds) | 26.5% | **98.0%** |
| decomposed gripfix_s2 | 79.2% | 23.5% | 70.2% |

**The monolithic model does not align its wrist either — and does not need to.**
Same corner rate, but its corner grips succeed at 98%. **Grip angle was a
symptom of weak grips, not a cause.** `Results/handoff_diagnostic.txt` is
superseded on this point by `Results/monolithic_control.txt`.

*Lesson: a monotonic, highly significant correlation reproduced across three
policies was still not causal. The cheap control — an existing model that
solved the same sub-task — falsified it in one afternoon and would have saved
four training campaigns.*

### 9.9 The fix: robosuite's own shaped reward

`train_v7.py` used `reward_shaping=True`, i.e. robosuite's
`PickPlace.staged_rewards()`. Its structural difference from this project's
custom grasp reward is a **dense lift term**:

```python
r_lift = 0.35 + (1 - tanh(15 * z_dist)) * 0.15     # only while grasped
```

Once grasped, reward rises **continuously with object height, every step**. The
decomposed grasp stage had **no lift term at all** — it paid `W_GRASP = 10/step`
for merely being grasped, terminated at 8 holds, and certified the lift as a
*terminal binary*. "Grip it well enough to lift it" was a terminal scalar rather
than a per-step gradient, on a variable (height) the policy directly controls.

`--builtin-reward` swaps it in. Certified grasp over 800 deterministic episodes:

| Arm | Certified grasp |
|---|---|
| gripfix (after 4 reward interventions) | 79.2% |
| reward-v2 | 77.2% |
| **built-in** | **87.1%** (88.9 / 83.4 / 89.0) |

`bi_s2` vs `gripfix_s2`: **z = +5.63**. And the angle dependence **dissolved
without being targeted** — end-to-end flat→corner spread went 22.4 pts →
2.7 pts, matching the monolithic model's 2.0. Confirms §9.8's correction.

**A pricing trap was caught before training.** Straight built-in reward scored
"never gripped" at **105.2**, second-best of all outcomes: v7 ran a 500-step
horizon with *no early termination*, so lingering was free, whereas this stage
terminates on a certified grasp and so *forfeits* remaining income. Adding an
idle cost restored the ordering (margin 3.5×). See
`Results/builtin_reward_results.txt`.

### 9.10 Reward auditing as a method (transferable)

`reward_audit.py` rolls out a policy and reports **accumulated value per reward
term per outcome class**, using per-term accounting emitted from inside
`_grasp_reward` itself (so it cannot drift from the reward under test). Every
candidate change is **priced on recorded episodes before training**.

On the then-current policy it found:

* The success bonus was **min 150.0, max 150.0, sd 0.000** across 290 certified
  grasps while quality varied 8-fold (lift rise 38–318 mm, alignment 0.01–0.99).
  `corr(total reward, alignment) = +0.05`. **The reward was blind to grasp
  quality** — no gradient existed past the certification bar.
* **Flicker-farming was suppressed but not dead.** It scored 150.5, *second-best*
  of all outcomes. Net income +8.43 per held step made farming beat a clean
  success at **26 held steps of 200**; the policy already reached 14 — a margin
  of 1.79×. The earlier `W_GRASP_SUCCESS` 20 → 150 override had only pushed
  break-even to 39, never closing it. Capping paid hold steps makes it
  **mathematically impossible**.
* Parking near the object netted **+54.1**: `P_IDLE` (−0.4) never covered
  `W_REACH` (up to +1.0).

`--reward-v2` implements all three. Predicted vs measured accumulated value
matched **to 0.1 on all five outcome classes**. It fixed flicker (17–22 per 50
episodes → 5–6) but left end-to-end unchanged — the alignment half was aimed at
the wrong variable (§9.8). `Results/reward_v2_results.txt`.

Per-term reward columns are now written to `logs/*/episodes.csv`
(`r_reach`, `r_grasp_hold`, `r_success_bonus`, …, `n_drops`, `grip_align`,
`lift_rise`), so a reward regression can be attributed to a *term* rather than
merely observed in the score.

### 9.11 Validating the FSM that actually ships

Every end-to-end number before this point came from `test_place.py` /
`place_env_wrapper.py` — the **training/eval wrapper**, not
`pick_and_place_INT8_FSM.ino`. The two are not equivalent: the sketch has **no
transport-stall early termination** (the wrapper kills a carry after 50 steps
without new-best progress; the sketch allows the full 300), and `GRASP_CAP` is
250 vs 200.

`fsm_sim.py` is a host-side replica of the sketch — all 9 phases and constants
transcribed verbatim, plus `hil_main.py`'s respawn-and-retry loop — with only
the serial round-trip removed.

| Configuration | Success | 1st-try handoff |
|---|---|---|
| bi_s0 + original transport, **fixed** | **200/200 = 100.0%** | 100% |
| bi_s0 + original transport, **random** | 336/400 = 84.0%<sup>†</sup> | 98% |
| gripfix_s2 + original transport, random | 154/200 = 77.0% | 92% |

<sup>†</sup> This 4-seed estimate was later corrected to **87.3%** over 1200
episodes with the same configuration; see §9.15.

The harnesses **agree** (84.0% vs the wrapper's 86.0%, z = −0.79 ns), so the
86.0% was not an eval-wrapper artifact. The FSM also **discriminates more
sharply**: bi_s0 beats gripfix_s2 by 7.0 points through the sketch vs 2.2
through the wrapper. `Results/fsm_fp32_validation.txt`.

### 9.12 INT8: the QAT step was the damage

| Configuration | Fixed | Random |
|---|---|---|
| FP32 chain | 100.0% | 87.3% |
| **INT8 chain (no QAT)** | **96.6%** | **76.1%** |
| INT8 chain, 50-epoch QAT (prior path) | — | 50.5% |

All figures in this subsection use 4–7 eval seeds and are therefore mutually
comparable but **low in absolute terms by 3–6 points**; see §9.15 for the
corrected headline values. The *rankings* below are unaffected.

Isolating the grasp model (INT8 grasp + FP32 transport, 420 episodes each):

| Variant | Success | corr to FP32 |
|---|---|---|
| gripfix_s2, **no QAT** | **73.3%** | 0.898 |
| bi_s0, **no QAT** | **70.2%** | 0.872 |
| bi_s0, 50-epoch QAT | 57.5% | 0.939 |
| bi_s0, 600-epoch QAT | 51.7% | 0.919 |
| gripfix_s2, 50-epoch QAT | **37.1%** | **0.952** |

**Output fidelity ranks the variants backwards.** The best-correlated model is
the worst-behaving. Following the diff metric would have shipped the 37.1%
artifact. §7's QAT-fine-tuning recipe should be treated as **superseded**:
skip QAT and PTQ the original weights.

Longer QAT is not the fix — 12× epochs at 3× LR moved refined-vs-teacher MSE
from 0.275990 to 0.276077, a floor set by per-tensor fake-quantization error.

**Mechanism.** Per-tensor INT8 uses one scale per tensor, set by the largest
weight:

| Model | layer | max/std | effective bits, typical weight |
|---|---|---|---|
| bi_s0 | fc1 | 10.0 | 3.67 |
| gripfix_s2 | fc1 | 9.3 | 3.78 |
| **transport** | fc1 | **5.7** | **4.48** |

Transport quantizes **for free** — FP32 grasp + INT8 transport scored 85.5% vs
84.0% all-FP32 — and the grasp model alone cost 26.5 points on the old path.

**The better FP32 policy is not automatically the better deployed one.**
`gripfix_s2` loses half as much to quantization (−7.2 vs −14.0) and **ties**
bi_s0 on random spawn (69.8% vs 70.0%, z = −0.08) despite a 7-point FP32
deficit. bi_s0 is kept for its fixed-spawn margin (95.2% vs 83.3%).

**Calibration matters.** The transport model has no replay buffer; calibrating
it on the *grasp* buffer would set activation ranges from a distribution it
never sees. It was calibrated on **9,132 real TRANSPORT-phase states** collected
by instrumenting `fsm_sim.py`. `Results/int8_deployment.txt`.

### 9.13 Two settled negative results — do not retry

* **Retraining transport against a new grasp distribution.** Attempted twice,
  six seeds total, both times losing to the untouched fixed-spawn original
  (best retrained 75% vs 82%; and 83.5 / 79.5 / 52.0 vs 86.0). The second
  attempt made `transport_stall` — the failure it targeted — **worse**
  (144 / 57 / 44 per 400 vs 30). Treat as a property of this pipeline.
* **Critic resets on the place stage** (§9.6).

**A trap in the second of these — read before concluding "rotation: settled".**
`KEEP_ROTATION=1` is rejected at **−52.33 points** because it passes the place
policy's *large* rotational outputs through, and the rule layer is what makes
that channel survivable. That result stands. It says nothing about writing a
*numerically negligible* value into `a[3:6]`, which is a different intervention
with the opposite sign (**+24.9 / +2.92**, §9.17). The two are easy to conflate
and the conflation is expensive: it is why the joint-limit stall class sat
"unfixable" for a full campaign.

### 9.14 Perception sweep — measured, and the input to the AprilTag work

`sweep_perception.py` degrades the object-pose channel and re-runs the pipeline.
At **10 mm** position noise, spawn level 2.0:

| Update period | `recompute` | `frozen` |
|---|---|---|
| 1 | 76.7% | — |
| 2 | 73.3% | 73.3% |
| 5 | 73.3% | 56.7% |
| 10 | 86.7% | 26.7% |
| 20 | **80.0%** | **6.7%** |

**`recompute` is flat out to period 20; `frozen` collapses.** The distinction is
that `recompute` re-derives `obs[7:14]` (object pose relative to the
end-effector) from *proprioception* between detections, holding only the world
pose stale — a zero-order hold on the tag, not on the observation. That drops
the required ESP32 perception duty cycle from 20% to **5%**.

*This is the key enabling result for AprilTag: detection can run at ~1 Hz
against a 20 Hz control loop provided the relative pose is recomputed from
proprioception each step.*

An earlier version of this sweep routed `--spawn random` through robosuite's
native sampler while scoring a level-1.0 policy, reporting 6.7% for a policy
that actually scored 93.3%. Spawn-condition mismatches between training and
evaluation have been the single most common source of wrong numbers in this
project.

### 9.14.1 AprilTag end-to-end — the camera goes blind at grasp, so latch

`Results/apriltag_perception.txt`. Real detector (tag injected into the scene,
OpenCV AprilTag + PnP) under the real FSM.

**Detection dies at grasp in every camera configuration** — GRASP 65% (agentview)
/ 29% (wrist), then **0%** through TEST_LIFT and TRANSPORT: the gripper occludes
the tag. That is geometry, not resolution. It is survivable because a held
object's pose *relative to the gripper* is constant and the gripper pose is exact
from joint encoders: `mode=latch` latches object→gripper at grasp and propagates
it with forward kinematics. Recomputing a world pose after grasp is actively
wrong — it holds a stale pose while the arm carries the object away.

**Three implementation bugs were worth ~40 points** (latch 28% → 68%):
resolution (the tag renders below AprilTag's decode threshold at 320/640 px from
agentview); planar pose ambiguity (IPPE_SQUARE returned the flipped solution;
fixed with `solvePnPGeneric` plus an upright-normal prior, orientation p95
91.3° → 4.7°); and a tag-vs-object-centre offset, which decomposed as fixed in
the *world* frame — a camera-extrinsics artifact, corrected there.

**Matched cost of real perception: −15 points** (truth 94.0% vs latch 79.0%,
t=−8.66, 4/4 seeds). An earlier "100% truth ceiling" was 8 episodes.

**Three negatives, one principle.** Detecting 5× more often changed nothing
(68% vs 68%); latching the median of recent detections cost −2.5; a second
camera cost −3.5 (averaged) and −7.5 (selected by reprojection error). Before the
grasp the object is stationary, so staleness is nearly free while noise is not:
**take the single most accurate measurement and hold it** — do not fuse, refresh
or average. Also closed: tag size (already overhangs the object), resolution
(past returns at 1280×960), and camera motion (provably static).

### 9.14.2 The learned residual corrector (MLP) — 77% of the gap closed

`Results/apriltag_perception.txt`, `Results/corrector_on_device.txt`.

**Framing.** Predict the *residual* (true − detected position) from features
available at runtime, not the absolute pose — regressing to truth lets a model
score well by echoing an already ~90%-right input. The fixed calibration offset
is the zero-feature special case. Data: 2,181 detections over 6 seeds, logged
under the real FSM. Two rounds of leakage had to be removed first: features
derived from the simulator's ground-truth pose inflated the fit from +39% to a
spurious +60%. In simulation the answer key sits beside the inputs.

| Corrector | Pose error (median) | End-to-end (12 seeds × 50) |
|---|---|---|
| none | 11.56 mm | 75.2% |
| ridge (linear) | 7.08 mm | 81.8% (+6.7, p=0.0037) |
| MLP 64-64 | 2.30 mm | 86.8% (+11.7, 12/0 seeds) |
| **MLP, tuned** | **0.69 mm** held-out | **89.7%** vs 94.0% ceiling |

The residual is **nonlinear** — an earlier "it is linear, no MLP needed" was
retracted once an MLP was actually fitted. Tuning used a seed split (tune on
four, hold out two) so the search could not leak into the result; **batch size
dominated architecture**. The size curve is flat at those hyperparameters: a
**1,571-parameter** net reaches 92% of the error reduction, smaller than the
policy it serves.

**A unit bug scored 0.0% on all 12 seeds**: the export emitted millimetres into
a position in metres. Fixed by rescaling the final layer and guarded by a
magnitude-ratio check (0.999).

**Composes benignly with INT8:** INT8 + AprilTag + corrector **89.50%** vs FP32
89.7% (p=0.93). Perception costs 5.83 points; quantisation costs 0.

**It must run FP32.** Per-tensor INT8 put 6.4–13.2 mm of error on a 11.56 mm
residual, because the 12 input features span a 2,198× range of scales and
per-tensor INT8 sets one scale for the whole input. FP32 costs 6–20 KB on a
board with an FPU and ~192× timing headroom, so the sketch runs a hand-rolled
MLP with no third interpreter. INT8 was necessary for the policies and
unnecessary for the corrector — "quantise everything" is the wrong default.

**One detection per episode is enough** (88.0% vs 86.5% at period 5; 7 → 1
detections per episode), which collapses the on-device *timing* requirement.

*Per-setup calibration:* it encodes this camera in this scene and must be refit
against real ground truth before it means anything on hardware.

### 9.14.3 Full perception on a plain ESP32 — no PSRAM

`Results/wrist_camera_route.txt`, `Results/tag16h5_deployment.txt`,
`Results/tag16h5_deployment_path.txt`, `Results/on_device_perception.txt`.

**Tag pixels, not resolution, are binding.** Agentview needs 1280×960 (1,200 KB
frame); the wrist camera sits ~10× closer and resolves a ~20 px tag at 320×240
(75 KB). Detecting once at t=0 with the arm at home makes the camera pose fixed,
so the residual is one calibration map: 20.6 → 0.39 mm (98%), end-to-end 86.8%
vs agentview's 89.7% (not significant). **tag16h5** is used because the ESP32
port ships its code table; at n=3,200 it detects equivalently to 36h11 (68.3% vs
66.9%) — an earlier +23-point spot check at n=30 was noise.

**The C detector port beats OpenCV**: its raw error is a near-constant 20.5 mm
offset, so the corrector reaches 0.06 mm (vs 0.39 mm). Deployment configuration:
ROI 180×160, tag16h5, `bits_corrected=0`, `AT_MIN_REGION=60`, zarray growth
1.25×, 4-byte union-find, two inherited memory leaks fixed — **89.67%** against a
94% ceiling, indistinguishable from the same detector with unlimited memory.

**Memory lessons.** The binding pool is `MALLOC_CAP_8BIT` (~211 KB), not
`getFreeHeap()` (281 KB, which includes 32-bit-only IRAM); two predicted fits
against the latter were wrong. Every panic (`0x1c`/`0x1d`) was out-of-memory —
upstream AprilTag never checks a malloc — and a host stack measurement had
overstated the need. A silent fallback to the PC's ground-truth pose once scored
19/20 with zero detections; the HIL harness now reports perception outcomes.

**On hardware:** peak heap 176.9 KB of 213.7 KB, detection 192–576 ms, 0 crashes,
no heap drift; HIL 18/20 episodes on the board's own pose. Higher resolution does
not help (640×480 does not fit and detects no better); 160×120 cannot buy back
error correction. **Deployed corrector:** FP32, 1,571 parameters
(`tag_residual_wrist32.json`, 6.1 KB).

### 9.15 Rule-layer sweep, and a 3-point measurement correction

**The sweep.** Sec 5's precedent (FSM parameters alone worth 78% -> 92% at fixed
spawn) motivated re-tuning the rule layer for random spawn, where the grasp
stage now reaches transport in **400/400** episodes and every remaining failure
is downstream. `fsm_sim.py` gained CLI overrides for all 14 rule-layer
constants; a 25-config coordinate sweep at 100 episodes each screened them.

Only one parameter showed a coherent trend — `TRANSLATE_SCALE`, the multiplier
on the transport policy's commanded translation:

| 0.4 | 0.5 (base) | 0.6 | 0.7 | 0.8 | 1.0 |
|---|---|---|---|---|---|
| 80% | 82% | 86% | 86% | 85% | 83% |

`PLACE_HORIZON` did nothing at 200, 400 or 500, which is itself informative:
the failing carries are **stuck, not slow**, so more time cannot help them.
`NEAR_TARGET_XY` 0.10 cost 11 points (release trigger too tight to fire).

Confirmed at 1200 episodes per arm:

| Rule layer | Success |
|---|---|
| baseline | 1048/1200 = **87.33%** |
| tuned (`TRANSLATE_SCALE` 0.65, `CARRY_GAIN` 6.0, `NEAR_TARGET_XY` 0.18) | 1075/1200 = **89.58%** |
| difference | **+2.25 pts, z = +1.73, not significant** |

Real but small, and **no Sec 5-scale win** — those parameters were already close
to right for this regime. Taken anyway (free, and all five stage-2 candidates
beat baseline), but it should not be claimed as significant.

**The correction, which matters more.** The *baseline* — unchanged
configuration — scored **87.33%** over 1200 episodes against the **84.0%** this
file previously reported from 400. Per-seed results for the identical config:

```
seed    7  31  47  89 101 123 211 307 401 503 555 2024
      82  90  85  89  88  88  88  91  90  91  82   84
```

Per-seed sd is **3.3 points**, so a 4-seed mean carries **SE 1.6** — and the
original draw (7, 123, 555, 2024) happened to take three of the four lowest
seeds in the set.

*This is the 100-episode lesson one level up.* Moving from 100 to 400 episodes
fixed within-seed noise but kept only **4 eval seeds**, and at that scale
**between-seed variance dominates**. The rule is: **>= 12 eval seeds**, not
merely >= 400 episodes. Every random-spawn figure in this file was re-measured
at 12 seeds x 100 episodes as a result.

Corrected figures (tuned rule layer, 1200 episodes each):

| | Fixed spawn | Random spawn |
|---|---|---|
| FP32 | 100.0% (200 ep, baseline layer) | **89.58%** |
| INT8 | **96.58%** | **76.08%** |
| *INT8 cost* | *~-3.4* | *-13.50* |

The INT8 penalty on random spawn is real and large (-13.5 pts); on fixed spawn
it is small (-3.4). The mechanism is in §9.12 — per-tensor quantization of the
grasp actor, whose weight dynamic range is the widest of the three models.

### 9.15.1 The INT8 campaign — `c2m512_s1` replaces `bi_s0` (08-30 → 09-02)

The quantisation gap on random spawn (FP32 90.67% vs INT8 78.33% for `bi_s0`)
was closed and then inverted, one measured lever at a time:

| Lever | INT8 effect | File |
|---|---|---|
| Weight-range clipping, \|w\| ≤ 8·std | **+7.25** (p=0.0002), removes catastrophic seeds | `weight_range_regularisation.txt` |
| QAT *inside the RL loop* (fake-quant, STE, optimises return) | **+4.43**; FP32 unchanged | `qat_in_training.txt` |
| Wider actor via net2wider | −5.93, confounded (critic left small) | `capacity_experiments.txt` |
| → `smallc64_s5` | INT8 93.08% / FP32 91.67% held-out | `validated_90_model.txt` |
| Replay buffer 200k → 500k / 1M | **+15.17 / +14.00**; seed sd 17.18 → 2.84 | `buffer_size.txt` |
| Critic 512/256 given 2M buffer, batch 1024, 55k episodes | +3.12 INT8, +6.04 FP32 — free at deployment | `critic_capacity_2m.txt` |

The in-loop QAT result overturns §9.12's "QAT was the damage": that verdict was
about a distillation loss anti-correlated with deployed success, not about QAT.
The first large-critic test had been a budget artifact.

**Current artifact `c2m512_s1`** (held-out, 12 seeds × 100):

| | Fixed spawn | Random spawn |
|---|---|---|
| FP32 | 100.00% | 94.92% |
| INT8 | 99.83% | **95.33%** |

A strict improvement over `bi_s0` on every cell. **On hardware (09-04): 97/100
under random spawn** (95% CI 91.5–99.4), on-device inference **0.26 ms**, 0
communication errors over 14,996 cycles (`Results/hil_hardware_validation.txt`).

Also measured: jerk is not what drops objects; grip *quality* is, and the
monolithic v7 grip is 11–13 points better than the decomposed INT8 grip
(`Results/grip_robustness.txt`).

**Anchor:** §9.17's `ROT_ANCHOR_EPS` gains were measured on `bi_s0`. On `c2m512_s1` the anchor
costs 2.75 FP32 / 2.50 INT8 points on random spawn (§9.17.1); the figures above are anchor-off.

### 9.15.2 Beyond bread — generalisation, orientation, compression, cereal (09-06 → 09-08)

* **Zero-shot to unseen objects fails**, ordered by departure from bread's
  near-cubic shape, not size: can 71.67%, milk 44.00%, cereal 27.00% vs bread
  95.67% (`Results/object_generalisation.txt`).
* **World-frame orientation is nearly free, gripper-frame is essential**:
  `obj_quat` → identity costs −1.0; also blanking `obj_to_eef_quat` costs −84.3.
  The grasp policy does not align jaw yaw at all — error is uniform
  (`Results/orientation_ablation.txt`, `Results/yaw_alignment.txt`).
* **Pruning and distillation**: 48×24 keeps 90.67%, 32×16 83.33%, 24×12 66%;
  pruned initialisation beats random by up to 49 points, and mean action error
  is a poor proxy for task success (`Results/pruning_distillation.txt`).
* **Cereal per-object grasp is solved** (96.5–100% vs 27% zero-shot; warm start
  beats cold) — **but the grasp metric is anti-correlated with task success
  (r = −0.918)**: `--align-grip` improves the grasp and costs 15 points end to end
  (`Results/cereal_per_object.txt`). Stage-local metrics can fight the task.
* **Cereal transport**: the bread place actor operates 9σ outside its training
  distribution (`obj_to_eef_z` 0.057 vs 0.002); the high carry is load-bearing
  (every attempt to lower it hurt); a scripted waypoint transport went 31% → 60%
  after fixing three implementation bugs (`Results/transport_diagnosis.txt`),
  and later to 89.2% with §9.17's fixes.

### 9.16 Where things stand for a fresh session

**Read `REPOSITORY_MAP.md` first** for where anything lives — scripts,
checkpoint naming, which results file answers which question.

**Done and validated:** the deployed artifact — grasp `c2m512_s1` INT8, the original
place actor (`place_orig_int8`, never retrained) and an FP32 AprilTag corrector
(`tag_residual_wrist32.json`, 1,571 params). Held-out, 12 seeds × 100: random spawn INT8
**95.33%** / FP32 94.92%, fixed spawn INT8 99.83% / FP32 100.0% (§9.15.1). **HIL 97/100**
under random spawn on the ESP32 (§8.4, §9.15.1). AprilTag perception on the ESP32 itself:
**89.67%** vs a 94% ceiling (§9.14.1–9.14.3). Rule-layer tuning (§9.15). §9.17's
`ROT_ANCHOR_EPS` helped the superseded `bi_s0` and **costs `c2m512_s1` 2.50–2.75 points** on random
spawn (§9.17.1): leave it off.

**Open:**

1. **Decide `fsm_sim.py`'s `ROT_ANCHOR_EPS` default.** It is 1e-6, which understates the deployed
   `c2m512_s1` by 2.50–2.75 points on random spawn (§9.17.1); the place wrapper shares the constant.
   Until decided, pass `--rot-anchor-eps 0` for `c2m512_s1`. Untested: re-anchor only near joint 5's stop.
2. **Do not mirror `ROT_ANCHOR_EPS` to `pick_and_place_INT8_FSM.ino`** for `c2m512_s1` (§9.17.1):
   the firmware's `a[3:6] = 0` is the better configuration for the deployed artifact.
3. `train_place.py` still uses a 50-episode best-window (§9.7).
4. The `.tflite` files in the repo root are stale; they match neither deployed model (§9.4).
5. **Decide how cereal is scored.** The environment's z-window makes physical
   placement (99.7%) and scored success (89.7%) diverge by ~10 points for
   cereal alone, confounding every bread-vs-cereal comparison (§9.17). The
   no-retry cereal cells of §9.17 caveat 1 still need CLI re-runs before quoting.
6. `can` grasp is trained (best 0.995, seeds 0/1 to ~55k); `milk` is untrained.
7. **Random-spawn place training (§9.18)** — finished and evaluated. The open problem is
   late-training instability: every configuration rises then degrades, and final weights
   score 33–34 points below each run's peak on held-out seeds. End-to-end selection rescues
   1 run of 9 (§9.18). Also untested: robosuite built-in reward on the place stage;
   wrapper scripted phases still write `a[3:6] = 0`; training wrapper vs FSM release
   constants disagree.

**Methodological rules earned the hard way in this campaign:**

* **100-episode evaluations ran ~10 points optimistic four separate times**, and
  a 400-episode/4-seed protocol then ran **3 points pessimistic** because
  between-seed sd is 3.3 points (§9.15). The rule is **≥12 eval seeds × 100
  episodes**, not merely ≥400 episodes.
* **Behavioural evaluation is the only acceptance test for quantization.**
  Output diff was *anti*-correlated with deployed success (§9.12).
* **Price a reward change on recorded episodes before training it** (§9.10).
* **Run the cheap control before the expensive intervention** (§9.8).
* **Verify the harness matches the artifact that ships** (§9.11).
* Check `.tflite` **tensor dtypes**, never exit status — a prior conversion
  reported a perfect 0.00000 diff while silently remaining FP32.
* Evaluations are embarrassingly parallel; shard ~28-wide on this 32-core host.
  One process per (arm, seed) under `xargs -P 28`, one core each
  (`OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`). MuJoCo will not
  parallelise a single environment, so a sequential sweep sits at ~3% CPU.
* **When a failure class survives every rule-layer remedy, suspect the layer
  below.** Five hypotheses drawn from the failure distribution were refuted
  before the mechanism was found by reading the controller source (§9.17).
* **A finding recorded only in `Results/` gets re-derived.** The joint-5 stall
  was documented in `fsm_sim.py` *and* `transport_stall_diagnosis.txt` and was
  still rediscovered from scratch months later. It must be reachable from this
  section, or from `REPOSITORY_MAP.md`, to count as recorded.

### 9.17 The orientation anchor — a controller-interface defect

`Results/orientation_anchor.txt`. Supersedes the disposition (not the analysis)
of `Results/transport_stall_diagnosis.txt` Part 3. **Every main-pipeline cell in this
section uses the superseded `bi_s0` grasp; on the deployed `c2m512_s1` the anchor
costs 2.50–2.75 points (§9.17.1).**

**The mechanism.** robosuite's OSC updates the position goal unconditionally but
the orientation goal only when the rotational delta is nonzero
(`controllers/osc.py:259`):

```python
bools = [0.0 if math.isclose(e, 0.0) else 1.0 for e in scaled_delta[3:]]
if sum(bools) > 0.0 or set_ori is not None:
    self.goal_ori = set_goal_orientation(...)      # CONDITIONAL
self.goal_pos = set_goal_position(...)             # UNCONDITIONAL
```

So `a[3:6] = 0.0` — which TRANSPORT executes every step, on both the scripted
paths and the place-actor path — does **not** mean "no orientation command". It
freezes `goal_ori` at whatever absolute *world* attitude was last commanded,
back in GRASP, i.e. whatever arbitrary yaw the object happened to be picked at.
The orientation PD (kp=150) then fights to hold that attitude across the whole
0.65 m traverse from pick bin to place bin. Panda joint 5 absorbs the arc; its
range is one-sided, `[-0.02, 3.75]`, where every other joint is ±2.9 or wider.
It saturates.

| | stalled windows | successful windows |
|---|---|---|
| q5 | **3.753** (limit 3.75) | 3.541 |
| limit proximity (0 = at a stop) | 0.007 | 0.070 |
| motion achieved / commanded | 4.6% | 15.5% |
| windows moving <5 mm in 12 steps | 67% | 4% |
| contacts (arm or box vs bins) | 0.0% of 2,802 stalled steps¹ | — |

¹ *Correction and re-measurement (2026-09-11):* this row first read "none / none" from a check that matched geom names containing `bin` — but the only named bin geoms are visual legs, so collision was never actually measured. Re-measured with geoms classified by **parent body** (validated on a box resting on the bin floor): at `ROT_ANCHOR_EPS = 0`, 2,802 stalled steps were **100% joint-pinned with 0.0% arm/object–environment contact**; with the fix, stalled steps fell to 0 and success rose 16/40 → 33/40. The original conclusion holds — it is now measured rather than assumed.

Nothing in the loop bounds joint angles: OSC consumes Cartesian deltas, and
`nullspace_torques` regulates posture toward `initial_joint` but never reads
`jnt_range`. The only thing stopping a joint is MuJoCo's mechanical stop. The
arm jams silently, still holding the object, commanding full scale — and the FSM
reads the stall as convergence.

**The fix.** `ROT_ANCHOR_EPS = 1e-6` in `fsm_sim.py`, written into `a[3:6]`
instead of exactly 0.0. It scales to ~5e-7 rad and moves nothing; its only
effect is to flip the `isclose()` branch so `goal_ori` re-anchors each step.
`--rot-anchor-eps 0.0` reproduces every number recorded before this was found.

| cell (12 seeds × 100, paired) | before | after | Δ | t(11) | seeds |
|---|---|---|---|---|---|
| cereal + waypoint transport | 57.4% | 82.3% | **+24.92** | +24.3 | 12u/0d |
| ↳ + release radius 0.18→0.08 | 82.3% | **89.2%** | +6.83 | +5.30 | 12u/0d |
| **FP32** + place actor (**main pipeline**) | 90.67% | **93.58%** | **+2.92** | +5.12 | 11u/1d |
| **INT8** + place actor (**main pipeline**) | 78.33% | **83.58%** | **+5.25** | +8.19 | 12u/0d |
| bread + waypoint transport (CLI) | 93.58% | **95.50%** | +1.92 | +2.87 | 9u/2d/1t |
| cereal + waypoint transport (CLI) | 73.17%† | **89.17%** | +16.00 | +9.61 | 12u/0d |

Both main-pipeline baselines reproduce the recorded A+C figures **to the
episode** (1088/1200 and 940/1200), so both cells are directly quotable. Each
precision recovers ~90% of its own blocked class — 2.92/3.08 and 5.25/5.92 —
on classes differing by 1.9×. **Quantization cost on random spawn: 12.33 →
10.00.** († the cereal "before" column is the **bread** place policy transferred; no
cereal place policy existed then (§9.18), so that row is cost-matched, not scripted-vs-trained.)

**This is the class §9.15's campaign could not fix, and its attribution was
correct.** `transport_stall_diagnosis.txt` measured the blocked class at 3.08%
FP32, found joint 5 pinned in 76/88 cases, tried three open-loop escapes at 1200
episodes each and recovered **0** from every one. Its conclusion — *"no
open-loop rule-layer maneuver returns it to a workable configuration; removing
it needs joint-limit-aware control… neither of which lives in the FSM"* — was
right. The bread gain here is +2.92 against that 3.08% class: it collects ~95%
of exactly those failures. Not a new capability; the same class, addressed one
layer down. **Prevention, not escape.**

**A structural result worth keeping.** A 7-DOF arm against a 6-DOF task leaves
exactly **one** nullspace dimension — the elbow swivel. Joint 5 is wrist
flexion, which *determines* the task orientation, so `(I − J̄J)` projects out
precisely the component that would move it. This holds identically for posture
regulation, for gradient-projection joint-limit avoidance, and for CLIK's
`(I − J⁺J)q̇₀`. A PID integral does not help either: the steady-state error is a
*kinematic constraint*, not a disturbance, so integral action winds up and
saturates against a mechanical stop. The feedback path was never missing — OSC
is already a closed-loop task-space PD. **The lever is not more redundancy; it
is choosing which task dimensions to constrain.**

A preset joint configuration at staging — the intuitive fix — would have worked
*before* this. After it, joint angles at staging separate success from failure
at d′ ≤ 0.22 on all seven joints, with j5 now 0.23 rad clear of the stop. It
would standardise a variable that no longer predicts anything, at the cost of a
controller swap. Recorded so it is not re-proposed.

**Five refutations on the way** (all paired, at protocol or better): nullspace
posture gain (monotonically worse, 33→17→10%), joint-limit-aware nullspace
(null; the pinned-episode rate did not move), DESCEND touch margin
(bit-identical), DESCEND budget (−25.0, 0u/6d), post-release window
`RT_STEPS` 12→60→120 (bit-identical, twice). Every hypothesis drawn from the
failure *distribution* was wrong. The mechanism was found by reading `osc.py`.

**The cereal scoring artifact.** `pick_place.not_in_bin()` requires
`bin_z < obj_z < bin_z + 0.1`, i.e. `0.80 < z < 0.90`. **A cereal box lying flat
on the bin floor rests at exactly 0.90** — the exclusive upper bound. Of 62
scored failures, **60 come to rest inside the correct compartment**; release
accuracy (0.0120 vs 0.0124) and height (1.214 vs 1.211) are identical between
classes. What separates them is 1.4 cm of resting height at a boundary the
object cannot reliably clear. Physical placement is 598/600 = **99.7%** against
a scored 89.7%. Bread rests at ~0.825 and is unaffected, so **every
bread-vs-cereal comparison in this project is confounded by it.** This also
explains why `RT_STEPS` was inert twice: the box has settled *above* the window,
permanently — waiting cannot help. `fsm_sim.py --settle-steps N` now reports
both criteria; report both, do not substitute ours.

**Caveats — read before rewriting the headline table.**
1. **Resolved — and the cause was the harness, not the configuration.** The
   first FP32 pass read 88.7%. `fsm_sim.py:main()` wraps every episode in a
   **respawn-and-retry handoff loop** — reset until the FSM reaches TRANSPORT
   (up to `MAX_GRASP_ATTEMPTS`), score only after handoff, and record episodes
   that never hand off as `handoff_failed`. The custom worker used for that
   pass ran one episode straight through with no retry: strictly harsher, worth
   ~2 points. Through the real CLI the baseline reproduces exactly and the fix
   measures **+2.92**. The *delta* was robust to the protocol; the *absolute*
   was not — which is also why the INT8 grid, run through the CLI from the
   start, reproduced on the first attempt.
   **Consequence:** the cereal cells (57.4 / 82.3 / 89.2%) came from the
   no-retry worker. Their paired deltas stand, their absolutes do not; a CLI
   spot-check of the fixed configuration returned 19/20 = 95%. Re-run them
   through `fsm_sim.py` before quoting.
2. **INT8 measured, and the prediction held.** 78.33% → **83.58%**, +5.25,
   t(11)=+8.19, 12u/0d. Its baseline reproduced the recorded A+C figure
   *exactly* (940/1200), so unlike caveat 1 this cell **is** directly
   comparable to the headline table. The recovery fraction is identical on
   both precisions — +5.25 of a 5.92% class (89%) and +2.92 of a 3.08% class
   (95%) — which is strong evidence the mechanism is the same one on class
   sizes differing by 1.9×. INT8 random spawn now reads
   **76.08% (base) → 78.33% (A+C) → 83.58% (A+C + anchor)**.
   With the FP32 cell re-measured through the CLI (caveat 1), the `bi_s0`
   quantization cost restates as 12.33 → 10.00.
3. `NEAR_TARGET_XY 0.08` (+6.83) was measured on the **cereal scripted path
   only**. It is a shared rule-layer constant, so the default stays 0.18 and the
   scripted arm passes the flag.
4. Not mirrored to `pick_and_place_INT8_FSM.ino`; `check_fsm_sync.py` must gain
   `ROT_ANCHOR_EPS` when it is. The `isclose()` branch is robosuite-specific,
   but holding a fixed world attitude through a large arc is a physical error
   and a real Panda's joint 5 has the same one-sided range.

**Scripted transport beats the learned place policy, by less than first
reported.** Measured through the CLI at 12×100: bread **95.50% vs 93.58%**
(+1.92, t(11)=+2.87), cereal **89.17% vs 73.17%** (+16.00, t(11)=+9.61). The
first pass claimed +4.17 and +25.2 from the no-retry worker; the retry loop
lifts the *learned* arms far more than the scripted ones, because retries only
fire when the grasp fails to hand off — cereal's grasp is 96.5–100% and almost
never retries, bread's `bi_s0` is 87.1% certified and often does.
**On cereal the "learned" arm is the bread policy transferred** — no cereal
place policy had been trained when this was measured (§9.18) — so that row is cost-matched (zero training
either way), not evidence that a trained cereal policy would lose. Bears on §10.

### 9.17.1 The anchor on the deployed `c2m512_s1` — it hurts

`Results/orientation_anchor_c2m512.txt`. Same protocol as the recorded headline (held-out
seeds, 12 × 100, paired, through `fsm_sim.py`). The eps-0 baseline reproduced every recorded
cell with **0 episode mismatches**, and the prediction was committed before the treatment ran.

| cell | eps 0 | eps 1e-6 | Δ | t(11) | seeds |
|---|---|---|---|---|---|
| random FP32 | 94.92% | 92.17% | **−2.75** | −4.29 | 1u/10d/1t |
| random INT8 | 95.33% | 92.83% | **−2.50** | −3.74 | 0u/8d/4t |
| fixed FP32 / INT8 | 100.0 / 99.83% | 100.0 / 99.92% | 0 / +0.08 | — | ties |

**Prediction falsified.** A passive probe of `fsm_sim.main()` (positive control: `bi_s0` blocked
3.17% against the recorded 3.08%) sized `c2m512_s1`'s blocked class at 0.42% FP32 / 0.17% INT8, so
at most +0.4 was predicted. The anchor did recover that class (4 of 5, 1 of 2), but
released-not-placed failures rose 41 → 73.

**Mechanism.** With `a[3:6] = 0` the OSC keeps restoring the grasp-time attitude. The anchor is
written only in TRANSPORT, and there the goal follows the measured attitude, so attitude error is no
longer corrected. RECENTER, DESCEND, OPEN and RETRACT still write exact zeros (`FSM.step` starts every
action from `np.zeros`), so they freeze whatever attitude TRANSPORT ended with. Steep grasps drift
during the carry (misses: 6° → 13° over TRANSPORT, 75° → 82° at OPEN), that drifted attitude is held
through the release, and RECENTER releases them ~0.16 m off target.
The loss is entirely in grasps handed off at ≥ 60° from vertical (19% of episodes): failures
44 → 77; below 60°, 17 → 17. `bi_s0` pinned joint 5 in 3.2% of episodes and `c2m512_s1` in 0.4%, so
the trade that paid for `bi_s0` loses for `c2m512_s1`. (Tilt is read from the grip-site frame, which
equals the controller's `ee_ori_mat` and reads 9.8° at the initial pose.)

**Consequences.** The headline stands: it is the anchor-off configuration the firmware runs. Do not
mirror the anchor to the `.ino` for `c2m512_s1`. `fsm_sim.py` defaults to `ROT_ANCHOR_EPS = 1e-6`, so
pass `--rot-anchor-eps 0` for any `c2m512_s1` evaluation. Untested candidate: re-anchor only while
joint 5 is near its stop, which leaves every carry that never approaches the stop unchanged.

### 9.18 Random-spawn place training — what fails, and what does not

`Results/place_random_spawn_investigation.txt` (final); raw data in
`Results/curriculum_pair/` and `Results/cereal_scratch/`.

**A warm-started per-object cereal place policy loses to not training one.** Warm-started
from the bread place actor, 5 seeds: mean **67.77%** (sd 18.52; a `best/` figure — the
final weights score 39.9%) against
**73.33%** for the bread policy transferred and **89.17%** scripted. A first
evaluation that read `best/` while trainers overwrote it reported 1–16% for
three seeds; they were 36–72% once the trainers exited.

**Grasp-stage levers do not transfer to place — a second falsified transfer.**
Batch 1024 + critic 512/256 cost **−28.44** and collapsed seed variance
(sd 18.52 → 2.03) exactly as on grasp (17.18 → 2.84), onto a worse mean. Buffer
size alone was worth +4; a 2M buffer never fills at 8,000 episodes.

**Not the cause:** warm actor with cold critic (loading critics and skipping
warmup still collapsed 85% → 11%), curriculum forgetting, policy freezing, drops,
a kinematic block, or bin collision.

**What was established.** The curriculum never advanced past ~0.40 in any
collapsed warm-started run, while the original fixed-spawn run reached 0.96. With the policy
removed, fixed spawn scores **100%** at frac 0.2 and random spawn ~80% (bread),
37.5% (cereal): fixed spawn's first rungs are free, because `frac` is relative to
each episode's distance while the release radius is absolute. Collapsed policies
sit at that zero-action floor at every distance, commanding *harder* than the
working policy, and fail by self-induced distribution shift — correct on the old
policy's states, wrong on their own.

**Random spawn itself is not the problem.** The early collapsed runs were all
cereal, warm-started, or both. In the controlled bread pair —
from scratch, only spawn differs — both arms show the same advance → collapse →
recover-or-regress cycle at ~0.40 and no gap between them. Snapshots must be
compared on `time_step`, not episode: `learn()` is a no-op below 5,120
transitions, and fixed-spawn episodes are short enough that learning starts ~250
episodes later.

Smaller defects found: the wrapper's scripted carry still writes `a[3:6] = 0`
and pins joint 5 in ~10% of random-spawn carries; the training wrapper and the
deployed FSM disagree on release radius (0.10 vs 0.18), hold count (5 vs 3) and
translation scale (0.5 vs 0.65). robosuite's built-in reward has **never** been
tried on the place stage.

**Status 2026-09-11 evening (resume here).** Everything finished and was evaluated end-to-end;
nothing is training. Evidence: the final STATUS UPDATE of the results file,
`Results/curriculum_pair/correlation_report.txt`, `Results/cereal_scratch/report.txt` (positive
control: `cereal_s2/best` reproduces 85.33% exactly through the new evaluator).

*Curriculum pair* (bread, from scratch): the selection metric (frac50 × place50) tracks end-to-end
success at pooled Spearman **+0.59 fixed / +0.73 random**, but within a run only −0.18 … +0.68.
Latest-snapshot means **20.0% fixed / 15.7% random** — no between-arm gap.

*Cereal from scratch* (random spawn): `best/` at 12×50 scores **88.67%** (s2; paired against the
best warm cereal policy +3.33, t(11)=+1.64), 39.50% (s0) and 17.17% (s1); latest snapshots
27.5 / 0.0 / 20.5%. Warm-started final weights: default recipe **39.9%**
(70.0 / 73.5 / 15.5 / 2.5 / 38.0) against its 67.77% `best/`, and code-matched `pairfix` 3.0%.

**Every configuration rises then degrades.** `cerealscratch_s2` 88.0% at episode 2,250 (12×50) →
20.5% final; bread `curF_s0` 85.0% at 2,500–2,750 → 24.5%; warm starts decay from their
initialisation (`pairfix` 85.33% → 3.0%). From-scratch drops coincide with a curriculum advance to
~0.41; `pairfix` s1/s2 decayed with frac held at 0.20. A cereal policy trains from scratch to
88.67%, so neither the object nor warm-starting explains it: the open problem is **late-training
instability** of the place stage, which the rolling metric and `best/` only partly track.
Confound: `cereal_s0-4` launched before the wrapper gained `ROT_ANCHOR_EPS` on the policy path
(`0fc9a20`); `pairfix` and the scratch arm share code.

**Checkpoint selection on end-to-end success, tested out of sample** (`Results/place_selection/`).
Each run's peak snapshot on the 4 selection seeds, its `best/` and its final weights, re-scored on
8 held-out seeds × 100 with the same grasp and protocol (bread with the `bi_s0` grasp):

| mean over runs | peak | `best/` | final | reference, same seeds |
|---|---|---|---|---|
| bread (6 runs) | 51.67% | 32.96% | 17.62% | `td3_place/best` 93.25% |
| cereal (3 runs) | 46.83% | 48.79% | 14.04% | bread actor transferred 73.50% |

The peaks held out of sample (every held-out peak within 3 points of its selection value), so the
gap to the final weights is real: +34 points bread, +33 cereal. `best/` caught the peak on cereal
but missed it by 58 points on `curF_s0` and 32 on `curR_s1`. **Selection rescues a usable policy in
one run of nine:** `cerealscratch_s2` peak 87.25% (`best/` 89.00%) against 73.50% transfer, +13.75,
t(7)=+6.73, 8u/0d. The best bread peaks (86.12, 84.50%) stay significantly below the deployed place
actor (−7.12, −8.75). Four runs never exceed ~35% at any snapshot: selection cannot rescue a run that
never learned. End-to-end selection is necessary, not sufficient.

**Next (ask first):** training-side changes against the late collapse — early stopping on periodic
end-to-end evaluation, or a lower actor learning rate after the first peak; and the open items in §9.16.

---

## 10. Thesis Claim Status

| Claim | Status | Evidence |
|---|---|---|
| Decomposition makes each sub-problem learnable with tiny networks | **Demonstrated** | 87.1% certified grasp and 100.0% end-to-end (fixed spawn) with two 64→32 actors, 5.2k params each (§9.9, §9.11) |
| A deterministic rule layer converts a good policy into a reliable system | **Demonstrated** | 78% → 92% from FSM parameters alone (§5); FSM replica reproduces the eval harness within noise (§9.11) |
| Sub-policies fit and run on a commodity MCU in real time | **Demonstrated** | 15.8 KB total (2 × 7.9 KB), 9.49 ms/cycle vs 50 ms budget (§7–8) |
| Quantized deployment preserves task behavior | **Demonstrated** | `c2m512_s1`: INT8 95.33% vs FP32 94.92% random, 99.83% vs 100.0% fixed (held-out); clipping, in-loop QAT and a larger buffer/critic closed a 12-point gap (§9.15.1). HIL 97/100 on the ESP32 |
| Decomposition confines spawn randomness to the grasp stage | **Demonstrated** | Grasp stage absorbed the randomness (79.2% → 87.1%); the transport policy was **never retrained** and the two attempts to retrain it both lost (§9.9, §9.13) |
| A learned transport policy is not required for these tasks | **Demonstrated, modestly** | Scripted three-leg transport vs the learned place actor, both through `fsm_sim.py` at 12×100 with the superseded `bi_s0` grasp and `ROT_ANCHOR_EPS`: **bread 95.50% vs 93.58%, +1.92, t(11)=+2.87** (9u/2d/1t) — the fair test, since that actor was trained on bread. On **cereal** the learned arm is the bread policy *transferred* (a cereal place policy trained from scratch has since reached 88.67% at 12×50, §9.18, but is unpaired against scripted), so **89.17% vs 73.17%, +16.00** is a *cost-matched* claim — scripting and transfer both cost zero training — not scripted-vs-trained. The stage is replaceable; the bread margin is small |
| Failure classes that resist the rule layer are defects one layer down | **Demonstrated** | The 3.08% "blocked at a joint limit" class survived three open-loop escapes at 1200 episodes each (0/88 recovered) and was recovered by a one-constant change to the controller interface (§9.17) — on `bi_s0`. On `c2m512_s1`, whose class is 0.42%, the same change costs 2.75 points (§9.17.1) |
| The stage interface is the dominant design variable | **Demonstrated** | Aligning stage-1 certification with stage-2 handoff: 13% → 82% end-to-end, no architecture change (§9.5) |
| Plasticity-loss remedies transfer across stages | **Falsified** | Critic resets: +20 to +30 on grasp, −30 on transport (§9.6) |
| Reward shaping can direct fine-grained grasp geometry | **Falsified** | Four interventions failed; the critic never represents the axis (corr −0.004 across six critics) (§9.8) |
| Perception can run far below control rate | **Demonstrated** | One AprilTag detection per episode suffices when latched through the grasp — 88.0% vs 86.5% at period 5 (§9.14.1–9.14.2) |
| AprilTag perception on the MCU | **Demonstrated (sim + HIL)** | Detector + learned residual corrector on a plain ESP32, no PSRAM: 89.67% vs 94% ground-truth ceiling over 600 episodes; hardware 18/20 episodes on the board's own pose, 0 crashes (§9.14.3) |
| A learned residual corrector closes most of the perception gap | **Demonstrated** | MLP: 75.2% → 89.7% vs a 94.0% ceiling, 12/12 seeds; must run FP32 because INT8 error exceeds the residual (§9.14.2) |
| Grasp-stage hyperparameter levers transfer to the place stage | **Falsified** | Critic resets +20..+30 grasp, −30 transport (§9.6); batch 1024 + critic 512/256 −28.44 on place, collapsing variance onto a worse mean (§9.18) |
| Random spawn itself prevents place-policy training | **Refuted** | Controlled bread pair from scratch: no fixed/random gap (latest snapshots 20.0% vs 15.7%); cereal from scratch at random spawn reached 88.67% (`best/`, 12×50). Place training rises then degrades in every configuration — late-training instability is the open problem (§9.18) |

---
