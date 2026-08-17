#!/usr/bin/env python3
# Last updated: 2026-07-04 20:53 +0600
"""
Place-only environment wrapper for decomposed RL training (Stage 2).

Wraps the GymWrapper (post-flattening) to provide:
  - Grasp rollout on reset: runs the trained grasp model until stable grasp,
    then hands off to the place policy
  - Lift + transport + place reward signal
  - Termination on _check_success(), object drop, or timeout

Architecture:
    raw_env → GymWrapper → PlaceGymWrapper
    (not raw_env → PlaceRewardWrapper → GymWrapper)

This avoids observation-flattening mismatches because the grasp agent
receives the same flat observations that GymWrapper produces.
"""

import os
import sys

import numpy as np
import torch as T

# Add parent directory for shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class PlaceGymWrapper:
    """
    Place sub-task wrapper that sits on top of GymWrapper.

    On reset(), runs the trained grasp policy until a stable grasp is achieved,
    then returns the post-grasp flat observation for the TRANSPORT policy.

    Transport/release decomposition: the learned policy only TRANSPORTS the
    grasped object over the bin (gripper scripted closed). When it has held the
    object stably over the bin (_RELEASE_TRIGGER_HOLD steps), a scripted
    P-controller lowers and releases it (_scripted_lower_and_release). This
    removes the transport-vs-release conflict that no single reward could solve.

    Reward = potential shaping (progress toward placed) + one-time events:
      - shaping:  gamma*Phi(s') - Phi(s), where Phi is a bounded cumulative
                  progress potential (lift -> approach -> place). See the
                  PHI_* constants below for the full rationale.
      - success:  W_SUCCESS when the scripted release lands the object in the
                  bin (credited to the transport policy).
      - P_IDLE:   small per-step time pressure so the agent finishes.
      - P_DROP:   one-time penalty for slipping the grasp away from the bin.
      - W_TIMEOUT_HELD: flat penalty for ending still gripping without ever
                  transporting to the bin (never triggered release).

    Termination:
      - done=True when the object is in the target bin (success)
      - done=True when object falls below table (fell_off_table)
      - done=True when the scripted release fired but missed (release_miss)
      - done=True when PLACE_HORIZON reached (timeout_*), or _STALL_LIMIT
        insurance / raw env horizon.

    info["place_done_reason"] is one of: "success", "fell_off_table",
    "release_miss", "stalled_holding", "transport_stall", "raw_env_horizon",
    "grasp_handoff_failed", "timeout_dropped", "timeout_released_miss",
    "timeout_held".
    """

    # --- POTENTIAL-BASED SHAPING --------------------------------------------
    # The per-step reward is gamma*Phi(s') - Phi(s), where Phi is a bounded
    # "progress toward placed" potential. This is the key design choice: any
    # *state*-based per-step reward (the old W_HOVER/W_LIFT/W_GRASP_MAINTAIN)
    # can be farmed by simply lingering in a high-reward state — e.g. hovering
    # the grasped object above the bin forever nets ~7/step, which discounts
    # out to more return than actually releasing. Potential shaping removes
    # this by construction: holding still gives ~0 per step (Phi unchanged),
    # and the total shaping collected along any path telescopes to
    # Phi(end) - Phi(start), independent of how long the agent lingers. It is
    # also policy-invariant (Ng et al. 1999), so it can't create new exploits.
    #
    # Phi is a cumulative-milestone function over the object pose (NOT gripper
    # state, so releasing at the bin keeps Phi high):
    #   Phi = PHI_LIFT     * lift_latch
    #       + PHI_APPROACH * lift_latch * xy_frac
    #       + PHI_PLACE    * lift_latch * xy_frac * place_frac
    # lift_latch is the *max* lift fraction achieved so far (a ratchet), so
    # lowering the object into the bin at the end doesn't erase lift credit.
    PHI_LIFT     = 2.0    # got the object off the table
    PHI_APPROACH = 4.0    # moved it over the target bin (XY). Raised 3->4 to
                          #   steepen the toward-bin gradient: the policy had
                          #   collapsed to a "hold lifted, do nothing" optimum
                          #   (score pinned at -14), so edging toward the bin
                          #   needs to pay more immediately.
    PHI_PLACE    = 6.0    # lowered it onto the target (final, weighted most).
                          #   Raised 5->6 alongside PHI_APPROACH.
    _GAMMA       = 0.99   # must match Agent's gamma (train_place.py)

    # --- genuine one-time / event rewards -----------------------------------
    W_SUCCESS        = 150.0  # one-time: _check_success() (object in bin),
                              #   credited to the transport policy when the
                              #   scripted release lands the object in the bin.
                              #   (There is no learned-release bonus: release is
                              #   scripted, so the old W_RELEASE was removed.)

    # Terminal penalty for ending the episode STILL GRIPPING, never having
    # released (done_reason timeout_held / stalled_holding). FLAT (uniform)
    # penalty, chosen after an A/B/C sweep proved distance-shaping is worse:
    #   * FLAT -20            : uniform -20 pressure at every distance. Best
    #                           performer — climbed to frac ~0.65 before hitting
    #                           a *transport* wall (holds far, no pull to cross).
    #   * SHAPED -8 + -18*far : worst case -42; deep high-variance target
    #                           destabilized the critic -> DUMPING (drops up to
    #                           17/50) alternating with safe-hold; capped at 0.40.
    #   * SHAPED -8 + -10*far : softer, drops fixed BUT the low -8 floor at the
    #                           bin removed the release pressure -> safe-hold
    #                           returned even at frac 0.30 (timeout_held 36/50).
    # Conclusion: the flat penalty's *uniform* release pressure is what works;
    # the shaped floor was too weak at the bin, and a strong-enough floor makes
    # the far end too deep to be stable. Magnitude/shape of the HOLD penalty is
    # exhausted. To break the ~0.65 transport wall the next lever is the APPROACH
    # POTENTIAL (PHI_APPROACH) — a policy-invariant pull toward the bin that adds
    # a transport gradient WITHOUT touching this penalty's release pressure.
    W_TIMEOUT_HELD   = -20.0  # one-time, on any "ended still holding" outcome

    # --- penalties ---
    # P_IDLE must stay SMALL relative to the shaping budget (PHI_* sum ~10).
    # At -0.3 it accumulated to -90 over the 150-step horizon — ~9x the whole
    # shaping signal — so the only way to cut the loss was to end the episode
    # early by DROPPING the object. That made "fail fast" the second-best
    # outcome after placing, and the drop-fast policy is exactly what the
    # done-reason tally showed (mostly timeout + fell_off_table, 0 success).
    # Now: idle over the full horizon (~-7.5) is milder than a drop, so
    # holding-and-trying always beats giving up.
    P_IDLE           = -0.05  # per-step time pressure (gentle)
    P_DROP           = -8.0   # one-time: lost grasp away from target. Lowered
                              #   from -25: that value existed to stop "fail-
                              #   fast drop farming" back when the object began
                              #   ON THE TABLE. With the handoff filter, grasps
                              #   are now robust and the object starts already
                              #   lifted, so that threat is gone. -25 was
                              #   instead POISONING the approach gradient — the
                              #   critic saw "move toward bin -> often -25" and
                              #   the policy retreated to the safe -14 hold. -8
                              #   keeps a drop slightly worse than holding (no
                              #   farming) while making an attempt roughly
                              #   break-even on the downside, so exploring
                              #   toward the bin is no longer punished.

    # --- motion gentling ---
    # OSC_POSE translation actions are per-step target deltas: large values
    # command hard acceleration, which inertially slips the bread out of the
    # gripper. The done-reason diagnostics proved this — grasps died mid-lift
    # (median step ~11-17) after a REAL lift (peak ~0.3 of full), i.e. shaken
    # loose by motion, not a marginal handoff (which would die at step ~1-5
    # with ~0 lift). Scaling translation caps the commanded acceleration so a
    # real grasp survives transport. The policy learns within the reduced
    # range; this mirrors the FSM notion of "carry slowly".
    _TRANSLATE_SCALE = 0.5    # multiplier on dx,dy,dz (dims 0:3)

    # --- distance scales ---
    _LIFT_CEIL       = 0.12   # lift height (m) that counts as fully lifted
    _HOVER_SCALE     = 0.60   # XY distance scale for approach potential.
                              #   RAISED 0.25 -> 0.60: xy_frac = max(0, 1-d/scale)
                              #   is EXACTLY ZERO beyond the scale distance, so at
                              #   0.25 the approach gradient only existed within
                              #   25cm of the bin — but full transport is ~0.53m.
                              #   Any handoff beyond curriculum frac ~0.47 landed
                              #   in a zero-gradient "desert" the policy had to
                              #   cross on exploration noise alone. This predicted
                              #   every observed wall: all runs ground at frac
                              #   0.40-0.45 (inside the field) and collapsed into
                              #   safe-hold at 0.5-0.65 (desert opens: holding IS
                              #   locally optimal where no gradient exists). 0.60
                              #   covers the full spawn->bin distance with margin,
                              #   so "toward the bin" pays from the first step at
                              #   every curriculum level. Slope flattens (4/0.6
                              #   ~6.7/m vs 16/m) but is nonzero EVERYWHERE.
    _PLACE_SCALE     = 0.20   # 3D distance scale for placement potential
    _NEAR_TARGET_XY  = 0.10   # XY threshold for "near target" (release/drop logic)

    # --- termination ---
    PLACE_HORIZON    = 200    # max steps for the place phase. Raised from 150
                              #   because _TRANSLATE_SCALE=0.5 roughly doubles
                              #   transport time; a successful place took 87
                              #   steps at full speed, so ~150-175 scaled — 200
                              #   leaves margin without wasting wall-clock.
    _STALL_LIMIT     = 60     # steps held over target before forced termination
    _TABLE_Z         = 0.78   # below this = object fell off the table (dropped)
    # Transport-stall early termination. Safe-hold collapses proved to be a
    # DATA-COMPOSITION problem no sampling fix can touch (PER_A 0.6->0.4 changed
    # nothing): a held episode rides the full 200-step horizon, so one bad
    # 50-episode window feeds the buffer ~9,000 "hold -> small negative"
    # transitions vs ~300 from successes — 96% of intake teaches "holding is
    # what we do" and the critic starves of transport data, cascading the
    # collapse down through mastered levels. Fix at the source: if the object
    # has not set a NEW BEST XY distance to the bin (by at least
    # _TRANSPORT_STALL_DELTA) for _TRANSPORT_STALL_LIMIT consecutive steps
    # while still gripped, terminate immediately as "transport_stall" (same
    # hold penalty). Real transport moves ~6mm/step, so 50 steps without 1cm
    # of new progress is a genuine stall, not a slow carry. Poison per failure
    # drops ~4x and episode turnover quadruples during recovery.
    _TRANSPORT_STALL_LIMIT = 50    # steps without new-best progress -> terminate
    _TRANSPORT_STALL_DELTA = 0.01  # (m) improvement that counts as progress

    # --- scripted release (transport/release decomposition) ----------------
    # The learned policy is a TRANSPORT policy only: its job is to bring the
    # grasped object stably over the bin. It does NOT control release (the
    # gripper is scripted closed the whole episode). Once the object has been
    # held within _NEAR_TARGET_XY of the bin center for _RELEASE_TRIGGER_HOLD
    # consecutive steps, an FSM-style scripted routine takes over: a P-controller
    # holds XY over the center while lowering the object to just above the bin
    # floor, then opens the gripper. This dissolves the transport-vs-release
    # conflict that no single reward could reconcile — "held stably over the bin"
    # becomes the transport policy's SUCCESS trigger instead of a failure mode.
    # Release needs no learning and no force feedback: lowering close before
    # opening is a deterministic descent, gentler than any sensor-blind policy
    # could manage. timeout_held now means ONLY "never reached the bin"
    # (transport failure) — clean, unconflated diagnostics.
    _RELEASE_TRIGGER_HOLD = 5     # consecutive over-bin+grasped steps -> release
    # The trigger fires at up to _NEAR_TARGET_XY (0.10 m) off-center — too far
    # to drop reliably into the bin (release_miss dominated). So the scripted
    # release first RECENTERS tightly (to _RELEASE_CENTER_TOL) before lowering,
    # rather than demanding tight centering from the learning policy.
    _RELEASE_CENTER_STEPS = 30    # max XY recenter steps before lowering
    _RELEASE_CENTER_TOL   = 0.03  # recenter to within this of bin center (m)
    _RELEASE_LOWER_STEPS  = 30    # max scripted descent steps
    _RELEASE_LOWER_DZ     = -0.12 # per-step -z descent command (gentle)
    _RELEASE_TOUCH_MARGIN = 0.02  # object within this of bin z => touched down
    _RELEASE_OPEN_STEPS   = 8     # steps holding gripper open to release + settle
    # robosuite's _check_success (pick_place.py) counts an object ONLY if the
    # gripper site is also ~4.2cm+ AWAY from it (r_reach = 1-tanh(10*dist) must
    # be < 0.6). Opening in place leaves the gripper parked on the object, so
    # genuinely-placed objects scored as release_miss (~70% "miss" rate that
    # recentering didn't dent). After opening, retract the empty gripper up and
    # re-check success as it clears.
    _RELEASE_RETRACT_STEPS = 12   # max +z retract steps after release
    _RELEASE_RETRACT_DZ    = 0.3  # per-step +z retract command (gripper empty)

    # --- grasp rollout ---
    _GRASP_HORIZON   = 200    # max steps for grasp rollout in reset()
    _GRASP_HOLD      = 8      # consecutive grasp steps to confirm stable grasp
    _MAX_GRASP_ATTEMPTS = 8   # retry resets if grasp fails / fails test-lift

    # --- handoff robustness filter (scripted test-lift) ---------------------
    # A grasp that passes _check_grasp statically can still be too weak to
    # survive transport — diagnostics showed real grasps (peak lift ~0.3)
    # dying mid-lift at step ~10-22. The grasp model was rewarded to ACHIEVE a
    # grasp, never to HOLD one under motion, so the handoff often hands off a
    # marginal grip. After confirming a grasp, reset() runs a scripted gentle
    # straight-up lift over exactly that danger window and only accepts grasps
    # that survive it. Survivors also hand off ALREADY lifted (past the fragile
    # window); non-survivors are rejected and retried. If reset() can rarely
    # find a surviving grasp (many episodes end "grasp_handoff_failed"), the
    # grasp model is categorically too weak and needs fine-tuning; if drops
    # collapse, robust grasps existed and just needed selecting.
    _TEST_LIFT_STEPS   = 20    # steps of scripted lift (covers the drop window)
    _TEST_LIFT_DZ      = 0.5   # +z command per step (~matches scaled transport)
    _TEST_LIFT_MIN_RISE = 0.03 # object must gain >=3cm and stay grasped to pass

    # --- reverse curriculum (scripted carry-toward-bin) --------------------
    # The policy can FIND placement but successes are far too sparse (1-5/50)
    # to bootstrap a stable value function — a discovery wall no reward tweak
    # fixes. So we shrink the task at the start: after the test-lift, a gentle
    # scripted P-controller carries the (grasped) object toward the bin until
    # it is within curriculum_frac * full_distance of the target, then hands
    # off. At curriculum_frac=0.2 the object starts ~right over the bin
    # (success = just lower + release; frequent reward), and the frac ramps to
    # 1.0 (true spawn handoff) as the policy masters each distance. Each env
    # tracks its OWN recent success rate and advances its OWN frac, so no
    # inter-process messaging is needed. NOTE: eval (test_place.py) should
    # construct the wrapper with curriculum=False for full-difficulty scoring.
    # Advancement is deliberately conservative and REVERSIBLE: a first run
    # advanced too eagerly (a hot streak at frac=0.2 cleared a lenient 50%/15
    # rule), stranding the half-baked policy at frac=0.35 with no way back —
    # Place% then decayed and the curriculum dead-ended. Now: advance only on
    # sustained mastery (0.7 over 25 eps), take smaller steps (0.1), and step
    # BACK DOWN if a level collapses (<0.25), so the curriculum self-corrects
    # to the hardest level the policy can actually hold.
    _CURRIC_START        = 0.2   # initial handoff distance as frac of full
    _CURRIC_STEP         = 0.1   # frac increment/decrement (low frac)
    # Above _CURRIC_HI_THRESH the task gets transport-sensitive: once the handoff
    # is far enough that placing requires real traversal (not just a release),
    # a 0.1 jump demands too much new skill at once. The agent then defaults to
    # SAFE-HOLD (timeout_held) — the flat -20 hold penalty is terminal, so
    # inching toward the bin and standing still both end held for the same cost,
    # giving no gradient to traverse the added distance. It reached ~0.40 then
    # regressed, oscillating ~0.35-0.40 instead of ratcheting up. Lowered the
    # threshold 0.4->0.3 so the small 0.05 step kicks in at the FIRST hard rung
    # (0.30->0.35->0.40...): each increment is small enough that the transport
    # learned at the previous rung transfers, letting the curriculum actually
    # bridge the exploration gap it exists to bridge. (Also guards the old 0.5-0.6
    # overshoot: fell_off_table with grasp_lost_step=-1 from too-big accuracy
    # jumps.) The trivial 0.20->0.30 climb stays on the coarse 0.1 step.
    _CURRIC_STEP_HI      = 0.05  # frac increment/decrement at/above HI_THRESH
    _CURRIC_HI_THRESH    = 0.3   # frac at/above which the smaller step applies
    # Window widened and regression made rarer to break a LIMIT CYCLE seen at
    # frac~0.40: the policy climbed to 0.40, reached ~58% (near the advance
    # bar), then a normal TD3 dip crossed the old 0.25 regress line and yanked
    # it all the way back to 0.20 — so it never accumulated enough sustained
    # practice at 0.40 to cross 0.70. A wider window (decisions on more data,
    # less noise) plus a much lower regress threshold (retreat only on a true
    # collapse, not a routine dip) lets it GRIND at the hard level until it
    # either masters it or genuinely fails.
    _CURRIC_WINDOW       = 40    # episodes (per env) to judge mastery over
    _CURRIC_ADVANCE_RATE = 0.75  # success rate in the window to advance (harder)
    _CURRIC_REGRESS_RATE = 0.15  # below this -> step back down a level (rarer)
    _CARRY_MAX_STEPS     = 120   # cap on scripted carry steps
    _CARRY_GAIN          = 4.0   # P-gain: action = gain * (bin_xy - obj_xy)
    _CARRY_CLIP          = 0.5   # clamp on per-step carry translation
    _CARRY_HOLD_DZ       = 0.1   # small +z each carry step to counteract sag

    def __init__(self, gym_env, raw_env, grasp_chkpt_dir,
                 grasp_layer1=64, grasp_layer2=32, curriculum=True):
        """
        Args:
            gym_env:          GymWrapper instance (provides flat observations)
            raw_env:          Raw robosuite env (for obs_dict, grasp check, etc.)
            grasp_chkpt_dir:  Path to trained grasp model checkpoint
            grasp_layer1:     Grasp model hidden layer 1 size
            grasp_layer2:     Grasp model hidden layer 2 size
            curriculum:       If True (training), scripted carry shrinks the
                              task and auto-recedes with mastery. Set False for
                              eval (test_place.py) to score full difficulty
                              (handoff at the true spawn, no carry).
        """
        self._gym_env = gym_env
        self._raw_env = raw_env
        self._grasp_chkpt_dir = grasp_chkpt_dir
        self._grasp_layer1 = grasp_layer1
        self._grasp_layer2 = grasp_layer2

        # Expose gym spaces (place agent uses the same obs/action spaces)
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space
        self._action_space_high = np.asarray(gym_env.action_space.high)
        self._action_space_low = np.asarray(gym_env.action_space.low)

        # Lazy-loaded grasp agent
        self._grasp_agent = None

        # When True (set by eval scripts with --render), the scripted phases
        # (grasp rollout, test-lift, carry, lower-and-release) also render each
        # sim step — otherwise the viewer freezes through reset() and the
        # release, since those loops step the env outside the caller's loop.
        self._render_scripted = False

        # Episode state
        self._init_z = None
        self._handoff_base_z = None   # object z BEFORE the scripted test-lift
        self._handoff_attempts = 0    # grasp+test-lift tries this reset
        self._target_bin = None
        self._success_given = False
        self._step_count = 0
        self._grasp_succeeded = False
        self._handoff_failed = False
        self._prev_grasped = False
        self._prev_phi = 0.0        # Phi(s) from the previous step (for shaping)
        self._max_lift_frac = 0.0   # ratchet: best lift fraction seen this episode
        self._over_target_steps = 0 # consecutive steps held over the target

        # Reverse-curriculum state (per env, self-advancing)
        from collections import deque
        self._curriculum = curriculum
        self._curriculum_frac = self._CURRIC_START if curriculum else 1.0
        self._curric_history = deque(maxlen=self._CURRIC_WINDOW)
        self._episode_ran = False   # guards curriculum update before ep 1

    # --- Gym-like interface ------------------------------------------------

    def _maybe_render(self):
        """Render one frame during scripted phases when eval rendering is on."""
        if self._render_scripted:
            try:
                self._raw_env.render()
            except Exception:
                self._render_scripted = False  # renderer unavailable — stop trying

    def reset(self):
        """Reset env, run grasp policy until a stable grasp that ALSO survives a
        scripted test-lift, then return the (already lifted) flat obs.

        Each attempt: run the grasp policy until _GRASP_HOLD consecutive grasp
        checks pass, then probe the grasp with a gentle straight-up lift. Only
        grasps that stay held (and actually rise) are accepted; the rest are
        rejected and retried. This filters out grips too weak to survive
        transport and hands off past the fragile initial-lift window.
        """
        self._ensure_grasp_agent()
        # Advance the curriculum based on the episode that just ended.
        self._update_curriculum()
        self._grasp_succeeded = False
        self._handoff_base_z = None
        self._handoff_attempts = 0

        for attempt in range(self._MAX_GRASP_ATTEMPTS):
            self._handoff_attempts = attempt + 1
            self._grasp_succeeded = False
            obs = self._gym_env.reset()
            grasp_hold = 0

            for step in range(self._GRASP_HORIZON):
                action = self._grasp_agent.choose_action(obs, validation=True)
                obs, _, done_raw, _ = self._gym_env.step(action)
                self._maybe_render()

                if done_raw:
                    break  # raw env horizon hit — retry

                if self._is_grasped():
                    grasp_hold += 1
                    if grasp_hold >= self._GRASP_HOLD:
                        self._grasp_succeeded = True
                        break
                else:
                    grasp_hold = 0

            if not self._grasp_succeeded:
                continue  # never got a stable grasp — retry

            # Handoff robustness filter: keep only grasps that survive a lift.
            obs2, survived = self._probe_test_lift(obs)
            if not survived:
                self._grasp_succeeded = False  # slipped — retry
                continue

            # Reverse curriculum: scripted-carry the object toward the bin so
            # the place task starts easy, then recedes as the policy improves.
            if self._curriculum and self._curriculum_frac < 1.0:
                obs2, carried = self._scripted_carry_toward_bin(obs2)
                if not carried:
                    self._grasp_succeeded = False  # dropped in carry — retry
                    continue
            obs = obs2
            break

        # Initialize place episode state
        self._init_place_state()
        return obs

    def _update_curriculum(self):
        """Advance this env's curriculum_frac based on its own recent success.

        Records the just-finished episode's outcome; once the rolling window is
        full and the success rate clears _CURRIC_ADVANCE_RATE, the handoff
        recedes _CURRIC_STEP further from the bin (harder) and the window
        resets. Runs entirely locally per env — no cross-process messaging.
        """
        if not self._curriculum or not self._episode_ran:
            self._episode_ran = True
            return
        self._curric_history.append(1.0 if self._success_given else 0.0)
        if len(self._curric_history) < self._CURRIC_WINDOW:
            return
        rate = float(np.mean(self._curric_history))
        # Smaller steps once the task is in the precision-sensitive high range.
        step = (self._CURRIC_STEP_HI
                if self._curriculum_frac >= self._CURRIC_HI_THRESH
                else self._CURRIC_STEP)
        if rate >= self._CURRIC_ADVANCE_RATE and self._curriculum_frac < 1.0:
            # Mastered this level -> make the handoff harder (farther from bin).
            self._curriculum_frac = min(1.0, self._curriculum_frac + step)
            self._curric_history.clear()
        elif (rate < self._CURRIC_REGRESS_RATE
              and self._curriculum_frac > self._CURRIC_START):
            # Level collapsed -> back off so the policy can recover and rebuild.
            self._curriculum_frac = max(self._CURRIC_START,
                                        self._curriculum_frac - step)
            self._curric_history.clear()

    def _scripted_carry_toward_bin(self, obs):
        """Gently carry the grasped, lifted object toward the bin (curriculum).

        A P-controller drives the object's XY toward the target until it is
        within curriculum_frac * full_distance of the bin, holding height and
        gripper closed. Returns (obs, still_grasped). If the grasp slips during
        the carry, returns False so the caller retries the handoff.
        """
        target = self._get_target_bin_pos()
        if target is None:
            return obs, True
        start = self._get_obj_pos(self._raw_env._get_observations())
        if start is None:
            return obs, True
        full_dist = float(np.linalg.norm(start[:2] - target[:2]))
        stop_dist = self._curriculum_frac * full_dist

        n = self.action_space.shape[0]
        for _ in range(self._CARRY_MAX_STEPS):
            obj = self._get_obj_pos(self._raw_env._get_observations())
            if obj is None:
                break
            d_xy = target[:2] - obj[:2]
            if np.linalg.norm(d_xy) <= stop_dist:
                break
            a = np.zeros(n, dtype=np.float32)
            a[0:2] = np.clip(self._CARRY_GAIN * d_xy,
                             -self._CARRY_CLIP, self._CARRY_CLIP)
            a[2] = self._CARRY_HOLD_DZ                  # hold height
            a[-1] = self._action_space_high[-1]         # gripper closed
            obs, _, done_raw, _ = self._gym_env.step(a)
            self._maybe_render()
            if done_raw or not self._is_grasped():
                return obs, False
        return obs, self._is_grasped()

    def _scripted_lower_and_release(self, obs):
        """Scripted FSM release: lower the grasped object onto the bin, open.

        Pure P-control, no learning and no force feedback. Two phases:
          1. Descend: hold XY over the bin center (P-controller) while commanding
             a gentle -z, until the object is within _RELEASE_TOUCH_MARGIN of the
             bin z or stops descending (a touchdown proxy — there is no contact
             sensor). Gripper stays closed.
          2. Open: release the gripper for _RELEASE_OPEN_STEPS while still
             holding XY, letting the object settle into the bin.
        Returns (obs, placed) where placed = _check_success() after the release.
        Lowering close before opening is what makes the place gentle; opening at
        transport height would let the object bounce out.
        """
        target = self._get_target_bin_pos()
        if target is None:
            return obs, False
        n = self.action_space.shape[0]
        gripper_closed = self._action_space_high[-1]
        gripper_open = self._action_space_low[-1]

        def _xy_cmd(a):
            """Set a[0:2] as a P-correction toward the bin center. Returns obj."""
            obj = self._get_obj_pos(self._raw_env._get_observations())
            if obj is not None:
                d_xy = target[:2] - obj[:2]
                a[0:2] = np.clip(self._CARRY_GAIN * d_xy,
                                 -self._CARRY_CLIP, self._CARRY_CLIP)
            return obj

        # --- Phase 0: recenter tightly over the bin BEFORE lowering ---------
        # The trigger allows up to _NEAR_TARGET_XY off-center, which drops onto
        # the rim. Hold height and P-correct XY until within _RELEASE_CENTER_TOL.
        for _ in range(self._RELEASE_CENTER_STEPS):
            obj = self._get_obj_pos(self._raw_env._get_observations())
            if obj is None or not self._is_grasped():
                break
            if np.linalg.norm(target[:2] - obj[:2]) <= self._RELEASE_CENTER_TOL:
                break                                        # centered enough
            a = np.zeros(n, dtype=np.float32)
            _xy_cmd(a)
            a[-1] = gripper_closed
            obs, _, done_raw, _ = self._gym_env.step(a)
            self._maybe_render()
            if done_raw:
                break

        # --- Phase 1: descend to touchdown (keep XY centered) --------------
        prev_z = None
        for _ in range(self._RELEASE_LOWER_STEPS):
            if not self._is_grasped():
                break  # slipped during descent — go straight to success check
            a = np.zeros(n, dtype=np.float32)
            _xy_cmd(a)
            a[2] = self._RELEASE_LOWER_DZ
            a[-1] = gripper_closed
            obs, _, done_raw, _ = self._gym_env.step(a)
            self._maybe_render()
            if done_raw:
                break
            obj = self._get_obj_pos(self._raw_env._get_observations())
            if obj is not None:
                if obj[2] <= target[2] + self._RELEASE_TOUCH_MARGIN:
                    break                                   # reached bin floor
                if prev_z is not None and obj[2] >= prev_z - 1e-4:
                    break                                   # z stalled = resting
                prev_z = obj[2]

        # --- Phase 2: open and settle (no arm motion, let it drop in) ------
        for _ in range(self._RELEASE_OPEN_STEPS):
            a = np.zeros(n, dtype=np.float32)
            a[-1] = gripper_open
            obs, _, done_raw, _ = self._gym_env.step(a)
            self._maybe_render()
            if done_raw:
                break

        # --- Phase 3: retract clear, then judge ------------------------------
        # _check_success needs the gripper ~4.2cm+ away from the object; lift
        # the (now empty) gripper straight up and accept the first success seen.
        placed = False
        for _ in range(self._RELEASE_RETRACT_STEPS):
            a = np.zeros(n, dtype=np.float32)
            a[2] = self._RELEASE_RETRACT_DZ
            a[-1] = gripper_open
            obs, _, done_raw, _ = self._gym_env.step(a)
            self._maybe_render()
            try:
                if self._raw_env._check_success():
                    placed = True
                    break
            except Exception:
                pass
            if done_raw:
                break
        return obs, placed

    def _probe_test_lift(self, obs):
        """Scripted gentle straight-up lift to probe grasp robustness.

        Commands +z (gripper closed, no rotation, no XY) for _TEST_LIFT_STEPS
        steps — the same window where real grasps were slipping. Returns
        (obs, survived). On success the object has risen a few cm and this
        lifted state becomes the handoff. Records the pre-lift object height in
        self._handoff_base_z so the lift potential is credited from the table,
        not from the already-lifted handoff pose.
        """
        start = self._get_obj_pos(self._raw_env._get_observations())
        self._handoff_base_z = float(start[2]) if start is not None else 0.845

        lift_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        lift_action[2] = self._TEST_LIFT_DZ                # +z
        lift_action[-1] = self._action_space_high[-1]      # gripper closed

        for _ in range(self._TEST_LIFT_STEPS):
            obs, _, done_raw, _ = self._gym_env.step(lift_action)
            self._maybe_render()
            if done_raw or not self._is_grasped():
                return obs, False

        end = self._get_obj_pos(self._raw_env._get_observations())
        rose = (end is not None
                and (end[2] - self._handoff_base_z) >= self._TEST_LIFT_MIN_RISE)
        return obs, bool(self._is_grasped() and rose)

    def step(self, action):
        """Step the env with transport-specific reward and termination.

        The learned policy is a TRANSPORT policy: it controls translation only,
        to bring the grasped object over the bin. Release is NOT learned — it is
        a scripted FSM routine triggered once the object is stably over the bin
        (see _scripted_lower_and_release). Three action shims apply (OSC_POSE
        layout: [dx, dy, dz, droll, dpitch, dyaw, gripper]):

        1. Orientation freeze (dims 3:6 -> 0): commanding rotations shears the
           bread out of the gripper; the grasp pose is already fine for placing.

        2. Gripper scripted closed (dim 6 -> +1, EVERY step): the policy never
           releases. Opening is owned entirely by the scripted release routine,
           so the transport policy can't pop the grasp open mid-carry and has no
           release-commitment decision to conflict with its transport objective.

        3. Translation gentling (dims 0:3 *= _TRANSLATE_SCALE): caps commanded
           acceleration so a real grasp survives transport (motion slippage
           diagnostics).

        The masked action actually executed is returned in info["applied_action"]
        so the training loop stores the same action the environment saw.
        """
        action = np.asarray(action, dtype=np.float32).copy()
        action[3:6] = 0.0                                   # freeze orientation
        action[0:3] *= self._TRANSLATE_SCALE                # gentle translation
        action[-1] = self._action_space_high[-1]            # gripper scripted closed

        obs, _, done_raw, info = self._gym_env.step(action)
        self._step_count += 1

        # Get raw obs_dict for reward computation
        obs_dict = self._raw_env._get_observations()
        reward = self._place_reward(obs_dict)

        # --- Scripted release trigger: object held stably over the bin ---------
        # Once transport has parked the grasped object over the bin, hand off to
        # the scripted lower-and-release. A successful place credits the transport
        # policy with W_SUCCESS; a miss ends the episode as "release_miss".
        released = False
        if (not self._success_given
                and self._is_grasped()
                and self._over_target_steps >= self._RELEASE_TRIGGER_HOLD):
            obs, placed = self._scripted_lower_and_release(obs)
            obs_dict = self._raw_env._get_observations()
            released = True
            if placed and not self._success_given:
                reward += self.W_SUCCESS
                self._success_given = True

        done = self._check_done(obs_dict, info, done_raw, released)

        # timeout_held now means ONLY "never transported to the bin" (the object
        # never triggered release): flat penalty so failing to transport is worse
        # than a drop, exactly as the hold-penalty A/B/C sweep concluded.
        # transport_stall is the same failure detected early (see
        # _TRANSPORT_STALL_LIMIT) and takes the same penalty.
        if done and info.get("place_done_reason") in ("timeout_held",
                                                       "stalled_holding",
                                                       "transport_stall"):
            reward += self.W_TIMEOUT_HELD

        info["applied_action"] = action
        return obs, reward, done, info

    def close(self):
        self._gym_env.close()

    def seed(self, seed):
        self._gym_env.seed(seed)

    def render(self, **kwargs):
        self._gym_env.render(**kwargs)

    # --- Grasp agent management --------------------------------------------

    def _ensure_grasp_agent(self):
        """Lazy-load the trained grasp agent (only on first reset)."""
        if self._grasp_agent is not None:
            return

        from td3 import Agent

        self._grasp_agent = Agent(
            alpha=0.0003,
            beta=0.0003,
            tau=0.005,
            input_dims=self._gym_env.observation_space.shape,
            env=self._gym_env,   # provides correct action_space bounds
            n_actions=self._gym_env.action_space.shape[0],
            layer1_size=self._grasp_layer1,
            layer2_size=self._grasp_layer2,
            batch_size=512,
            chkpt_dir=self._grasp_chkpt_dir,
        )
        self._grasp_agent.load_models()

        # Force CPU for safety in forked subprocesses
        device = T.device('cpu')
        self._grasp_agent.actor.to(device)
        self._grasp_agent.actor.device = device
        self._grasp_agent.device = device

        print(f"  [PlaceWrapper] Grasp agent loaded from {self._grasp_chkpt_dir}")

    # --- Place state management --------------------------------------------

    def _init_place_state(self):
        """Reset internal tracking for the place phase."""
        obs_dict = self._raw_env._get_observations()
        obj_pos = self._get_obj_pos(obs_dict)
        # Credit lift from the pre-test-lift (table) height so the scripted
        # handoff lift doesn't zero out the lift potential the policy can earn.
        if self._handoff_base_z is not None:
            self._init_z = self._handoff_base_z
        else:
            self._init_z = float(obj_pos[2]) if obj_pos is not None else 0.845
        self._target_bin = None
        self._success_given = False
        self._step_count = 0
        self._handoff_failed = not self._grasp_succeeded
        # Only assume "previously grasped" if the handoff actually succeeded —
        # otherwise the first step would wrongly fire the drop penalty.
        self._prev_grasped = self._grasp_succeeded
        self._max_lift_frac = 0.0
        self._over_target_steps = 0
        self._dropped_away = False   # True once a P_DROP (grasp lost away) fires
        self._grasp_lost_step = -1   # step at which grasp was first lost (-1 = never)
        # Starts at spawn (not over the bin), so the gripper is masked closed
        # on the first step until the object is transported over the target.
        self._last_over_target = False
        # Transport-stall tracking: best (smallest) XY distance to the bin so
        # far, and how many consecutive steps since it last improved.
        self._best_dxy = None
        self._no_progress_steps = 0
        # Seed the shaping baseline with the potential of the handoff state,
        # so the first step's gamma*Phi(s') - Phi(s) measures real progress
        # rather than a spurious jump up from zero.
        self._prev_phi = self._potential(obj_pos, self._get_target_bin_pos())

    # --- Place reward computation ------------------------------------------

    def _potential(self, obj_pos, target):
        """Bounded 'progress toward placed' potential Phi(s) in [0, PHI_MAX].

        Cumulative milestones (each gated by the previous so they must be
        achieved in order): lift the object -> move it over the bin (XY) ->
        lower it onto the target (3D proximity). Based on the *object* pose
        only (not the gripper), so releasing at the bin keeps Phi high.

        lift credit uses a per-episode ratchet (max lift seen so far) so that
        lowering the object into the bin at the end does not erase the lift
        milestone and create a negative-shaping disincentive to place.
        """
        if obj_pos is None:
            return 0.0

        lift = max(0.0, obj_pos[2] - self._init_z)
        lift_frac = min(1.0, lift / self._LIFT_CEIL)
        self._max_lift_frac = max(self._max_lift_frac, lift_frac)
        lift_latch = self._max_lift_frac

        phi = self.PHI_LIFT * lift_latch

        if target is not None:
            d_xy = np.linalg.norm(obj_pos[:2] - target[:2])
            xy_frac = max(0.0, 1.0 - d_xy / self._HOVER_SCALE)
            phi += self.PHI_APPROACH * lift_latch * xy_frac

            d_3d = np.linalg.norm(obj_pos - target)
            place_frac = max(0.0, 1.0 - d_3d / self._PLACE_SCALE)
            phi += self.PHI_PLACE * lift_latch * xy_frac * place_frac

        return phi

    def _place_reward(self, obs_dict):
        obj_pos = self._get_obj_pos(obs_dict)
        if obj_pos is None:
            return 0.0

        grasped = self._is_grasped()
        target = self._get_target_bin_pos()

        # --- Potential-based shaping: gamma*Phi(s') - Phi(s) ---------------
        phi = self._potential(obj_pos, target)
        r = self._GAMMA * phi - self._prev_phi
        self._prev_phi = phi

        # --- Time pressure --------------------------------------------------
        r += self.P_IDLE

        # --- Track lingering over the target (for stall termination) --------
        over_target = False
        if target is not None:
            d_xy = np.linalg.norm(obj_pos[:2] - target[:2])
            over_target = d_xy <= self._NEAR_TARGET_XY
            # Transport-stall tracking: only new-best progress toward the bin
            # (by at least _TRANSPORT_STALL_DELTA) resets the counter, and only
            # while gripped — after a drop the episode resolves via drop paths.
            if grasped:
                if (self._best_dxy is None
                        or d_xy < self._best_dxy - self._TRANSPORT_STALL_DELTA):
                    self._best_dxy = float(d_xy)
                    self._no_progress_steps = 0
                else:
                    self._no_progress_steps += 1
        self._over_target_steps = self._over_target_steps + 1 if over_target else 0
        # Drives gripper masking on the next step: closed until over the bin.
        self._last_over_target = over_target

        # --- One-time event: grasp lost DURING TRANSPORT --------------------
        # Release is scripted, so any grasp loss here is an unwanted slip. Only
        # penalize slips away from the bin (P_DROP); a slip within _NEAR_TARGET_XY
        # is essentially at the goal and may still land in the bin, so it is left
        # neutral rather than punished.
        if self._prev_grasped and not grasped and target is not None:
            # Diagnostic: record WHEN the grasp first died. Early (~1-5) =>
            # handoff grasp was never solid; mid-transport => motion slippage.
            if self._grasp_lost_step < 0:
                self._grasp_lost_step = self._step_count
            d_xy = np.linalg.norm(obj_pos[:2] - target[:2])
            if d_xy > self._NEAR_TARGET_XY:
                r += self.P_DROP
                self._dropped_away = True

        self._prev_grasped = grasped

        # --- One-time success bonus -----------------------------------------
        if not self._success_given:
            try:
                if self._raw_env._check_success():
                    r += self.W_SUCCESS
                    self._success_given = True
            except Exception:
                pass

        return float(r)

    # --- Termination check -------------------------------------------------

    def _check_done(self, obs_dict, info, done_raw, released=False):
        # Diagnostics attached on every step (read by the training loop only at
        # termination): when the grasp died and how high the object ever got.
        info["grasp_lost_step"] = self._grasp_lost_step
        info["max_lift_frac"] = round(float(self._max_lift_frac), 3)
        info["handoff_attempts"] = self._handoff_attempts
        info["curriculum_frac"] = round(float(self._curriculum_frac), 3)

        # Success: object placed in target bin
        if self._success_given:
            info["place_success"] = True
            info["place_done_reason"] = "success"
            return True

        # Object fell off table
        obj_pos = self._get_obj_pos(obs_dict)
        if obj_pos is not None and obj_pos[2] < self._TABLE_Z:
            info["place_success"] = False
            info["place_done_reason"] = "fell_off_table"
            return True

        # Scripted release fired over the bin but the object did not end up in
        # it (and didn't fall off the table): transport delivered, the release
        # missed. High rates here => tune the descent (height/steps/XY), not RL.
        if released:
            info["place_success"] = False
            info["place_done_reason"] = "release_miss"
            return True

        # Stalled: object held over the target for too long without placing.
        # Insurance only — potential shaping already removes the incentive to
        # linger — but it keeps degenerate trajectories from eating the full
        # horizon and slowing buffer turnover.
        if self._over_target_steps > self._STALL_LIMIT:
            info["place_success"] = False
            info["place_done_reason"] = "stalled_holding"
            return True

        # Transport stall: gripped but no new-best progress toward the bin for
        # too long — the safe-hold pattern. Terminate NOW instead of letting it
        # ride the 200-step horizon, so a failing policy can't flood the buffer
        # with hold transitions (the data-composition collapse amplifier).
        if (self._no_progress_steps >= self._TRANSPORT_STALL_LIMIT
                and self._is_grasped()):
            info["place_success"] = False
            info["place_done_reason"] = "transport_stall"
            return True

        # Raw env horizon reached
        if done_raw:
            info["place_success"] = False
            info["place_done_reason"] = "raw_env_horizon"
            return True

        # Place phase timeout — split by what actually happened so the tally
        # distinguishes "held the object the whole time but never moved it to
        # the target" (exploration problem) from "dropped it away from the
        # target but it stayed on a surface, then idled out" (grip-stability
        # problem). These two look identical in the score but need opposite
        # fixes, so they must be separated in the diagnostics.
        if self._step_count >= self.PLACE_HORIZON:
            info["place_success"] = False
            if self._handoff_failed:
                reason = "grasp_handoff_failed"
            elif self._dropped_away:
                reason = "timeout_dropped"
            elif not self._is_grasped():
                # Grasp slipped near the bin (neutral zone, no P_DROP) but the
                # object never reached the bin and never triggered the scripted
                # release — a rare boundary slip.
                reason = "timeout_released_miss"
            else:
                reason = "timeout_held"   # still gripping: never transported to bin
            info["place_done_reason"] = reason
            return True

        return False

    # --- Shared helpers ----------------------------------------------------

    def _get_target_obj_name(self):
        try:
            return self._raw_env.obj_to_use
        except AttributeError:
            pass
        for cand in ("Can", "Milk", "Cereal", "Bread"):
            try:
                if self._raw_env.object_to_id.get(cand) is not None:
                    return cand
            except AttributeError:
                pass
        return None

    def _get_obj_pos(self, obs_dict):
        name = self._get_target_obj_name()
        if name is not None:
            key = f"{name}_pos"
            if key in obs_dict:
                return np.array(obs_dict[key])
        for cand in ("Can_pos", "Milk_pos", "Cereal_pos", "Bread_pos"):
            if cand in obs_dict:
                return np.array(obs_dict[cand])
        return None

    def _get_target_bin_pos(self):
        """Return XYZ position of the target bin for the active object."""
        if self._target_bin is not None:
            return self._target_bin
        try:
            name = self._get_target_obj_name()
            obj_idx = self._raw_env.object_to_id[name.lower()]
            self._target_bin = np.array(
                self._raw_env.target_bin_placements[obj_idx]
            )
            return self._target_bin
        except Exception:
            return None

    def _is_grasped(self):
        try:
            return self._raw_env._check_grasp(
                gripper=self._raw_env.robots[0].gripper,
                object_geoms=self._raw_env.objects[self._raw_env.object_id],
            )
        except Exception:
            return False
