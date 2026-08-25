#!/usr/bin/env python3
# Last updated: 2026-06-30
"""
Grasp-only environment wrapper for decomposed RL training.

Wraps the raw robosuite PickPlace env (before GymWrapper) to provide:
  - Reach + grasp-only reward signal (no lift/hover/place rewards)
  - Early termination on stable grasp (N consecutive grasp-confirmed steps)
  - Shorter horizon (200 steps) — reaching + grasping shouldn't need 25s

This is the first stage in the two-model decomposition:
  Stage 1 (this): Reach → Grasp  |  done when _check_grasp() stable
  Stage 2 (future): Lift → Place  |  starts from grasp terminal states
"""

import numpy as np


class GraspRewardWrapper:
    """
    Staged dense reward for the grasp sub-task only.

    Reward pipeline:
      1. reach:   gripper → object proximity  (0 to W_REACH per step)
      2. grip:    gripper-close bonus when near object
      3. grasp:   sustained grasp bonus       (W_GRASP per step while held)
      4. success: one-time stable-grasp bonus  (W_GRASP_SUCCESS, awarded once)

    Termination:
      - done=True when _check_grasp() is True for N_GRASP_HOLD consecutive steps
      - done=True when step count reaches GRASP_HORIZON (timeout/failure)
    """

    # --- reward weights (per step unless noted) ---
    W_REACH         = 1.0
    W_GRIP_CLOSE    = 0.5
    W_GRASP         = 10.0
    W_GRASP_SUCCESS = 20.0   # one-time bonus for stable grasp

    # --- penalties ---
    P_IDLE          = -0.4   # small per-step cost to discourage doing nothing
    P_DROP          = -5.0   # dropping a grasped object
    P_AWAY          = -0.7   # moving gripper away from object when not grasped

    # --- distance scales ---
    _REACH_SCALE    = 0.30
    _GRIP_RANGE     = 0.06

    # --- termination ---
    N_GRASP_HOLD    = 5      # consecutive grasp-confirmed steps for success
    GRASP_HORIZON   = 200    # max steps per episode (shorter than full task)

    # --- lift certification (require_lift=True) -----------------------------
    # The bare N_GRASP_HOLD criterion certifies momentary contact, not a grip
    # that can be carried. Measured end-to-end: 86% of episodes satisfied the
    # 5-step hold, but only 43% survived the place stage's handoff, which adds
    # an 8-step hold plus a scripted lift. The stage-1 metric was rewarding
    # fast snatches (25-30 steps) that fail the moment the object leaves the
    # table. These mirror place_env_wrapper's probe EXACTLY so the two stages
    # agree on what a grasp is.
    N_GRASP_HOLD_LIFT   = 8     # matches place_env_wrapper._GRASP_HOLD
    # Partial credit for a failed lift must stay WELL below the success bonus.
    # The first version paid W_GRASP_SUCCESS * (rise/threshold) linearly, on top
    # of the +20 already paid for reaching the hold -- so a lift that failed at
    # 14mm scored 29.2 while a lift that PASSED scored 20. Failing paid more
    # than succeeding, and the policy duly plateaued at a 13.8mm median rise.
    LIFT_PARTIAL_CAP    = 0.25  # max share of W_GRASP_SUCCESS a failure may earn
    LIFT_PARTIAL_POW    = 3.0   # cubic: flat until close, steep near threshold
    TEST_LIFT_STEPS     = 20    # matches _TEST_LIFT_STEPS
    TEST_LIFT_DZ        = 0.5   # matches _TEST_LIFT_DZ
    TEST_LIFT_MIN_RISE  = 0.03  # matches _TEST_LIFT_MIN_RISE
    # No flat penalty for a failed lift. Reaching the hold ENDS the episode, so
    # a failed lift already costs the agent every remaining per-step W_GRASP;
    # stacking a penalty on top makes grasping-and-failing worse than never
    # grasping, and the agent can learn to avoid grasping altogether. Instead
    # the success bonus is paid out in proportion to the rise actually
    # achieved, which keeps the signal dense while a policy is learning to
    # firm up its grip.

    # --- wrist alignment (align_grip=True) ----------------------------------
    # The handoff diagnostic (Results/handoff_diagnostic.txt) measured object
    # yaw relative to the wrist at handoff, over 400 episodes x 3 independent
    # grasp policies. The bread is a box: at 0 or 90 degrees the fingers close
    # on flat faces, at ~45 degrees they close on CORNERS. End-to-end success,
    # flat -> corner:
    #   baseline 74.7 -> 54.7 | gripfix_s0 72.2 -> 45.7 | gripfix_s2 92.6 -> 70.2
    # Monotonic in all three, p < 5e-3, and 24% of all handoffs are corner
    # grips. A corner grip PASSES lift certification -- it survives a straight
    # vertical pull -- and only fails later under transport's lateral loads, so
    # W_GRASP_SUCCESS pays it in full. The policy commands wrist yaw through
    # OSC_POSE; it can align to a face, it just has no reason to.
    #
    # Two pieces, deliberately separate:
    #   (a) the success bonus is scaled by alignment, so a corner grip earns
    #       less for the same certified lift;
    #   (b) potential-based shaping gives a dense gradient to rotate the wrist
    #       BEFORE closing -- without it (a) is a near-zero-gradient signal
    #       arriving only at episode end.
    ALIGN_PERIOD    = 90.0   # deg; box symmetry -- 0 and 90 are both flat faces
    ALIGN_WORST     = 45.0   # deg from a flat face = corner grip
    # Floor on the success multiplier. MUST stay above LIFT_PARTIAL_CAP (0.25):
    # below it a certified-but-misaligned lift would pay LESS than a near-miss
    # failure -- the same reward inversion that produced unliftable grips the
    # first time (see the LIFT_PARTIAL_CAP note above). At 0.5 the worst
    # certified grip earns 10.0 against a failure's 5.0 ceiling.
    ALIGN_MIN       = 0.5
    # Potential-based shaping (Ng et al. 1999): r += gamma*Phi(s') - Phi(s).
    # Policy-invariant by construction, so unlike a plain per-step alignment
    # bonus it cannot be farmed by hovering near the object -- the safe-hold
    # collapse this codebase has hit before. Gated by proximity so the term is
    # silent during the reach and shapes only the final approach.
    # Two consequences of gamma < 1 worth knowing, both benign here:
    #  - lingering in a well-aligned state costs ~0.02/step (the potential
    #    decays), against P_IDLE's -0.4 -- negligible, and the right sign;
    #  - Phi(terminal) is not subtracted, so ending an episode well aligned
    #    keeps ~+2 of potential. That is strict policy-invariance broken, but
    #    in the direction the multiplier already pushes, and it is dwarfed by
    #    the 200-step idle cost of stalling to collect it.
    W_ALIGN_POT     = 3.0
    ALIGN_GAMMA     = 0.99   # matches the agents' discount
    ALIGN_NEAR      = 0.10   # (m) proximity scale of the shaping gate

    def __init__(self, env, require_lift=False, align_grip=False):
        """require_lift: certify grasps with a scripted lift before calling
        them a success. Default False so every result recorded before this
        existed stays reproducible; new runs should pass True."""
        self._rs_env = env
        self._require_lift = bool(require_lift)
        self._align_grip = bool(align_grip)
        self._success_given = False
        self._prev_grasped = False
        self._prev_d_reach = None
        self._prev_align_pot = None
        self._grasp_hold_count = 0
        self._step_count = 0

    def __getattr__(self, name):
        return getattr(self._rs_env, name)

    def reset(self):
        obs_dict = self._rs_env.reset()
        self._success_given = False
        self._prev_grasped = False
        self._prev_d_reach = None
        self._prev_align_pot = None
        self._grasp_hold_count = 0
        self._step_count = 0
        return obs_dict

    def step(self, action):
        obs_dict, _, _, info = self._rs_env.step(action)
        self._step_count += 1

        reward = self._grasp_reward(obs_dict)

        # Check termination: stable grasp or timeout
        grasped = self._is_grasped()
        if grasped:
            self._grasp_hold_count += 1
        else:
            self._grasp_hold_count = 0

        # Stable grasp achieved
        done = False
        hold_needed = (self.N_GRASP_HOLD_LIFT if self._require_lift
                       else self.N_GRASP_HOLD)
        if self._grasp_hold_count >= hold_needed:
            done = True
            if self._require_lift:
                survived, rise = self._certify_by_lift(obs_dict)
                info["grasp_success"] = survived
                info["lift_certified"] = survived
                info["lift_rise"] = rise
                frac = float(np.clip(rise / self.TEST_LIFT_MIN_RISE, 0.0, 1.0))
                # Scale the bonus by how squarely the fingers sit on a face.
                # A corner grip certifies but transports badly, so it must not
                # earn what a flat grip earns. Floored at ALIGN_MIN to keep a
                # certified success strictly above the failure ceiling.
                align_q = self._wrist_alignment(obs_dict) if self._align_grip else None
                align_mult = 1.0 if align_q is None else (
                    self.ALIGN_MIN + (1.0 - self.ALIGN_MIN) * align_q)
                info["grip_align"] = align_q
                if survived:
                    # The bonus is paid HERE, not on reaching the hold, so it
                    # is contingent on the lift actually succeeding.
                    reward += self.W_GRASP_SUCCESS * align_mult
                else:
                    reward += (self.W_GRASP_SUCCESS * self.LIFT_PARTIAL_CAP
                               * frac ** self.LIFT_PARTIAL_POW)
            else:
                info["grasp_success"] = True

        # Timeout
        if self._step_count >= self.GRASP_HORIZON:
            done = True
            if "grasp_success" not in info:
                info["grasp_success"] = False

        return obs_dict, reward, done, info

    # --- helpers -----------------------------------------------------------

    def _certify_by_lift(self, obs_dict):
        """Scripted straight-up lift. Returns (survived, rise_metres).

        True only if the object actually rises and stays held. Mirrors
        place_env_wrapper._probe_test_lift so a grasp this wrapper accepts is
        one the place stage will also accept.

        Consumes raw-env steps inside a single wrapper step. That is fine here:
        the episode ends either way, so no post-lift transition is lost, and the
        agent still gets the outcome through the reward and grasp_success.
        """
        start = self._get_obj_pos(obs_dict)
        base_z = float(start[2]) if start is not None else 0.845

        dim = self._rs_env.action_spec[0].shape[0]
        lift_action = np.zeros(dim, dtype=np.float32)
        lift_action[2] = self.TEST_LIFT_DZ     # +z
        lift_action[-1] = 1.0                  # keep the gripper closed

        best_rise = 0.0
        for _ in range(self.TEST_LIFT_STEPS):
            obs_dict, _, done_raw, _ = self._rs_env.step(lift_action)
            cur = self._get_obj_pos(obs_dict)
            if cur is not None:
                best_rise = max(best_rise, float(cur[2] - base_z))
            if done_raw or not self._is_grasped():
                return False, best_rise

        end = self._get_obj_pos(obs_dict)
        rise = float(end[2] - base_z) if end is not None else 0.0
        best_rise = max(best_rise, rise)
        survived = bool(self._is_grasped() and rise >= self.TEST_LIFT_MIN_RISE)
        return survived, best_rise

    def _get_target_obj_name(self):
        try:
            return self._rs_env.obj_to_use
        except AttributeError:
            pass
        for cand in ("Can", "Milk", "Cereal", "Bread"):
            try:
                if self._rs_env.object_to_id.get(cand) is not None:
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

    def _is_grasped(self):
        try:
            return self._rs_env._check_grasp(
                gripper=self._rs_env.robots[0].gripper,
                object_geoms=self._rs_env.objects[self._rs_env.object_id],
            )
        except Exception:
            return False

    @staticmethod
    def _quat_yaw(q):
        """Yaw about world z from a robosuite (xyzw) quaternion."""
        x, y, z, w = q
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _wrist_alignment(self, obs_dict):
        """Grip alignment quality in [0, 1]: 1.0 = fingers on a flat face,
        0.0 = fingers on a corner.

        The residual between object yaw and wrist yaw is wrapped into
        [0, ALIGN_PERIOD) -- a two-finger gripper on a box is symmetric under
        90 degrees, so 0 and 90 are the same (good) grip and 45 is the worst.
        Returns None when the pose is unavailable, so callers can skip the term
        rather than silently score it as a corner grip.
        """
        name = self._get_target_obj_name()
        key = f"{name}_quat" if name is not None else None
        if key is None or key not in obs_dict or "robot0_eef_quat" not in obs_dict:
            return None
        rel = np.degrees(self._quat_yaw(np.array(obs_dict[key]))
                         - self._quat_yaw(np.array(obs_dict["robot0_eef_quat"])))
        off = abs((rel + self.ALIGN_PERIOD / 2) % self.ALIGN_PERIOD
                  - self.ALIGN_PERIOD / 2)          # deg from nearest flat face
        return float(np.clip(1.0 - off / self.ALIGN_WORST, 0.0, 1.0))

    def _align_potential(self, obs_dict, d_reach):
        """Phi(s) for potential-based alignment shaping. Zero far from the
        object so the term never competes with the reach reward."""
        q = self._wrist_alignment(obs_dict)
        if q is None:
            return 0.0
        gate = float(np.exp(-d_reach / self.ALIGN_NEAR))
        return self.W_ALIGN_POT * q * gate

    def _gripper_aperture(self, obs_dict):
        """Return normalised gripper opening: 0 = closed, 1 = fully open."""
        try:
            qpos = np.array(obs_dict["robot0_gripper_qpos"])
            # Panda gripper: qpos ≈ [+0.02, -0.02] when open, [0, 0] closed
            return float(np.clip(abs(qpos[0]) / 0.04, 0.0, 1.0))
        except Exception:
            return 0.5

    def _grasp_reward(self, obs_dict):
        r = 0.0
        eef_pos = np.array(obs_dict["robot0_eef_pos"])
        obj_pos = self._get_obj_pos(obs_dict)
        if obj_pos is None:
            return r

        d_reach = np.linalg.norm(eef_pos - obj_pos)

        # ---- Stage 1: Reach toward object ----------------------------------
        r += self.W_REACH * max(0.0, 1.0 - d_reach / self._REACH_SCALE)

        # ---- Stage 2: Encourage gripper closing when very close ------------
        if d_reach < self._GRIP_RANGE:
            aperture = self._gripper_aperture(obs_dict)
            r += self.W_GRIP_CLOSE * (1.0 - aperture)  # reward closing

        # ---- Grasp check ---------------------------------------------------
        grasped = self._is_grasped()

        # ---- Penalties -----------------------------------------------------
        r += self.P_IDLE

        if self._prev_grasped and not grasped:
            r += self.P_DROP

        if not grasped and self._prev_d_reach is not None:
            if d_reach > self._prev_d_reach + 0.005:
                r += self.P_AWAY

        # ---- Wrist alignment (potential-based) -----------------------------
        # gamma*Phi(s') - Phi(s). The first step of an episode has no previous
        # potential, so it contributes nothing rather than a spurious +Phi.
        if self._align_grip:
            pot = self._align_potential(obs_dict, d_reach)
            if self._prev_align_pot is not None:
                r += self.ALIGN_GAMMA * pot - self._prev_align_pot
            self._prev_align_pot = pot

        self._prev_grasped = grasped
        self._prev_d_reach = d_reach

        if grasped:
            # ---- Stage 3: Sustained grasp ----------------------------------
            r += self.W_GRASP

        # ---- One-time stable grasp bonus -----------------------------------
        # With require_lift the bonus is contingent on the lift, and is paid in
        # step() once _certify_by_lift has run. Paying it here as well would
        # reward merely holding contact, which is the behaviour that produced
        # unliftable grips in the first place.
        if (not self._require_lift and not self._success_given
                and self._grasp_hold_count >= self.N_GRASP_HOLD):
            r += self.W_GRASP_SUCCESS
            self._success_given = True

        return float(r)
