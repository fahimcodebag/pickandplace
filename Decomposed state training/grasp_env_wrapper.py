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

    def __init__(self, env):
        self._rs_env = env
        self._success_given = False
        self._prev_grasped = False
        self._prev_d_reach = None
        self._grasp_hold_count = 0
        self._step_count = 0

    def __getattr__(self, name):
        return getattr(self._rs_env, name)

    def reset(self):
        obs_dict = self._rs_env.reset()
        self._success_given = False
        self._prev_grasped = False
        self._prev_d_reach = None
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
        if self._grasp_hold_count >= self.N_GRASP_HOLD:
            done = True
            info["grasp_success"] = True

        # Timeout
        if self._step_count >= self.GRASP_HORIZON:
            done = True
            if "grasp_success" not in info:
                info["grasp_success"] = False

        return obs_dict, reward, done, info

    # --- helpers -----------------------------------------------------------

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

        self._prev_grasped = grasped
        self._prev_d_reach = d_reach

        if grasped:
            # ---- Stage 3: Sustained grasp ----------------------------------
            r += self.W_GRASP

        # ---- One-time stable grasp bonus -----------------------------------
        if not self._success_given and self._grasp_hold_count >= self.N_GRASP_HOLD:
            r += self.W_GRASP_SUCCESS
            self._success_given = True

        return float(r)
