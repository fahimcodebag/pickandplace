#!/usr/bin/env python3
# Last updated: 2026-07-05 13:02 +0600
"""
Spawn-curriculum grasp wrapper (random-spawn grasp retraining).

Extends the proven GraspRewardWrapper (rewards/termination unchanged) with a
per-env self-advancing SPAWN curriculum: the object's spawn region grows as
the policy masters each level, from (nearly) the fixed training pose out to
robosuite's native full-bin randomization including rotation.

Why: the fixed-spawn grasp model collapses under native spawn randomization
(28/30 grasp_handoff_failed in end-to-end eval), while the transport stage
largely generalizes. Decomposition assigns spawn randomness to THIS stage.

Level schedule (single scalar `spawn_level` in [0.1, 2.0]):
  0.1 .. 1.0   position phase: spawn box = level * native half-ranges,
               rotation fixed at 0
  1.0 .. 2.0   rotation phase: full position box, z-rotation uniform in
               +/- (level-1.0) * pi
Position first, rotation last — rotation changes the required approach pose
and is the hardest axis, so it enters only after position is mastered.

Native half-ranges (robosuite pick_place.py:412-413, table (0.39, 0.49),
0.05 margin): x +/-0.145, y +/-0.195, reference = bin1 center.

Curriculum dynamics mirror the place-training curriculum that worked:
advance on sustained mastery (>= _ADVANCE over a full _WINDOW), regress
rarely (< _REGRESS), one step at a time, fully reversible.
"""

import os
import sys
from collections import deque

import numpy as np

# Parent dir ("Decomposed state training") for grasp_env_wrapper;
# grandparent (pickandplace root) for shared modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from grasp_env_wrapper import GraspRewardWrapper


class SpawnCurriculumGraspWrapper(GraspRewardWrapper):
    """GraspRewardWrapper + growing spawn-region curriculum."""

    # REWARD OVERRIDE (local to random-spawn training; the base wrapper and
    # the fixed-spawn model are untouched). At hard spawn poses the base
    # W_GRASP_SUCCESS=20 made FAILURE more lucrative than success: a
    # flickering marginal grip (hold 3-4 steps, slip, regrasp) collects
    # W_GRASP=10/step for up to 200 steps (~150-200 total) while never
    # hitting the 5-consecutive stable check, whereas a clean success at
    # step ~40 pays ~100-130 and TERMINATES the income. Observed at spawn
    # level 0.90: success collapsed 65%->26-43% while scores stayed 85-100+
    # (failures scoring 180+). Raising the one-time success bonus makes the
    # stable grasp strictly dominate any farming trajectory.
    W_GRASP_SUCCESS = 150.0

    # NOTE: a GRASP_HORIZON 200->120 override was tried here (flooding
    # hypothesis, by analogy to the place stage) and REVERTED: it lowered the
    # peak (76%@0.80 vs 86%@0.83) and brought the decay EARLIER, with the
    # buffer far from full — flooding is not the engine of the grasp-stage
    # decay, and the cut clipped legitimate slow far-corner grasps. Base 200
    # stands.

    # Native spawn half-ranges (see module docstring).
    _X_HALF = 0.145
    _Y_HALF = 0.195

    # Level schedule
    _LEVEL_START = 0.1    # first position fraction (spawn box ~ +/-1.5-2cm)
    _LEVEL_STEP  = 0.1
    _LEVEL_MAX   = 2.0    # 1.0 = full position box; 2.0 = + full rotation

    # Advancement dynamics. Gate LOWERED 0.8 -> 0.7 after three runs showed
    # the same pattern: decay onset tracks PLATEAU DURATION (collapse began
    # after ~400-600 episodes parked at one level, in runs with different
    # incentives, horizons, and buffer fill — while the curriculum was
    # moving, training stayed healthy). A parked level feeds self-similar
    # data; TD3 overfits and drifts; success decays. 0.7 (the gate the place
    # campaign succeeded with, and below the 74-86% the policy actually
    # showed at levels 0.8-0.9) keeps the curriculum ratcheting — fresh
    # spawn distributions are what preserve the policy, not longer grinding.
    _WINDOW      = 40     # episodes per mastery judgment
    _ADVANCE     = 0.7    # success rate to level up
    _REGRESS     = 0.15   # below this -> step back down (rare)

    def __init__(self, env, curriculum=True, level=None, static_spec=None,
                 require_lift=False, align_grip=False, reward_v2=False):
        """
        Args:
            env:          raw robosuite env whose _get_placement_initializer
                          has been patched to read env._spawn_spec (see
                          make_spawn_grasp_env)
            curriculum:   True for training (self-advancing level);
                          False for eval (fixed level / static_spec)
            level:        fixed spawn level when curriculum=False
                          (default: _LEVEL_MAX = hardest)
            static_spec:  optional dict {"x": (lo,hi), "y": (lo,hi),
                          "rot": 0.0|(lo,hi)|None} that overrides the level
                          entirely (used by eval scripts for metric ranges)
            require_lift: certify grasps with a scripted lift (see
                          GraspRewardWrapper). Default False keeps every
                          pre-existing result reproducible.
            align_grip:   reward closing the fingers on a flat face rather than
                          a corner (see GraspRewardWrapper). Default False for
                          the same reason.
        """
        super().__init__(env, require_lift=require_lift,
                         align_grip=align_grip, reward_v2=reward_v2)
        self._curriculum = curriculum
        self._static_spec = static_spec
        if curriculum:
            self._spawn_level = self._LEVEL_START
        else:
            self._spawn_level = self._LEVEL_MAX if level is None else float(level)
        self._level_history = deque(maxlen=self._WINDOW)
        self._last_outcome = None     # grasp_success of the just-ended episode
        self._episode_ran = False

    # --- curriculum ---------------------------------------------------------

    def _update_curriculum(self):
        """Advance/regress spawn_level from the episode that just ended."""
        if not self._curriculum or not self._episode_ran:
            self._episode_ran = True
            return
        if self._last_outcome is None:
            return
        self._level_history.append(1.0 if self._last_outcome else 0.0)
        self._last_outcome = None
        if len(self._level_history) < self._WINDOW:
            return
        rate = float(np.mean(self._level_history))
        if rate >= self._ADVANCE and self._spawn_level < self._LEVEL_MAX:
            self._spawn_level = min(self._LEVEL_MAX,
                                    self._spawn_level + self._LEVEL_STEP)
            self._level_history.clear()
        elif rate < self._REGRESS and self._spawn_level > self._LEVEL_START:
            self._spawn_level = max(self._LEVEL_START,
                                    self._spawn_level - self._LEVEL_STEP)
            self._level_history.clear()

    def _apply_spawn_level(self):
        """Write the current level's spawn spec for the placement patch."""
        if self._static_spec is not None:
            self._rs_env._spawn_spec = self._static_spec
            return
        pos_frac = min(self._spawn_level, 1.0)
        rot_frac = max(0.0, self._spawn_level - 1.0)
        if rot_frac <= 0.0:
            rot = 0.0
        elif rot_frac >= 1.0:
            rot = None                      # robosuite: uniform full circle
        else:
            rot = (-rot_frac * np.pi, rot_frac * np.pi)
        self._rs_env._spawn_spec = {
            "x": (-pos_frac * self._X_HALF, pos_frac * self._X_HALF),
            "y": (-pos_frac * self._Y_HALF, pos_frac * self._Y_HALF),
            "rot": rot,
        }

    # --- overrides ----------------------------------------------------------

    def reset(self):
        self._update_curriculum()
        self._apply_spawn_level()
        return super().reset()

    def step(self, action):
        obs_dict, reward, done, info = super().step(action)
        if done:
            self._last_outcome = bool(info.get("grasp_success", False))
            info["spawn_level"] = round(float(self._spawn_level), 2)
        return obs_dict, reward, done, info


def make_spawn_grasp_env(env_name="PickPlace", seed=None, render=False,
                         curriculum=True, level=None, static_spec=None,
                         require_lift=False, align_grip=False,
                         reward_v2=False):
    """Create a robosuite env with grasp rewards + dynamic spawn control.

    The placement initializer is patched to read env._spawn_spec on every
    (hard) reset, so the wrapper can change the spawn region per episode.
    Returns the GymWrapper-flattened env (same obs/action interface the
    original grasp model trained on).
    """
    import robosuite as suite
    from robosuite.wrappers import GymWrapper

    env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(
            default_controller="OSC_POSE"
        ),
        has_renderer=render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,              # robosuite hard limit (wrapper enforces 200)
        reward_shaping=False,     # GraspRewardWrapper provides the reward
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )

    # Dynamic spawn patch: honors env._spawn_spec, which the curriculum
    # wrapper rewrites before each reset.
    env._spawn_spec = {"x": (0.0, 0.0), "y": (0.0, 0.0), "rot": 0.0}
    _orig_gpi = env._get_placement_initializer

    def _dynamic_placement():
        _orig_gpi()
        s = env.placement_initializer.samplers["CollisionObjectSampler"]
        spec = env._spawn_spec
        s.x_range = np.array(spec["x"])
        s.y_range = np.array(spec["y"])
        s.rotation = spec["rot"]
        # MUST both be False (matches the original fixed-spawn patch): the
        # sampler places ALL FOUR PickPlace objects in this box, and with
        # validity/boundary checks on, a small box cannot fit them
        # ("Cannot place all objects"). Overlaps are harmless — in
        # single_object_mode the non-target objects are cleared from the
        # scene right after placement.
        s.ensure_object_boundary_in_range = False
        s.ensure_valid_placement = False

    env._get_placement_initializer = _dynamic_placement

    env = SpawnCurriculumGraspWrapper(env, curriculum=curriculum,
                                      level=level, static_spec=static_spec,
                                      require_lift=require_lift,
                                      align_grip=align_grip,
                                      reward_v2=reward_v2)
    env = GymWrapper(env)
    if seed is not None:
        env.seed(seed)
    return env
