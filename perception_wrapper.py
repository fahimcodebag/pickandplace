"""Periodic camera perception for the decomposed pipeline.

Models running object-pose perception on the MCU at a FRACTION of the control
rate — the ESP32 cannot detect a tag every 50 ms, but it does not have to.

THE KEY STRUCTURE
-----------------
Of the 46-D observation, only 7 numbers are camera-derived:

    [0:3]   object position   (world)     <- camera
    [3:7]   object quaternion (world)     <- camera
    [7:10]  object->eef position          <- DERIVED
    [10:14] object->eef quaternion        <- DERIVED
    [14:46] proprioception                <- joint encoders / FK

robosuite computes the derived block as `pose_inv(T_world_eef) @ T_world_obj`
(pick_place.py:641-651) — the object pose expressed in the gripper frame. That
composition is exact, so between perception ticks we hold the WORLD-frame object
pose (zero-order hold) and RECOMPUTE the relative block every control step from
fresh proprioception. Staleness then costs only genuine object motion, and
during GRASP the object is stationary on the table — so a 250 ms-old pose of a
motionless object is very nearly free.

`mode` selects what is being measured:
    "truth"     — passthrough baseline (no perception model)
    "recompute" — ZOH world pose + per-step relative recompute (the proposal)
    "frozen"    — naive: freeze all of [0:14] between ticks (the ablation that
                  isolates how much the recompute is actually worth)
    "latch"     — recompute until the object is GRASPED, then latch the
                  object->gripper transform and PROPAGATE the world pose from
                  proprioception. Measured (Results/tag_inflight): once the
                  object is held, tag detection is 0% at BOTH the wrist and
                  fixed cameras -- the gripper occludes it. But a held object's
                  pose relative to the gripper is constant, so it does not need
                  to be seen: world = eef (*) latched_relative, and eef is
                  exact from joint encoders. "recompute" is actively wrong here
                  (it holds a stale WORLD pose while the arm carries the object
                  away from it, so the relative block diverges).

Perception can be driven either by a fitted noise model (fast, no rendering —
use this for parameter sweeps) or by a real `TagDetector` on rendered frames
(ground truth, needs offscreen GL).
"""

import numpy as np
import robosuite.utils.transform_utils as TU

OBJ_POS = slice(0, 3)
OBJ_QUAT = slice(3, 7)
REL_POS = slice(7, 10)
REL_QUAT = slice(10, 14)
EEF_POS = slice(35, 38)
EEF_QUAT = slice(38, 42)


def _compose_world(rel_pos, rel_quat, eef_pos, eef_quat):
    """Inverse of relative_block: object pose in WORLD from its pose in the
    gripper frame plus the (exact) gripper pose. Used by mode="latch" to
    propagate a held object without seeing it."""
    T_we = TU.pose2mat((np.asarray(eef_pos), np.asarray(eef_quat)))
    T_eo = TU.pose2mat((np.asarray(rel_pos), np.asarray(rel_quat)))
    T_wo = TU.pose_in_A_to_pose_in_B(T_eo, T_we)
    return TU.mat2pose(T_wo)


def relative_block(obj_pos, obj_quat, eef_pos, eef_quat):
    """Object pose in the gripper frame — robosuite's own composition."""
    obj_pose = TU.pose2mat((obj_pos, obj_quat))
    world_in_gripper = TU.pose_inv(TU.pose2mat((eef_pos, eef_quat)))
    return TU.mat2pose(TU.pose_in_A_to_pose_in_B(obj_pose, world_in_gripper))


def perturb_pose(pos, quat, noise_pos=0.0, noise_yaw=0.0, rng=None):
    """Apply isotropic position noise and a yaw perturbation to a pose."""
    rng = rng or np.random
    pos = np.asarray(pos, dtype=float).copy()
    if noise_pos > 0:
        pos += rng.normal(0.0, noise_pos, 3)
    if noise_yaw > 0:
        dy = rng.normal(0.0, noise_yaw)
        dq = TU.mat2quat(TU.euler2mat(np.array([0.0, 0.0, dy])))
        quat = TU.quat_multiply(dq, np.asarray(quat, dtype=float))
    return pos, quat


class PeriodicPerceptionWrapper:
    """Wraps a GymWrapper-flattened env and rewrites its perception block.

    Args:
        env:        env returning the flat 46-D observation.
        period:     control steps between perception updates (1 = every step,
                    5 = the proposed 4 Hz against a 20 Hz control loop).
        latency:    additional control-step delay before a measurement is
                    usable (serial + detection time, in control steps).
        noise_pos:  std-dev of position noise, metres.
        noise_yaw:  std-dev of yaw noise, radians.
        dropout:    probability a perception tick returns nothing (occlusion).
        mode:       "truth" | "recompute" | "frozen".
        detector:   optional TagDetector; when given, poses come from rendered
                    frames instead of the noise model.
    """

    def __init__(self, env, period=5, latency=0, noise_pos=0.0, noise_yaw=0.0,
                 dropout=0.0, mode="recompute", detector=None, seed=None):
        if mode not in ("truth", "recompute", "frozen", "latch"):
            raise ValueError(f"unknown mode {mode!r}")
        self.env = env
        self.period = max(1, int(period))
        self.latency = max(0, int(latency))
        self.noise_pos, self.noise_yaw = noise_pos, noise_yaw
        self.dropout = dropout
        self.mode = mode
        self.detector = detector
        self.rng = np.random.RandomState(seed)

        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)

        self._step = 0
        self._held = None        # (pos, quat) currently believed
        self._latched = None     # object->gripper transform, once grasped
        self._pending = []       # [(ready_step, pos, quat)] in-flight measurements
        self.stats = dict(ticks=0, detections=0, dropouts=0, stale_steps=0,
                          latched_at=-1)

    def _is_grasped(self):
        """True contact check from the simulator. On hardware this is the same
        signal the FSM already uses (gripper width plus a lift test), so the
        latch does not require anything the deployed system lacks."""
        e = self.env
        raw = getattr(e, "env", e)
        raw = getattr(raw, "env", raw)
        try:
            return bool(raw._check_grasp(gripper=raw.robots[0].gripper,
                                         object_geoms=raw.objects[raw.object_id]))
        except Exception:
            return False

    # --- perception ---------------------------------------------------------

    def _measure(self, obs, obs_dict=None):
        """Take one perception measurement, or None if it fails."""
        self.stats["ticks"] += 1
        if self.rng.rand() < self.dropout:
            self.stats["dropouts"] += 1
            return None
        if self.detector is not None:
            if obs_dict is None:
                return None
            r = self.detector.detect(obs_dict)
            if r is None:
                self.stats["dropouts"] += 1
                return None
            self.stats["detections"] += 1
            return r
        self.stats["detections"] += 1
        return perturb_pose(obs[OBJ_POS], obs[OBJ_QUAT],
                            self.noise_pos, self.noise_yaw, self.rng)

    def _apply(self, obs, obs_dict=None):
        """Rewrite the perception block of a flat observation in place."""
        if self.mode == "truth":
            return obs
        obs = np.asarray(obs, dtype=np.float64).copy()

        if self._step % self.period == 0:
            m = self._measure(obs, obs_dict)
            if m is not None:
                # Relative block captured AT MEASUREMENT TIME so "frozen" can
                # hold it; "recompute" recomputes and ignores it.
                rel = relative_block(m[0], m[1], obs[EEF_POS], obs[EEF_QUAT])
                self._pending.append((self._step + self.latency,
                                      np.asarray(m[0]), np.asarray(m[1]),
                                      np.asarray(rel[0]), np.asarray(rel[1])))

        # Promote any measurement whose latency has elapsed (keep the newest).
        ready = [p for p in self._pending if p[0] <= self._step]
        if ready:
            self._held = ready[-1][1:]
            self._pending = [p for p in self._pending if p[0] > self._step]

        if self._held is None:
            # No measurement yet — fall back to truth so the episode can start.
            return obs
        if self._step % self.period != 0:
            self.stats["stale_steps"] += 1

        pos, quat, rel_pos0, rel_quat0 = self._held

        if self.mode == "latch":
            if self._latched is None and self._is_grasped():
                # Latch the CURRENT relative transform, from the last good
                # measurement propagated to now.
                self._latched = relative_block(pos, quat,
                                               obs[EEF_POS], obs[EEF_QUAT])
                self.stats["latched_at"] = self._step
            if self._latched is not None:
                rp, rq = self._latched
                obs[REL_POS], obs[REL_QUAT] = rp, rq
                wp, wq = _compose_world(rp, rq, obs[EEF_POS], obs[EEF_QUAT])
                obs[OBJ_POS], obs[OBJ_QUAT] = wp, wq
                return obs
            # not yet grasped -> behave exactly like recompute
            obs[OBJ_POS], obs[OBJ_QUAT] = pos, quat
            rel_pos, rel_quat = relative_block(pos, quat,
                                               obs[EEF_POS], obs[EEF_QUAT])
            obs[REL_POS], obs[REL_QUAT] = rel_pos, rel_quat
            return obs

        obs[OBJ_POS], obs[OBJ_QUAT] = pos, quat
        if self.mode == "recompute":
            # Fresh proprioception, stale world pose — the proposal.
            rel_pos, rel_quat = relative_block(pos, quat,
                                               obs[EEF_POS], obs[EEF_QUAT])
            obs[REL_POS], obs[REL_QUAT] = rel_pos, rel_quat
        else:   # "frozen": hold the whole block, as a naive port would
            obs[REL_POS], obs[REL_QUAT] = rel_pos0, rel_quat0
        return obs

    # --- env API -------------------------------------------------------------

    def reset(self):
        self._step = 0
        self._held = None
        self._latched = None
        self._pending = []
        obs = self.env.reset()
        return self._apply(obs, self._obs_dict())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._step += 1
        out = self._apply(obs, self._obs_dict())
        info = dict(info)
        info["perception"] = dict(self.stats)
        return out, reward, done, info

    def _obs_dict(self):
        """Rendered observation dict, needed only when a detector is attached."""
        if self.detector is None:
            return None
        base = self.env
        while hasattr(base, "env"):
            base = base.env
        return base._get_observations()

    def __getattr__(self, name):
        return getattr(self.__dict__["env"], name)
