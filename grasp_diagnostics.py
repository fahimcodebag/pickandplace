"""Episode-level diagnostics for the grasp stage — the §4 done-reason tally.

The place campaign was diagnosed with per-episode done-reason tallies and drop
diagnostics; the grasp wrapper emits only `grasp_success`, so identical scores
conflate failure modes that need different levers. This wrapper adds the
missing instrumentation.

STRICTLY ADDITIVE: reward, termination, action, and observation pass through
untouched, so runs logged with this wrapper stay comparable to every number
already in the thesis. It only reads state and writes extra `info` keys.

Done reasons emitted
--------------------
  success             stable grasp held N_GRASP_HOLD consecutive steps
  timeout_flicker     grasp confirmed on some steps but never held —
                      the MARGINAL-GRIP failure §9.2 Hypothesis 1 identified
                      as out-earning a clean success at hard spawn poses
  timeout_touched     reached the object (within REACH_NEAR) but never a
                      confirmed grasp
  timeout_no_reach    never got within REACH_NEAR of the object

Also recorded per episode: spawn pose (for success-vs-spawn-region heatmaps),
time to first grasp contact, longest consecutive grasp run, and closest
approach — enough to plot where in the spawn box the policy actually fails.
"""

import numpy as np

REACH_NEAR = 0.05          # m; "reached the object" threshold for the tally


class GraspDiagnosticsWrapper:
    def __init__(self, env):
        self.env = env
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self._reset_episode()

    def _reset_episode(self):
        self._steps = 0
        self._grasp_steps = 0
        self._first_grasp_step = -1
        self._run = 0
        self._max_run = 0
        self._min_reach = np.inf
        self._spawn = (np.nan, np.nan, np.nan)

    # --- state probes --------------------------------------------------------

    def _obj_pose(self):
        base = self._base()
        bid = base.sim.model.body_name2id(self._obj_body())
        pos = np.array(base.sim.data.body_xpos[bid])
        quat = np.array(base.sim.data.body_xquat[bid])       # wxyz
        yaw = np.arctan2(2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                         1 - 2 * (quat[2] ** 2 + quat[3] ** 2))
        return pos, yaw

    def _obj_body(self):
        try:
            return f"{self.env.obj_to_use}_main"
        except Exception:
            return "Bread_main"

    def _base(self):
        e = self.env
        while not hasattr(e, "sim") and hasattr(e, "env"):
            e = e.env
        return e

    def _eef(self):
        base = self._base()
        return np.array(base.sim.data.site_xpos[
            base.sim.model.site_name2id("gripper0_grip_site")])

    def _is_grasped(self):
        """Find `_is_grasped` anywhere down the wrapper chain (GymWrapper sits
        between this wrapper and GraspRewardWrapper)."""
        e = self.env
        for _ in range(8):
            fn = getattr(e, "_is_grasped", None)
            if callable(fn):
                try:
                    return bool(fn())
                except Exception:
                    return False
            e = getattr(e, "env", None) or getattr(e, "_rs_env", None)
            if e is None:
                break
        return False

    # --- env API -------------------------------------------------------------

    def reset(self):
        obs = self.env.reset()
        self._reset_episode()
        try:
            pos, yaw = self._obj_pose()
            self._spawn = (float(pos[0]), float(pos[1]), float(yaw))
        except Exception:
            pass
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._steps += 1

        if self._is_grasped():
            self._grasp_steps += 1
            self._run += 1
            self._max_run = max(self._max_run, self._run)
            if self._first_grasp_step < 0:
                self._first_grasp_step = self._steps
        else:
            self._run = 0

        try:
            pos, _ = self._obj_pose()
            self._min_reach = min(self._min_reach,
                                  float(np.linalg.norm(self._eef() - pos)))
        except Exception:
            pass

        if done:
            info = dict(info)
            success = bool(info.get("grasp_success", False))
            if success:
                reason = "success"
            elif self._grasp_steps > 0:
                reason = "timeout_flicker"
            elif np.isfinite(self._min_reach) and self._min_reach <= REACH_NEAR:
                reason = "timeout_touched"
            else:
                reason = "timeout_no_reach"
            info.update({
                "done_reason": reason,
                "spawn_x": self._spawn[0],
                "spawn_y": self._spawn[1],
                "spawn_yaw": self._spawn[2],
                "ep_steps": self._steps,
                "grasp_steps": self._grasp_steps,
                "first_grasp_step": self._first_grasp_step,
                "max_grasp_run": self._max_run,
                "min_reach_dist": (float(self._min_reach)
                                   if np.isfinite(self._min_reach) else -1.0),
            })
        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.__dict__["env"], name)


DONE_REASONS = ("success", "timeout_flicker", "timeout_touched",
                "timeout_no_reach")
