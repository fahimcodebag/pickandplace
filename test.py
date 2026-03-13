import time
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3 import Agent


# ---------------------------------------------------------------------------
# Inline curriculum reward wrapper (same as training, avoids importing
# train_vectorized which sets MUJOCO_GL=egl and breaks on-screen rendering)
# ---------------------------------------------------------------------------

class CurriculumRewardWrapper:
    """Must match the wrapper in train_vectorized.py exactly."""
    W_REACH       = 1.0
    W_GRIP_CLOSE  = 0.5
    W_GRASP       = 10.0
    W_LIFT        = 3.0
    W_HOVER       = 0.5
    W_SUCCESS     = 50.0

    P_IDLE        = -0.4
    P_DROP        = -5.0
    P_AWAY        = -0.7

    _REACH_SCALE  = 0.30
    _GRIP_RANGE   = 0.06
    _HOVER_SCALE  = 0.25
    _LIFT_CEIL    = 0.12

    def __init__(self, env):
        self._rs_env = env
        self._init_z = None
        self._success_given = False
        self._target_bin = None
        self._prev_grasped = False
        self._prev_d_reach = None

    def __getattr__(self, name):
        return getattr(self._rs_env, name)

    def reset(self):
        obs_dict = self._rs_env.reset()
        self._success_given = False
        self._target_bin = None
        self._prev_grasped = False
        self._prev_d_reach = None
        try:
            pos = self._get_obj_pos(obs_dict)
            self._init_z = float(pos[2]) if pos is not None else 0.82
        except Exception:
            self._init_z = 0.82
        return obs_dict

    def step(self, action):
        obs_dict, _, done, info = self._rs_env.step(action)
        reward = self._curriculum_reward(obs_dict)
        return obs_dict, reward, done, info

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

    def _get_target_bin_pos(self):
        if self._target_bin is not None:
            return self._target_bin
        try:
            name = self._get_target_obj_name()
            obj_idx = self._rs_env.object_to_id[name.lower()]
            self._target_bin = np.array(
                self._rs_env.target_bin_placements[obj_idx]
            )
            return self._target_bin
        except Exception:
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
        try:
            qpos = np.array(obs_dict["robot0_gripper_qpos"])
            return float(np.clip(abs(qpos[0]) / 0.04, 0.0, 1.0))
        except Exception:
            return 0.5

    def _curriculum_reward(self, obs_dict):
        r = 0.0
        eef_pos = np.array(obs_dict["robot0_eef_pos"])
        obj_pos = self._get_obj_pos(obs_dict)
        if obj_pos is None:
            return r

        d_reach = np.linalg.norm(eef_pos - obj_pos)
        r += self.W_REACH * max(0.0, 1.0 - d_reach / self._REACH_SCALE)

        if d_reach < self._GRIP_RANGE:
            aperture = self._gripper_aperture(obs_dict)
            r += self.W_GRIP_CLOSE * (1.0 - aperture)

        grasped = self._is_grasped()

        r += self.P_IDLE

        if self._prev_grasped and not grasped:
            r += self.P_DROP

        if not grasped and self._prev_d_reach is not None:
            if d_reach > self._prev_d_reach + 0.005:
                r += self.P_AWAY

        self._prev_grasped = grasped
        self._prev_d_reach = d_reach

        if grasped:
            r += self.W_GRASP
            init_z = self._init_z if self._init_z is not None else 0.82
            lift = max(0.0, obj_pos[2] - init_z)
            r += self.W_LIFT * min(1.0, lift / self._LIFT_CEIL)

            target = self._get_target_bin_pos()
            if target is not None:
                d_place = np.linalg.norm(obj_pos[:2] - target[:2])
                r += self.W_HOVER * max(0.0, 1.0 - d_place / self._HOVER_SCALE)

        if not self._success_given:
            try:
                if self._rs_env._check_success():
                    r += self.W_SUCCESS
                    self._success_given = True
            except Exception:
                pass

        return float(r)


if __name__ == '__main__':
    env_name = "PickPlace"

    # Build raw robosuite env — must match training config exactly
    rs_env = suite.make(
        env_name,
        robots="Panda",
        controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=500,
        reward_shaping=False,
        control_freq=20,
        single_object_mode=2,
        object_type="bread",
    )
    # Fix object spawn to a constant position
    # Fix object spawn (survives hard_reset)
    _orig_gpi = rs_env._get_placement_initializer
    def _fixed_placement():
        _orig_gpi()
        s = rs_env.placement_initializer.samplers["CollisionObjectSampler"]
        s.x_range = np.array([0.0, 0.0])
        s.y_range = np.array([0.0, 0.0])
        s.rotation = 0.0
        s.ensure_object_boundary_in_range = False
        s.ensure_valid_placement = False
    rs_env._get_placement_initializer = _fixed_placement

    # Wrap with the same curriculum reward used during training
    curriculum_env = CurriculumRewardWrapper(rs_env)
    env = GymWrapper(curriculum_env)

    # Hyperparameters — must match train_vectorized.py exactly
    agent = Agent(
        alpha=0.0005,
        beta=0.0005,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=512,
        layer2_size=256,
        batch_size=1024,
    )

    print("Loading trained models from ./checkpoints/td3/ ...")
    agent.load_models()
    print("Models loaded.\n")

    n_games = 10
    successes = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        step = 0
        success = False

        print(f"Episode {i + 1}/{n_games}", end=" ", flush=True)

        while not done:
            action = agent.choose_action(observation, validation=True)
            observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            step += 1
            time.sleep(0.02)   # slow down for visibility

        # Check task success via robosuite API
        try:
            success = rs_env._check_success()
        except Exception:
            success = False

        if success:
            successes += 1

        status = "SUCCESS" if success else "failed"
        print(f"| Steps: {step:3d} | Score: {score:8.2f} | {status}")

    print(f"\nResult: {successes}/{n_games} successful placements")
    env.close()

    # Hyperparameters — must match train_vectorized.py exactly
    agent = Agent(
        alpha=0.0005,
        beta=0.0005,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=512,
        layer2_size=256,
        batch_size=1024,
    )

    print("Loading trained models from ./checkpoints/td3/ ...")
    agent.load_models()
    print("Models loaded.\n")

    n_games = 10
    successes = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        step = 0
        success = False

        print(f"Episode {i + 1}/{n_games}", end=" ", flush=True)

        while not done:
            action = agent.choose_action(observation, validation=True)
            observation, reward, done, info = env.step(action)
            env.render()
            score += reward
            step += 1
            time.sleep(0.02)   # slow down for visibility

        # Check task success via robosuite API
        try:
            success = rs_env._check_success()
        except Exception:
            success = False

        if success:
            successes += 1

        status = "SUCCESS" if success else "failed"
        print(f"| Steps: {step:3d} | Score: {score:8.2f} | {status}")

    print(f"\nResult: {successes}/{n_games} successful placements")
    env.close()