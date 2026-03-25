"""Utility classes for enhanced RL training.

Contains:
  - OUNoise: Ornstein-Uhlenbeck temporally correlated exploration noise (#1)
  - NoiseScheduler: linear noise decay over training (#7)
  - RewardNormalizer: running mean/std reward normalization (#19)
  - compute_reward: sparse goal-conditioned reward for HER (#8)
  - potential_reward_shaping: potential-based reward shaping (#22)
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Ornstein-Uhlenbeck Noise
# ---------------------------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process for temporally correlated exploration."""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2, dt=1e-2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.size = size
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = (self.theta * (self.mu - self.state) * self.dt +
              self.sigma * np.sqrt(self.dt) * np.random.randn(self.size))
        self.state += dx
        return self.state.copy()

    def sample_batch(self, n):
        """Sample n noise vectors (one per environment)."""
        return np.array([self.sample() for _ in range(n)])


# ---------------------------------------------------------------------------
# 7. Noise Scheduler (linear decay)
# ---------------------------------------------------------------------------

class NoiseScheduler:
    """Linearly decays a noise scale from start to end over total_steps."""

    def __init__(self, noise_start=0.3, noise_end=0.05, total_steps=2_000_000):
        self.noise_start = noise_start
        self.noise_end = noise_end
        self.total_steps = total_steps
        self.current_step = 0

    def step(self, n=1):
        self.current_step = min(self.current_step + n, self.total_steps)

    @property
    def scale(self):
        frac = min(self.current_step / self.total_steps, 1.0)
        return self.noise_start + frac * (self.noise_end - self.noise_start)

    def state_dict(self):
        return {'current_step': self.current_step}

    def load_state_dict(self, d):
        self.current_step = d['current_step']


# ---------------------------------------------------------------------------
# 19. Reward Normalizer (running mean/std via Welford's algorithm)
# ---------------------------------------------------------------------------

class RewardNormalizer:
    """Running mean/std reward normalisation."""

    def __init__(self, clip=10.0):
        self.clip = clip
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, rewards):
        rewards = np.atleast_1d(np.asarray(rewards, dtype=np.float64))
        batch_mean = rewards.mean()
        batch_var = rewards.var()
        batch_count = len(rewards)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

    def normalize(self, reward):
        std = np.sqrt(self.var + 1e-8)
        return np.clip((reward - self.mean) / std, -self.clip, self.clip)

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


# ---------------------------------------------------------------------------
# 8 (helper). Sparse goal-conditioned reward for HER
# ---------------------------------------------------------------------------

def compute_reward(achieved_goal, desired_goal, threshold=0.05):
    """Dense reward: negative distance to goal, 0 when within threshold."""
    achieved_goal = np.asarray(achieved_goal)
    desired_goal = np.asarray(desired_goal)
    dist = np.linalg.norm(achieved_goal - desired_goal, axis=-1)
    return np.where(dist < threshold, 0.0, -dist).astype(np.float32)


# ---------------------------------------------------------------------------
# 22. Potential-based reward shaping
# ---------------------------------------------------------------------------

def potential_reward_shaping(obs, next_obs, goal, gamma=0.99):
    """Potential-based shaping: gamma * Phi(s') - Phi(s).

    Potential Phi(s) = -||bread_pos - goal||.
    obs[..., 0:3] is the bread position (from robosuite GymWrapper ordering).
    """
    bread_pos = np.asarray(obs[..., 0:3])
    next_bread_pos = np.asarray(next_obs[..., 0:3])
    goal = np.asarray(goal)

    phi_s = -np.linalg.norm(bread_pos - goal, axis=-1)
    phi_s_next = -np.linalg.norm(next_bread_pos - goal, axis=-1)

    return gamma * phi_s_next - phi_s


# ---------------------------------------------------------------------------
# Staged reward for training (reach → grasp → lift → place)
# ---------------------------------------------------------------------------
# Observation indices (GymWrapper: object-state first, then proprio)
#   [0:3]   bread_pos
#   [7:10]  bread_to_eef_pos
#   [35:38] eef_pos
#   [42:44] gripper_qpos

TABLE_Z = 0.845   # bread resting height on table
GRIP_OPEN = 0.04   # gripper_qpos[0]-gripper_qpos[1] when fully open


def compute_staged_reward(obs, next_obs, goal):
    """Staged reward with explicit bonuses for each pick-and-place phase.

    Stage 1 – REACH:  reward for moving end-effector toward bread
    Stage 2 – GRASP:  bonus when near bread and gripper is closing
    Stage 3 – LIFT:   bonus for lifting bread above the table
    Stage 4 – PLACE:  reward for moving bread toward the goal bin
    """
    eef = next_obs[35:38]
    bread = next_obs[0:3]
    gripper = next_obs[42:44]

    bread_height = bread[2] - TABLE_Z
    dist_eef_bread = np.linalg.norm(eef - bread)
    dist_bread_goal = np.linalg.norm(bread - goal)
    gripper_open = gripper[0] - gripper[1]   # ~0.04 open, ~0.01 closed

    reward = 0.0

    # Stage 1: REACH – always active
    reward += -dist_eef_bread

    # Stage 2: GRASP – near bread and closing gripper
    if dist_eef_bread < 0.05:
        reward += 0.5                                         # proximity bonus
        grasp_bonus = max(0.0, (GRIP_OPEN - gripper_open)) * 10.0
        reward += grasp_bonus                                 # up to ~0.3

    # Stage 3: LIFT – bread above table
    if bread_height > 0.02:
        reward += 2.0 + min(bread_height, 0.2) * 5.0         # up to 3.0

    # Stage 4: PLACE – bread toward goal (only when lifted)
    if bread_height > 0.02:
        reward += max(0.0, 0.5 - dist_bread_goal) * 5.0      # up to 2.5

    # Success bonus
    if dist_bread_goal < 0.05:
        reward += 10.0

    return float(reward)
