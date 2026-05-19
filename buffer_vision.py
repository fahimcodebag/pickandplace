"""Replay buffer for vision-based RL with HER and PER.

Stores images as uint8 (0-255) to save ~4x memory compared to float32.
Conversion to float32 [0, 1] happens at sampling time on GPU.

Image shape: (C, H, W) stored per transition.
"""

import numpy as np
import os


# ---------------------------------------------------------------------------
# SumTree for PER
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf holds a priority and parent nodes hold sums."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write_idx = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self):
        return self.tree[0]

    def add(self, priority):
        tree_idx = self.write_idx + self.capacity - 1
        self.update(tree_idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s):
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx


# ---------------------------------------------------------------------------
# Vision HER + PER Replay Buffer
# ---------------------------------------------------------------------------

class VisionHERReplayBuffer:
    """Replay buffer for image observations with HER + PER.

    Images stored as uint8 to minimize memory. With 84×84×3 images:
      - 500K transitions ≈ 10.6 GB (uint8)
      - Same in float32 would be ~42 GB
    """

    PER_E = 0.01
    PER_A = 0.6
    PER_B = 0.4
    PER_B_INC = 1e-6
    HER_K = 4
    GOAL_THRESHOLD = 0.05

    def __init__(self, max_size, img_shape, n_actions, goal_dim=3):
        """
        Args:
            max_size: maximum buffer capacity
            img_shape: (C, H, W) image shape
            n_actions: action dimensionality
            goal_dim: goal vector dimensionality
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.img_shape = img_shape
        self.n_actions = n_actions
        self.goal_dim = goal_dim

        # Store images as uint8 (0-255) to save memory
        self.img_memory = np.zeros((max_size, *img_shape), dtype=np.uint8)
        self.next_img_memory = np.zeros((max_size, *img_shape), dtype=np.uint8)
        self.action_memory = np.zeros((max_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(max_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_size, dtype=bool)
        self.goal_memory = np.zeros((max_size, goal_dim), dtype=np.float32)

        self.tree = SumTree(max_size)
        self._max_priority = 1.0

    # ------------------------------------------------------------------
    # Low-level storage
    # ------------------------------------------------------------------

    def _store_single(self, img, action, reward, next_img, done, goal):
        idx = self.mem_cntr % self.mem_size
        self.img_memory[idx] = img       # uint8
        self.next_img_memory[idx] = next_img
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        self.goal_memory[idx] = goal
        self.tree.add(self._max_priority ** self.PER_A)
        self.mem_cntr += 1

    # ------------------------------------------------------------------
    # Episode-level storage with HER relabeling
    # ------------------------------------------------------------------

    def store_episode(self, episode, compute_reward_fn):
        """Store episode with HER relabeling (future strategy).

        Args:
            episode: list of dicts with keys:
                img, action, reward, next_img, done, goal, achieved_goal
            compute_reward_fn: callable(achieved_goal, desired_goal) -> float
        """
        T_ep = len(episode)
        if T_ep == 0:
            return

        # 1. Store original transitions
        for tr in episode:
            self._store_single(
                tr['img'], tr['action'], tr['reward'],
                tr['next_img'], tr['done'], tr['goal'],
            )

        # 2. HER relabeling – "future" strategy
        for t in range(T_ep):
            tr = episode[t]
            future_start = t + 1
            if future_start >= T_ep:
                continue
            n_samples = min(self.HER_K, T_ep - future_start)
            future_indices = np.random.choice(
                range(future_start, T_ep), size=n_samples, replace=False,
            )
            for j in future_indices:
                new_goal = episode[j]['achieved_goal']
                new_reward = compute_reward_fn(tr['achieved_goal'], new_goal)
                self._store_single(
                    tr['img'], tr['action'], new_reward,
                    tr['next_img'], tr['done'], new_goal,
                )

    # ------------------------------------------------------------------
    # Direct storage
    # ------------------------------------------------------------------

    def store_transition(self, img, action, reward, next_img, done, goal):
        self._store_single(img, action, reward, next_img, done, goal)

    # ------------------------------------------------------------------
    # PER sampling (returns uint8 images, caller converts to float on GPU)
    # ------------------------------------------------------------------

    def sample_buffer_per(self, batch_size):
        """Returns (imgs, actions, rewards, next_imgs, dones,
                    goals, tree_indices, is_weights).

        Images are returned as uint8 numpy arrays.
        """
        indices = np.empty(batch_size, dtype=np.int64)
        tree_indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)

        segment = self.tree.total / batch_size
        self.PER_B = min(1.0, self.PER_B + self.PER_B_INC)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            t_idx, prio, d_idx = self.tree.get(s)
            d_idx = d_idx % min(self.mem_cntr, self.mem_size)
            tree_indices[i] = t_idx
            indices[i] = d_idx
            priorities[i] = max(prio, 1e-8)

        probs = priorities / self.tree.total
        n = min(self.mem_cntr, self.mem_size)
        is_weights = (n * probs) ** (-self.PER_B)
        is_weights /= is_weights.max()

        return (
            self.img_memory[indices],          # uint8
            self.action_memory[indices],
            self.reward_memory[indices],
            self.next_img_memory[indices],     # uint8
            self.terminal_memory[indices],
            self.goal_memory[indices],
            tree_indices,
            is_weights.astype(np.float32),
        )

    def update_priorities(self, tree_indices, td_errors):
        for t_idx, td in zip(tree_indices, td_errors):
            p = (abs(td) + self.PER_E) ** self.PER_A
            self.tree.update(t_idx, p)
            self._max_priority = max(self._max_priority, abs(td) + self.PER_E)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path='./checkpoints/td3_vision/replay_buffer.npz'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            img_memory=self.img_memory[:min(self.mem_cntr, self.mem_size)],
            next_img_memory=self.next_img_memory[:min(self.mem_cntr, self.mem_size)],
            action_memory=self.action_memory,
            reward_memory=self.reward_memory,
            terminal_memory=self.terminal_memory,
            goal_memory=self.goal_memory,
            mem_cntr=np.array([self.mem_cntr]),
            tree_data=self.tree.tree,
            max_priority=np.array([self._max_priority]),
            per_b=np.array([self.PER_B]),
        )

    def load(self, path='./checkpoints/td3_vision/replay_buffer.npz'):
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            n_stored = min(int(data['mem_cntr'][0]), self.mem_size)
            self.img_memory[:n_stored] = data['img_memory'][:n_stored]
            self.next_img_memory[:n_stored] = data['next_img_memory'][:n_stored]
            self.action_memory = data['action_memory']
            self.reward_memory = data['reward_memory']
            self.terminal_memory = data['terminal_memory']
            self.goal_memory = data['goal_memory']
            self.mem_cntr = int(data['mem_cntr'][0])
            if 'tree_data' in data:
                self.tree.tree = data['tree_data']
                self.tree.n_entries = min(self.mem_cntr, self.mem_size)
                self.tree.write_idx = self.mem_cntr % self.mem_size
                self._max_priority = float(data['max_priority'][0])
                self.PER_B = float(data['per_b'][0])
            else:
                n = min(self.mem_cntr, self.mem_size)
                for i in range(n):
                    t_idx = i + self.tree.capacity - 1
                    self.tree.update(t_idx, 1.0)
                self.tree.n_entries = n
                self.tree.write_idx = self.mem_cntr % self.mem_size
        except Exception as e:
            print(f"WARNING: Failed to load replay buffer ({e}). Starting fresh.")
            return False
        return True
