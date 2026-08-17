# Last updated: 2026-07-04 13:00 +0600
import numpy as np
import os


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
        """Sample one leaf. Returns (tree_idx, priority, data_idx)."""
        tree_idx = self._retrieve(0, s)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], data_idx


class ReplayBuffer:
    """Prioritized Experience Replay buffer using a SumTree.

    Falls back to uniform sampling when called via the old API
    (sample_buffer returns 5 arrays), and supports PER via
    sample_buffer_per which also returns indices and IS weights.
    """

    PER_E  = 0.01   # small constant to ensure non-zero priority
    PER_A  = 0.4    # how much prioritization to use (0 = uniform, 1 = full).
                    #   Lowered 0.6 -> 0.4 (place training): at 0.6, PER acted
                    #   as a collapse AMPLIFIER — once a policy dip produced a
                    #   burst of large-TD failure terminals (timeout_held -37s),
                    #   they were oversampled, dragging critic Q-values down for
                    #   all transport states, causing more holds, more failure
                    #   terminals, stronger priorities. Observed repeatedly as
                    #   fast/deep/sticky collapses right after curriculum peaks
                    #   (e.g. 84% at frac 0.85 -> 0% within ~250 episodes) while
                    #   climbs stayed slow. 0.4 keeps success transitions
                    #   replayed preferentially but flattens the failure-burst
                    #   feedback loop.
    PER_B  = 0.4    # initial importance-sampling exponent
    PER_B_INC = 1e-6  # anneal beta toward 1

    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        if isinstance(input_shape, int):
            input_shape = (input_shape,)

        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        self.tree = SumTree(max_size)
        self._max_priority = 1.0

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        # New transitions get max priority so they are sampled at least once
        self.tree.add(self._max_priority ** self.PER_A)
        self.mem_cntr += 1

    # --- Uniform sampling (backward compatible) ----------------------------
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        return (self.state_memory[batch], self.action_memory[batch],
                self.reward_memory[batch], self.new_state_memory[batch],
                self.terminal_memory[batch])

    # --- Prioritized sampling ----------------------------------------------
    def sample_buffer_per(self, batch_size):
        """Returns (states, actions, rewards, next_states, dones,
                    tree_indices, is_weights)."""
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
            # Guard against stale indices beyond current fill
            d_idx = d_idx % min(self.mem_cntr, self.mem_size)
            tree_indices[i] = t_idx
            indices[i] = d_idx
            priorities[i] = max(prio, 1e-8)

        probs = priorities / self.tree.total
        n = min(self.mem_cntr, self.mem_size)
        is_weights = (n * probs) ** (-self.PER_B)
        is_weights /= is_weights.max()

        return (self.state_memory[indices], self.action_memory[indices],
                self.reward_memory[indices], self.new_state_memory[indices],
                self.terminal_memory[indices], tree_indices,
                is_weights.astype(np.float32))

    def update_priorities(self, tree_indices, td_errors):
        """Update priorities for sampled transitions."""
        for t_idx, td in zip(tree_indices, td_errors):
            p = (abs(td) + self.PER_E) ** self.PER_A
            self.tree.update(t_idx, p)
            self._max_priority = max(self._max_priority, abs(td) + self.PER_E)

    # --- Save / Load -------------------------------------------------------
    def save(self, path='./checkpoints/td3/replay_buffer.npz'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(
            path,
            state_memory=self.state_memory,
            new_state_memory=self.new_state_memory,
            action_memory=self.action_memory,
            reward_memory=self.reward_memory,
            terminal_memory=self.terminal_memory,
            mem_cntr=np.array([self.mem_cntr]),
            tree_data=self.tree.tree,
            max_priority=np.array([self._max_priority]),
            per_b=np.array([self.PER_B]),
        )

    def load(self, path='./checkpoints/td3/replay_buffer.npz'):
        if not os.path.exists(path):
            return False
        try:
            data = np.load(path)
            self.state_memory     = data['state_memory']
            self.new_state_memory = data['new_state_memory']
            self.action_memory    = data['action_memory']
            self.reward_memory    = data['reward_memory']
            self.terminal_memory  = data['terminal_memory']
            self.mem_cntr         = int(data['mem_cntr'][0])
            # Restore tree if saved with PER, otherwise rebuild uniform
            if 'tree_data' in data:
                self.tree.tree = data['tree_data']
                self.tree.n_entries = min(self.mem_cntr, self.mem_size)
                self.tree.write_idx = self.mem_cntr % self.mem_size
                self._max_priority = float(data['max_priority'][0])
                self.PER_B = float(data['per_b'][0])
            else:
                # Old buffer without PER — assign uniform max priority
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