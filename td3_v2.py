"""Enhanced TD3 agent – goal-conditioned with all 10 RL enhancements.

Features integrated:
  #1  OU noise (temporally correlated exploration)
  #7  Scheduled noise decay (linear)
  #8  HER support (goal-conditioned policy, buffer handled externally)
  #13 LR scheduling (CosineAnnealingLR)
  #14 Gradient clipping (max_norm)
  #16 Multiple gradient steps per learn() call
  #17 4-critic ensemble (min over all for target value)
  #19 Reward normalisation (running mean/std)
  #22 Potential-based reward shaping (applied in training loop)
"""

import os
import json
import numpy as np
import torch as T
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import CosineAnnealingLR

from buffer_her import HERReplayBuffer
from networks_v2 import ActorNetwork, CriticNetwork
from utils_rl import OUNoise, NoiseScheduler, RewardNormalizer


class Agent:
    def __init__(
        self,
        alpha,
        beta,
        obs_dims,
        goal_dim,
        tau,
        env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,
        n_actions=7,
        max_size=2_000_000,
        layer1_size=1024,
        layer2_size=512,
        batch_size=2048,
        noise_start=0.3,
        noise_end=0.05,
        n_critics=4,
        n_gradient_steps=4,
        max_grad_norm=1.0,
        lr_total_steps=2_000_000,
        chkpt_dir='./checkpoints/td3_v2',
    ):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.chkpt_dir = chkpt_dir
        self.buf_path = os.path.join(chkpt_dir, 'replay_buffer.npz')
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.goal_dim = goal_dim
        self.update_actor_iter = update_actor_interval
        self.n_critics = n_critics
        self.n_gradient_steps = n_gradient_steps
        self.max_grad_norm = max_grad_norm
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        os.makedirs(chkpt_dir, exist_ok=True)

        # ---- Replay buffer (HER + PER) ----
        self.memory = HERReplayBuffer(max_size, obs_dims, n_actions, goal_dim)

        # ---- OU Noise + scheduler (#1, #7) ----
        self.ou_noise = OUNoise(n_actions, sigma=noise_start)
        self.noise_scheduler = NoiseScheduler(noise_start, noise_end, lr_total_steps)

        # ---- Reward normaliser (#19) ----
        self.reward_normalizer = RewardNormalizer()

        # ---- Networks ----
        obs_dim = obs_dims[0] if isinstance(obs_dims, tuple) else obs_dims

        self.actor = ActorNetwork(
            obs_dim, goal_dim, layer1_size, layer2_size,
            n_actions, 'actor', chkpt_dir, alpha,
        )
        self.target_actor = ActorNetwork(
            obs_dim, goal_dim, layer1_size, layer2_size,
            n_actions, 'target_actor', chkpt_dir, alpha,
        )

        # 4 critics + 4 targets (#17)
        self.critics = nn_module_list_helper(
            n_critics, obs_dim, goal_dim, n_actions,
            layer1_size, layer2_size, 'critic', chkpt_dir, beta,
        )
        self.target_critics = nn_module_list_helper(
            n_critics, obs_dim, goal_dim, n_actions,
            layer1_size, layer2_size, 'target_critic', chkpt_dir, beta,
        )

        # ---- LR schedulers (#13) ----
        self.actor_scheduler = CosineAnnealingLR(
            self.actor.optimizer, T_max=lr_total_steps, eta_min=1e-6,
        )
        self.critic_schedulers = [
            CosineAnnealingLR(c.optimizer, T_max=lr_total_steps, eta_min=1e-6)
            for c in self.critics
        ]

        # ---- Hard-copy params to targets ----
        self.update_network_parameters(tau=1)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, observation, goal, validation=False):
        if self.time_step < self.warmup and not validation:
            action = np.random.uniform(
                self.min_action[0], self.max_action[0],
                size=(self.n_actions,),
            )
        else:
            state = T.tensor(observation, dtype=T.float).unsqueeze(0).to(self.device)
            g = T.tensor(goal, dtype=T.float).unsqueeze(0).to(self.device)
            with T.no_grad():
                action = self.actor(state, g).squeeze(0).cpu().numpy()

        if not validation:
            noise = self.ou_noise.sample() * self.noise_scheduler.scale
            action = action + noise
            action = np.clip(action, self.min_action[0], self.max_action[0])
            self.time_step += 1
            self.noise_scheduler.step()

        return action

    def choose_action_batch(self, observations, goals):
        n_envs = observations.shape[0]
        if self.time_step < self.warmup:
            actions = np.random.uniform(
                self.min_action[0], self.max_action[0],
                size=(n_envs, self.n_actions),
            )
        else:
            states = T.tensor(observations, dtype=T.float).to(self.device)
            g = T.tensor(goals, dtype=T.float).to(self.device)
            with T.no_grad():
                actions = self.actor(states, g).cpu().numpy()

        noise = self.ou_noise.sample_batch(n_envs) * self.noise_scheduler.scale
        actions = actions + noise
        actions = np.clip(actions, self.min_action[0], self.max_action[0])
        self.time_step += n_envs
        self.noise_scheduler.step(n_envs)
        return actions

    # ------------------------------------------------------------------
    # Learning (#14 gradient clipping, #16 multi-step, #17 4 critics)
    # ------------------------------------------------------------------

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return
        for _ in range(self.n_gradient_steps):
            self._gradient_step()

    def _gradient_step(self):
        (state, action, reward, next_state, done,
         goal, tree_idx, is_weights) = self.memory.sample_buffer_per(self.batch_size)

        # Reward normalisation (#19)
        self.reward_normalizer.update(reward)
        reward = self.reward_normalizer.normalize(reward)

        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        goal = T.tensor(goal, dtype=T.float).to(self.device)
        is_weights = T.tensor(is_weights, dtype=T.float).to(self.device)

        # --- Target value ---
        with T.no_grad():
            target_actions = self.target_actor(next_state, goal)
            noise = T.clamp(T.randn_like(target_actions) * 0.2, -0.5, 0.5)
            target_actions = T.clamp(
                target_actions + noise,
                self.min_action[0], self.max_action[0],
            )
            # Min across all target critics (#17)
            next_qs = []
            for tc in self.target_critics:
                nq = tc(next_state, goal, target_actions)
                nq[done] = 0.0
                next_qs.append(nq.view(-1))
            next_critic_value = T.min(T.stack(next_qs), dim=0).values

            target = reward + self.gamma * next_critic_value
            target = target.view(self.batch_size, 1)

        # --- Update critics ---
        # PER priorities from first critic
        q1 = self.critics[0](state, goal, action)
        td_errors = (target - q1).detach().squeeze().cpu().numpy()
        self.memory.update_priorities(tree_idx, td_errors)

        # Zero grads
        for c in self.critics:
            c.optimizer.zero_grad()

        # Compute joint loss
        w = is_weights.view(-1, 1)
        critic_loss = T.tensor(0.0, device=self.device)
        for c in self.critics:
            q = c(state, goal, action)
            critic_loss = critic_loss + (w * (target - q).pow(2)).mean()

        critic_loss.backward()

        # Gradient clipping + step (#14)
        for c in self.critics:
            nn_utils.clip_grad_norm_(c.parameters(), self.max_grad_norm)
            c.optimizer.step()

        # LR scheduler step (#13)
        for sched in self.critic_schedulers:
            sched.step()

        self.learn_step_cntr += 1

        # --- Delayed actor update ---
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q = self.critics[0](state, goal, self.actor(state, goal))
        actor_loss = -T.mean(actor_q)
        actor_loss.backward()
        nn_utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor.optimizer.step()
        self.actor_scheduler.step()

        self.update_network_parameters()

    # ------------------------------------------------------------------
    # Soft update
    # ------------------------------------------------------------------

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Actor
        _soft_update(self.actor, self.target_actor, tau)

        # Critics
        for c, tc in zip(self.critics, self.target_critics):
            _soft_update(c, tc, tau)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        for c in self.critics:
            c.save_checkpoint()
        for tc in self.target_critics:
            tc.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        for c in self.critics:
            c.load_checkpoint()
        for tc in self.target_critics:
            tc.load_checkpoint()

    def save_best_models(self):
        self.actor.save_best()
        self.target_actor.save_best()
        for c in self.critics:
            c.save_best()
        for tc in self.target_critics:
            tc.save_best()

    def load_best_models(self):
        self.actor.load_best()
        self.target_actor.load_best()
        for c in self.critics:
            c.load_best()
        for tc in self.target_critics:
            tc.load_best()

    def save_training_state(self, extra=None):
        """Save training state (noise, counters, etc.) to JSON."""
        state = {
            'time_step': self.time_step,
            'learn_step_cntr': self.learn_step_cntr,
            'noise_scale': self.noise_scheduler.scale,
            'noise_step': self.noise_scheduler._step,
            'reward_mean': float(self.reward_normalizer.mean),
            'reward_var': float(self.reward_normalizer.var),
            'reward_count': int(self.reward_normalizer.count),
        }
        if extra:
            state.update(extra)
        path = os.path.join(self.chkpt_dir, 'training_state.json')
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_training_state(self):
        """Load training state from JSON. Returns extra keys dict or None."""
        path = os.path.join(self.chkpt_dir, 'training_state.json')
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            state = json.load(f)
        self.time_step = state.get('time_step', 0)
        self.learn_step_cntr = state.get('learn_step_cntr', 0)
        self.noise_scheduler.scale = state.get('noise_scale', self.noise_scheduler.scale)
        self.noise_scheduler._step = state.get('noise_step', 0)
        self.reward_normalizer.mean = state.get('reward_mean', 0.0)
        self.reward_normalizer.var = state.get('reward_var', 1.0)
        self.reward_normalizer.count = state.get('reward_count', 0)
        print(f"Resumed training state: time_step={self.time_step:,}, "
              f"noise={self.noise_scheduler.scale:.3f}, "
              f"learn_steps={self.learn_step_cntr:,}")
        return state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nn_module_list_helper(n, obs_dim, goal_dim, n_actions, l1, l2,
                          prefix, chkpt_dir, lr):
    """Create a list of CriticNetwork instances with indexed names."""
    return [
        CriticNetwork(obs_dim, goal_dim, n_actions, l1, l2,
                      f'{prefix}_{i}', chkpt_dir, lr)
        for i in range(n)
    ]


def _soft_update(source, target, tau):
    """Polyak averaging: target = tau * source + (1 - tau) * target."""
    for sp, tp in zip(source.parameters(), target.parameters()):
        tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)
