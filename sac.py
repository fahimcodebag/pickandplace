"""Soft Actor-Critic with PER, API-compatible with `td3.Agent`.

Drop-in for the random-spawn grasp comparison: same constructor keywords, same
`choose_action_batch / remember / learn / save_models / load_models` surface and
the same `ReplayBuffer`, so `train_rand.py` can swap algorithms without touching
the environment, reward, or logging stack.

Why SAC is a candidate here (§9.3): TD3 explores with a FIXED-scale Gaussian
(td3.py `noise=0.1`), so one noise magnitude must serve easy centre spawns and
hard corner spawns alike. SAC learns a STATE-DEPENDENT stochastic policy and
tunes its own temperature, which is the right shape for a uniformly diverse
spawn distribution and directly opposes the premature determinism collapse
observed five times in the grasp campaign.

Deployment is unaffected: only tanh(mean) is exported (see networks_sac).
"""

import os

import numpy as np
import torch as T
import torch.nn.functional as F

from buffer import ReplayBuffer
from networks_sac import CriticNetworkLN, GaussianActor


class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 update_actor_interval=1, warmup=1000, n_actions=2,
                 max_size=1000000, layer1_size=64, layer2_size=32,
                 batch_size=100, noise=0.1, chkpt_dir='./checkpoints/sac',
                 layer_norm=False, init_alpha=0.2, target_entropy=None):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.chkpt_dir = chkpt_dir
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.buf_path = os.path.join(chkpt_dir, 'replay_buffer.npz')
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.noise = noise          # unused; kept for API parity with td3
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')

        os.makedirs(chkpt_dir, exist_ok=True)

        self.actor = GaussianActor(
            input_dims, layer1_size, layer2_size, n_actions, 'actor',
            chkpt_dir=chkpt_dir, learning_rate=alpha,
            max_action=float(self.max_action[0]))
        mk = lambda name: CriticNetworkLN(
            input_dims, n_actions, layer1_size, layer2_size, name,
            chkpt_dir=chkpt_dir, learning_rate=beta, layer_norm=layer_norm)
        self.critic_1, self.critic_2 = mk('critic_1'), mk('critic_2')
        self.target_critic_1, self.target_critic_2 = (mk('target_critic_1'),
                                                      mk('target_critic_2'))
        self.update_network_parameters(tau=1)

        # Automatic temperature tuning. Target entropy -dim(A) is the standard
        # heuristic (Haarnoja et al. 2018); log-space keeps alpha positive.
        self.target_entropy = (-float(n_actions) if target_entropy is None
                               else float(target_entropy))
        self.log_alpha = T.tensor(np.log(init_alpha), dtype=T.float,
                                  requires_grad=True, device=self.device)
        self.alpha_optimizer = T.optim.Adam([self.log_alpha], lr=beta)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    # --- acting -------------------------------------------------------------

    def choose_action(self, observation, validation=False):
        if self.time_step < self.warmup and not validation:
            self.time_step += 1
            return np.random.uniform(self.min_action, self.max_action,
                                     self.n_actions).astype(np.float32)
        state = T.tensor(np.asarray(observation), dtype=T.float,
                         device=self.device).unsqueeze(0)
        with T.no_grad():
            if validation:
                action = self.actor(state)          # deterministic: tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        self.time_step += 1
        return action.squeeze(0).cpu().numpy()

    def choose_action_batch(self, observations, noise_scale=None):
        """Batched stochastic actions.

        `noise_scale` is accepted for API parity with td3.Agent but ignored:
        SAC's exploration is learned per-state, which is the whole point of
        testing it here. Passing it is not an error, so the caller stays
        algorithm-agnostic.
        """
        n_envs = len(observations)
        if self.time_step < self.warmup:
            self.time_step += n_envs
            return np.random.uniform(self.min_action, self.max_action,
                                     (n_envs, self.n_actions)).astype(np.float32)
        state = T.tensor(np.asarray(observations), dtype=T.float,
                         device=self.device)
        with T.no_grad():
            actions, _ = self.actor.sample(state)
        self.time_step += n_envs
        return actions.cpu().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # --- learning -----------------------------------------------------------

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return

        state, action, reward, next_state, done, tree_idx, is_w = \
            self.memory.sample_buffer_per(self.batch_size)

        to = lambda x, d=T.float: T.tensor(x, dtype=d, device=self.device)
        state, next_state = to(state), to(next_state)
        action, reward, is_w = to(action), to(reward), to(is_w)
        done = T.tensor(done, device=self.device)

        # --- critics: soft Bellman backup with entropy bonus -----------------
        with T.no_grad():
            next_action, next_logp = self.actor.sample(next_state)
            nq1 = self.target_critic_1(next_state, next_action)
            nq2 = self.target_critic_2(next_state, next_action)
            next_q = T.min(nq1, nq2) - self.alpha * next_logp
            next_q[done] = 0.0
            target = (reward.view(-1, 1) + self.gamma * next_q)

        q1 = self.critic_1(state, action)
        q2 = self.critic_2(state, action)

        td_errors = (target - q1).detach().squeeze().cpu().numpy()
        self.memory.update_priorities(tree_idx, td_errors)

        w = is_w.view(-1, 1)
        critic_loss = ((w * (target - q1).pow(2)).mean()
                       + (w * (target - q2).pow(2)).mean())
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # --- actor: maximize Q - alpha * log pi ------------------------------
        new_action, logp = self.actor.sample(state)
        q_new = T.min(self.critic_1(state, new_action),
                      self.critic_2(state, new_action))
        actor_loss = (self.alpha * logp - q_new).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # --- temperature -----------------------------------------------------
        alpha_loss = -(self.log_alpha
                       * (logp.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.learn_step_cntr += 1
        self.update_network_parameters()

    def reset_critic_heads(self):
        """Primacy-bias reset of critic output layers (buffer retained)."""
        self.critic_1.reset_head()
        self.critic_2.reset_head()
        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        tau = self.tau if tau is None else tau
        for net, target in ((self.critic_1, self.target_critic_1),
                            (self.critic_2, self.target_critic_2)):
            for p, tp in zip(net.parameters(), target.parameters()):
                tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

    # --- persistence ---------------------------------------------------------

    def save_models(self):
        for net in (self.actor, self.critic_1, self.critic_2,
                    self.target_critic_1, self.target_critic_2):
            net.save_checkpoint()
        T.save(self.log_alpha.detach().cpu(),
               os.path.join(self.chkpt_dir, 'log_alpha_td3'))

    def load_models(self):
        for net in (self.actor, self.critic_1, self.critic_2,
                    self.target_critic_1, self.target_critic_2):
            net.load_checkpoint()
        p = os.path.join(self.chkpt_dir, 'log_alpha_td3')
        if os.path.exists(p):
            with T.no_grad():
                self.log_alpha.copy_(T.load(p).to(self.device))
