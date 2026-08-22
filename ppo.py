"""PPO (clipped surrogate + GAE), API-shaped like `td3.Agent`.

On-policy, so it shares nothing internally with TD3/SAC — but it exposes the
same `choose_action_batch / remember / learn` surface, letting `train_rand.py`
run one identical loop for every algorithm. `learn()` is a no-op until the
rollout fills, then performs the full multi-epoch update.

Why PPO earns a slot in the §9 comparison: being on-policy it is STRUCTURALLY
IMMUNE to the two replay pathologies that dominated this project — failure-
length data flooding (§3 Hurdle 7) and the overfit-to-self-similar-replay decay
that survived four interventions in the grasp campaign (§9.2). It cannot park
on stale data because it has none. That is a real hypothesis about the observed
decay, not a box-tick. It was previously untestable on 2 envs; at 20 it is.

Policy parameterization: tanh-mean Gaussian with a state-independent log_std.
Actions are the distribution's own samples clipped to the action range, and the
mean path is tanh(output(trunk)) — identical to the deployed actor — so the
INT8 export path (§7) is unchanged here too.

Truncation note: episode-horizon endings are treated as terminal (no value
bootstrap), matching how td3.py/sac.py handle `done`. Uniform across
algorithms, so the comparison stays fair.
"""

import os

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks_sac import CriticNetworkLN, _as_int


class _NoBuffer:
    """Stand-in so the shared trainer can query buffer stats uniformly."""
    mem_cntr = 0
    mem_size = 0

    def save(self, path):
        return None

    def load(self, path):
        return False


class PPOActor(nn.Module):
    def __init__(self, input_dims, fc1=64, fc2=32, n_actions=7,
                 chkpt_dir='./checkpoints/ppo', lr=3e-4, max_action=1.0,
                 init_log_std=-0.5):
        super().__init__()
        input_dims = _as_int(input_dims)
        self.name = 'actor'
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_td3')
        self.max_action = max_action

        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.output = nn.Linear(fc2, n_actions)
        self.log_std = nn.Parameter(T.full((n_actions,), float(init_log_std)))

        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return T.tanh(self.output(x)) * self.max_action

    def dist(self, state):
        return T.distributions.Normal(self(state), self.log_std.exp())

    def deployment_state_dict(self):
        return {k: v.clone() for k, v in self.state_dict().items()
                if k != 'log_std'}

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        T.save(self.deployment_state_dict(),
               os.path.join(self.checkpoint_dir, 'actor_deploy_td3'))

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, input_dims, fc1=64, fc2=32,
                 chkpt_dir='./checkpoints/ppo', lr=3e-4, layer_norm=False):
        super().__init__()
        input_dims = _as_int(input_dims)
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_1_td3')
        self.layer_norm = layer_norm
        self.fc1 = nn.Linear(input_dims, fc1)
        self.fc2 = nn.Linear(fc1, fc2)
        self.v = nn.Linear(fc2, 1)
        if layer_norm:
            self.ln1, self.ln2 = nn.LayerNorm(fc1), nn.LayerNorm(fc2)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(self.ln1(x) if self.layer_norm else x)
        x = self.fc2(x)
        x = F.relu(self.ln2(x) if self.layer_norm else x)
        return self.v(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 n_actions=2, layer1_size=64, layer2_size=32,
                 batch_size=512, chkpt_dir='./checkpoints/ppo',
                 warmup=0, max_size=None, noise=None,
                 update_actor_interval=None, layer_norm=False,
                 rollout_steps=512, n_epochs=10, clip_eps=0.2,
                 gae_lambda=0.95, entropy_coef=0.003, vf_coef=0.5,
                 max_grad_norm=0.5, n_envs=1):
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_eps, self.n_epochs = clip_eps, n_epochs
        self.entropy_coef, self.vf_coef = entropy_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_steps, self.n_envs = rollout_steps, n_envs
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low
        self.chkpt_dir = chkpt_dir
        self.time_step, self.warmup = 0, 0     # PPO has no random warmup
        self.learn_step_cntr = 0
        self.memory = _NoBuffer()
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        os.makedirs(chkpt_dir, exist_ok=True)

        self.actor = PPOActor(input_dims, layer1_size, layer2_size, n_actions,
                              chkpt_dir, alpha, float(self.max_action[0]))
        self.critic = ValueNetwork(input_dims, layer1_size, layer2_size,
                                   chkpt_dir, beta, layer_norm)
        self.optimizer = optim.Adam(
            [{'params': self.actor.parameters(), 'lr': alpha},
             {'params': self.critic.parameters(), 'lr': beta}])

        self._reset_rollout()
        self._pending = None       # (log_prob, value) for the in-flight step

    def _reset_rollout(self):
        self._s, self._a, self._r = [], [], []
        self._d, self._lp, self._v = [], [], []

    # --- acting -------------------------------------------------------------

    def choose_action_batch(self, observations, noise_scale=None):
        state = T.tensor(np.asarray(observations), dtype=T.float,
                         device=self.device)
        with T.no_grad():
            dist = self.actor.dist(state)
            raw = dist.sample()
            log_prob = dist.log_prob(raw).sum(-1)
            value = self.critic(state).squeeze(-1)
        action = T.clamp(raw, float(self.min_action[0]),
                         float(self.max_action[0]))
        self._pending = (log_prob.cpu().numpy(), value.cpu().numpy(),
                         raw.cpu().numpy())
        self.time_step += len(observations)
        return action.cpu().numpy()

    def choose_action(self, observation, validation=False):
        state = T.tensor(np.asarray(observation), dtype=T.float,
                         device=self.device).unsqueeze(0)
        with T.no_grad():
            action = (self.actor(state) if validation
                      else self.actor.dist(state).sample())
        return np.clip(action.squeeze(0).cpu().numpy(),
                       self.min_action, self.max_action)

    def remember(self, state, action, reward, new_state, done, env_idx=0):
        """Append one transition. `action` is ignored in favour of the
        pre-clip sample whose log-prob was recorded — clipped actions would
        make the importance ratio inconsistent."""
        lp, v, raw = self._pending
        self._s.append(np.asarray(state, dtype=np.float32))
        self._a.append(raw[env_idx])
        self._lp.append(lp[env_idx])
        self._v.append(v[env_idx])
        self._r.append(float(reward))
        self._d.append(bool(done))

    def rollout_full(self):
        return len(self._r) >= self.rollout_steps * self.n_envs

    # --- learning -----------------------------------------------------------

    def learn(self):
        if not self.rollout_full():
            return

        s = T.tensor(np.array(self._s), dtype=T.float, device=self.device)
        a = T.tensor(np.array(self._a), dtype=T.float, device=self.device)
        old_lp = T.tensor(np.array(self._lp), dtype=T.float, device=self.device)
        values = np.array(self._v, dtype=np.float32)
        rewards = np.array(self._r, dtype=np.float32)
        dones = np.array(self._d, dtype=bool)

        # GAE. Transitions from the n_envs streams are interleaved in arrival
        # order; advantages are accumulated backwards and cut at every `done`,
        # so no credit crosses an episode boundary.
        adv = np.zeros_like(rewards)
        last = 0.0
        for t in reversed(range(len(rewards))):
            next_v = 0.0 if (t + 1 >= len(rewards) or dones[t]) else values[t + 1]
            nonterminal = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.gamma * next_v * nonterminal - values[t]
            last = delta + self.gamma * self.gae_lambda * nonterminal * last
            adv[t] = last
        returns = adv + values

        adv_t = T.tensor(adv, dtype=T.float, device=self.device)
        ret_t = T.tensor(returns, dtype=T.float, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(rewards)
        idx = np.arange(n)
        for _ in range(self.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                if len(b) < 2:
                    continue
                bt = T.tensor(b, dtype=T.long, device=self.device)
                dist = self.actor.dist(s[bt])
                lp = dist.log_prob(a[bt]).sum(-1)
                ratio = (lp - old_lp[bt]).exp()

                surr1 = ratio * adv_t[bt]
                surr2 = T.clamp(ratio, 1 - self.clip_eps,
                                1 + self.clip_eps) * adv_t[bt]
                policy_loss = -T.min(surr1, surr2).mean()
                value_loss = F.mse_loss(self.critic(s[bt]).squeeze(-1), ret_t[bt])
                entropy = dist.entropy().sum(-1).mean()

                loss = (policy_loss + self.vf_coef * value_loss
                        - self.entropy_coef * entropy)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        self.learn_step_cntr += 1
        self._reset_rollout()

    def reset_critic_heads(self):
        self.critic.v.reset_parameters()     # no-op-ish; PPO has no stale replay

    # --- persistence ---------------------------------------------------------

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
