import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class _FakeQuantSTE(T.autograd.Function):
    """Per-tensor symmetric INT8 quantise-dequantise, straight-through grad."""

    @staticmethod
    def forward(ctx, w):
        s = w.detach().abs().max() / 127.0
        if s == 0:
            return w
        return T.clamp(T.round(w / s), -127.0, 127.0) * s

    @staticmethod
    def backward(ctx, g):
        return g

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 update_actor_interval=2, warmup=1000, n_actions=2,
                 max_size=1000000, layer1_size=512, layer2_size=256,
                 batch_size=100, noise=0.1, chkpt_dir='./checkpoints/td3',
                 actor_layer1=None, actor_layer2=None):
        
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
        self.update_actor_iter = update_actor_interval
        self.noise = noise
        # Per-tensor INT8 sets ONE scale per tensor from the largest weight, so
        # a tensor's max/std IS its quantisation cost. Measured across three
        # deployed models (Results/int8_deployment.txt, Finding 4): transport
        # 5.7 -> quantises free, gripfix_s2 9.3 -> -7.2 pts, bi_s0 10.0 ->
        # -14.0 pts. actor_wclip projects each actor weight tensor onto
        # |w| <= k*std after every update, capping that ratio at k.
        # None = off, and the update path is then byte-identical to before.
        self.actor_wclip = None
        # Route (b): QAT in the RL loop. The only QAT tried before
        # (qat_finetune.py:184) minimised MSE to an FP32 teacher on a frozen
        # calibration buffer -- an objective Results/int8_deployment.txt Fnd 2
        # showed is ANTI-correlated with deployed success (best corr 0.952 ->
        # worst behaviour 37.1%). This instead fake-quantises the actor inside
        # training so the RL return itself is optimised under quantisation.
        # None = off, update path byte-identical to before.
        self.actor_fakequant = False
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        
        # Create checkpoint directory
        os.makedirs(chkpt_dir, exist_ok=True)
        
        # The actor may be WIDER than the critics. Critics never leave the
        # host (Sec 7), so their capacity is a training-time choice and there
        # is no reason to widen them alongside a deployed actor -- Q(s,a) has
        # the same 7-d action input at any actor width. Keeping them fixed also
        # preserves their warm start exactly, and avoids Net2WiderNet through
        # the td3_ln critics' LayerNorm, which does NOT preserve the function
        # (LN renormalises over a feature count that duplication changes).
        a1 = actor_layer1 or layer1_size
        a2 = actor_layer2 or layer2_size
        self.actor_layer1, self.actor_layer2 = a1, a2

        # Create networks
        self.actor = ActorNetwork(input_dims, a1, a2,
                                 n_actions, 'actor', chkpt_dir=chkpt_dir, learning_rate=alpha)
        self.critic_1 = CriticNetwork(input_dims, n_actions, layer1_size, 
                                     layer2_size, 'critic_1', chkpt_dir=chkpt_dir, learning_rate=beta)
        self.critic_2 = CriticNetwork(input_dims, n_actions, layer1_size, 
                                     layer2_size, 'critic_2', chkpt_dir=chkpt_dir, learning_rate=beta)
        
        # Target networks
        self.target_actor = ActorNetwork(input_dims, a1, a2,
                                        n_actions, 'target_actor', chkpt_dir=chkpt_dir, learning_rate=alpha)
        self.target_critic_1 = CriticNetwork(input_dims, n_actions, layer1_size, 
                                            layer2_size, 'target_critic_1', chkpt_dir=chkpt_dir, learning_rate=beta)
        self.target_critic_2 = CriticNetwork(input_dims, n_actions, layer1_size, 
                                            layer2_size, 'target_critic_2', chkpt_dir=chkpt_dir, learning_rate=beta)
        
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, validation=False):
        if self.time_step < self.warmup and not validation:
            mu = T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)), 
                         dtype=T.float).to(self.device)
        else:
            state = T.tensor(observation, dtype=T.float).to(self.device)
            mu = self.actor.forward(state)

        if validation:
            # No exploration noise during evaluation
            mu_prime = T.clamp(mu, self.min_action[0], self.max_action[0])
        else:
            mu_prime = mu + T.tensor(np.random.normal(scale=self.noise, size=(self.n_actions,)),
                                     dtype=T.float).to(self.device)
            mu_prime = T.clamp(mu_prime, self.min_action[0], self.max_action[0])
        self.time_step += 1
        return mu_prime.cpu().detach().numpy()

    def choose_action_batch(self, observations, noise_scale=None):
        """Batched action selection for vectorized environments.

        Passes all observations through the actor in a single forward pass.

        Args:
            observations: np.ndarray of shape (n_envs, obs_dim)
            noise_scale:  optional per-action-dim multiplier on the exploration
                          noise, shape (n_actions,). Defaults to None (uniform
                          scale), preserving the original behaviour for callers
                          that don't pass it. Used by the place task to shrink
                          exploration noise on the gripper dim so it doesn't
                          pop the grasp open.

        Returns:
            np.ndarray of shape (n_envs, n_actions)
        """
        n_envs = observations.shape[0]
        if self.time_step < self.warmup:
            # Wide uniform random exploration so robot sees diverse states
            actions = np.random.uniform(
                self.min_action[0], self.max_action[0],
                size=(n_envs, self.n_actions)
            )
        else:
            states = T.tensor(observations, dtype=T.float).to(self.device)
            with T.no_grad():
                actions = self.actor.forward(states).cpu().numpy()

        noise = np.random.normal(scale=self.noise, size=(n_envs, self.n_actions))
        if noise_scale is not None:
            noise = noise * np.asarray(noise_scale, dtype=noise.dtype)
        actions = actions + noise
        actions = np.clip(actions, self.min_action[0], self.max_action[0])
        self.time_step += n_envs
        return actions

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size * 10:
            return
            
        state, action, reward, next_state, done, tree_idx, is_weights = \
            self.memory.sample_buffer_per(self.batch_size)
        
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        done = T.tensor(done).to(self.device)
        next_state = T.tensor(next_state, dtype=T.float).to(self.device)
        state = T.tensor(state, dtype=T.float).to(self.device)
        action = T.tensor(action, dtype=T.float).to(self.device)
        is_weights = T.tensor(is_weights, dtype=T.float).to(self.device)
        
        target_actions = self.target_actor.forward(next_state)
        target_actions = target_actions + T.clamp(T.tensor(np.random.normal(scale=0.2, size=target_actions.shape), 
                                                           dtype=T.float).to(self.device), -0.5, 0.5)
        target_actions = T.clamp(target_actions, self.min_action[0], self.max_action[0])
        
        next_q1 = self.target_critic_1.forward(next_state, target_actions)
        next_q2 = self.target_critic_2.forward(next_state, target_actions)
        
        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)
        
        next_q1[done] = 0.0
        next_q2[done] = 0.0
        
        next_q1 = next_q1.view(-1)
        next_q2 = next_q2.view(-1)
        
        next_critic_value = T.min(next_q1, next_q2)
        
        target = reward + self.gamma * next_critic_value
        target = target.view(self.batch_size, 1)

        # Per-sample TD errors for priority update
        td_errors = (target - q1).detach().squeeze().cpu().numpy()
        self.memory.update_priorities(tree_idx, td_errors)
        
        # IS-weighted critic loss
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        
        w = is_weights.view(-1, 1)
        q1_loss = (w * (target - q1).pow(2)).mean()
        q2_loss = (w * (target - q2).pow(2)).mean()
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
        self.learn_step_cntr += 1
        
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return
            
        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self._clip_actor_weights()
        
        self.update_network_parameters()

    def enable_actor_fakequant(self):
        """Per-tensor symmetric INT8 fake-quant on actor weights, STE gradient.

        Mirrors the deployed converter's TENSORWISE granularity: one scale per
        weight tensor taken from max|w|, which is exactly the quantisation the
        ESP-NN kernels run. Patches self.actor only -- it is the artifact that
        ships, and it drives both choose_action and the actor loss, so the
        replay data reflects the quantised behaviour policy. Target networks
        stay full precision: they are a training-time bootstrap, and
        quantising them only injects noise into the targets.
        """
        self.actor_fakequant = True
        for m in self.actor.modules():
            if isinstance(m, T.nn.Linear) and not getattr(m, "_fq", False):
                m._fq = True
                m.forward = (lambda x, m=m:
                             F.linear(x, _FakeQuantSTE.apply(m.weight), m.bias))

    def _clip_actor_weights(self):
        """Project actor weight tensors onto |w| <= actor_wclip * std(w).

        Applied to Linear weights only: biases are a separate quantised tensor
        with their own scale, and clipping them does not affect the weight
        scale that costs the resolution. Applied every actor update, this
        reaches the fixed point exactly: measured 9.96/10.82 -> 6.00/6.00 at
        k=6, with tensors already below k left untouched.
        """
        if not self.actor_wclip:
            return
        with T.no_grad():
            for m in self.actor.modules():
                if isinstance(m, T.nn.Linear):
                    lim = self.actor_wclip * m.weight.std()
                    m.weight.clamp_(-lim, lim)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_1_params = self.target_critic_1.named_parameters()
        target_critic_2_params = self.target_critic_2.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        target_actor_state_dict = dict(target_actor_params)
        target_critic_1_state_dict = dict(target_critic_1_params)
        target_critic_2_state_dict = dict(target_critic_2_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = tau*critic_1_state_dict[name].clone() + \
                                       (1 - tau)*target_critic_1_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = tau * critic_2_state_dict[name].clone() + \
                                       (1 - tau) * target_critic_2_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau * actor_state_dict[name].clone() + \
                                    (1 - tau) * target_actor_state_dict[name].clone()

        self.target_critic_1.load_state_dict(critic_1_state_dict)
        self.target_critic_2.load_state_dict(critic_2_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.target_critic_1.save_checkpoint()
        self.target_critic_2.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.target_critic_1.load_checkpoint()
        self.target_critic_2.load_checkpoint()