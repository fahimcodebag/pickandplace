import os
import torch as T
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork

class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99,
                 update_actor_interval=2, warmup=1000, n_actions=2,
                 max_size=1000000, layer1_size=512, layer2_size=256,
                 batch_size=100, noise=0.1, chkpt_dir='./checkpoints/td3'):
        
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
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        
        # Create checkpoint directory
        os.makedirs(chkpt_dir, exist_ok=True)
        
        # Create networks
        self.actor = ActorNetwork(input_dims, layer1_size, layer2_size, 
                                 n_actions, 'actor', chkpt_dir=chkpt_dir, learning_rate=alpha)
        self.critic_1 = CriticNetwork(input_dims, n_actions, layer1_size, 
                                     layer2_size, 'critic_1', chkpt_dir=chkpt_dir, learning_rate=beta)
        self.critic_2 = CriticNetwork(input_dims, n_actions, layer1_size, 
                                     layer2_size, 'critic_2', chkpt_dir=chkpt_dir, learning_rate=beta)
        
        # Target networks
        self.target_actor = ActorNetwork(input_dims, layer1_size, layer2_size, 
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

    def choose_action_batch(self, observations):
        """Batched action selection for vectorized environments.
        
        Passes all observations through the actor in a single forward pass.
        
        Args:
            observations: np.ndarray of shape (n_envs, obs_dim)
            
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
        
        self.update_network_parameters()

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