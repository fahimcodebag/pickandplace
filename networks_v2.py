"""Goal-conditioned Actor and Critic networks with LayerNorm.

The networks accept (obs, goal) or (obs, goal, action) as separate inputs
and concatenate them internally. This keeps the interface clean for HER
where goals change between real and relabeled transitions.
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, n_actions, fc1_dims=1024, fc2_dims=512,
                 name='critic', chkpt_dir='./checkpoints/td3_v2', learning_rate=1e-3):
        super().__init__()

        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

        input_dim = obs_dim + goal_dim + n_actions

        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, goal, action):
        x = T.cat([state, goal, action], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.q(x)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))

    def save_best(self):
        T.save(self.state_dict(), self.checkpoint_file + '_best')

    def load_best(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_best', weights_only=True))


class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, goal_dim, fc1_dims=1024, fc2_dims=512, n_actions=7,
                 name='actor', chkpt_dir='./checkpoints/td3_v2', learning_rate=1e-3):
        super().__init__()

        if isinstance(obs_dim, tuple):
            obs_dim = obs_dim[0]

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

        input_dim = obs_dim + goal_dim

        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, goal):
        x = T.cat([state, goal], dim=-1)
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.ln2(self.fc2(x)))
        return T.tanh(self.output(x))

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file, weights_only=True))

    def save_best(self):
        T.save(self.state_dict(), self.checkpoint_file + '_best')

    def load_best(self):
        self.load_state_dict(T.load(self.checkpoint_file + '_best', weights_only=True))
