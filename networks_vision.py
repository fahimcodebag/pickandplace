"""Vision-based Actor and Critic networks for camera-feedback TD3.

Architecture:
  - CNN encoder: 3 conv layers (32→64→64) with ReLU, then flatten
  - The latent CNN features are concatenated with the goal (and action for critic)
  - MLP head with LayerNorm produces the final output

Designed for 84×84×3 RGB input images from robosuite cameras.
"""

import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CNNEncoder(nn.Module):
    """Shared CNN encoder for processing 84×84 RGB images.

    Architecture matches the Nature DQN encoder, which is standard
    for pixel-based RL:
      Conv2d(3, 32, 8, stride=4) → ReLU
      Conv2d(32, 64, 4, stride=2) → ReLU
      Conv2d(64, 64, 3, stride=1) → ReLU
      Flatten → Linear → LayerNorm → ReLU

    Output: latent_dim-dimensional feature vector.
    """

    def __init__(self, img_channels=3, latent_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # For 84×84 input: 32@20×20 → 64@9×9 → 64@7×7 = 3136
        self.conv_out_size = 64 * 7 * 7  # 3136

        self.fc = nn.Linear(self.conv_out_size, latent_dim)
        self.ln = nn.LayerNorm(latent_dim)

    def forward(self, img):
        """
        Args:
            img: (B, C, H, W) float tensor in [0, 1]
        Returns:
            (B, latent_dim) feature vector
        """
        x = self.conv(img)
        x = x.view(x.size(0), -1)
        x = F.relu(self.ln(self.fc(x)))
        return x


class VisionActorNetwork(nn.Module):
    """Actor: image + goal → action."""

    def __init__(self, goal_dim=3, latent_dim=256, fc1_dims=512,
                 fc2_dims=256, n_actions=7, img_channels=3,
                 name='actor', chkpt_dir='./checkpoints/td3_vision',
                 learning_rate=1e-4):
        super().__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

        # CNN encoder
        self.encoder = CNNEncoder(img_channels, latent_dim)

        # MLP head: latent + goal → action
        self.fc1 = nn.Linear(latent_dim + goal_dim, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, img, goal):
        """
        Args:
            img:  (B, C, H, W) normalised image tensor
            goal: (B, goal_dim)
        Returns:
            (B, n_actions) action in [-1, 1]
        """
        features = self.encoder(img)
        x = T.cat([features, goal], dim=-1)
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
        self.load_state_dict(T.load(self.checkpoint_file + '_best',
                                    weights_only=True))


class VisionCriticNetwork(nn.Module):
    """Critic: image + goal + action → Q-value."""

    def __init__(self, goal_dim=3, n_actions=7, latent_dim=256,
                 fc1_dims=512, fc2_dims=256, img_channels=3,
                 name='critic', chkpt_dir='./checkpoints/td3_vision',
                 learning_rate=1e-3):
        super().__init__()

        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')

        # CNN encoder
        self.encoder = CNNEncoder(img_channels, latent_dim)

        # MLP head: latent + goal + action → Q
        input_dim = latent_dim + goal_dim + n_actions
        self.fc1 = nn.Linear(input_dim, fc1_dims)
        self.ln1 = nn.LayerNorm(fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.ln2 = nn.LayerNorm(fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, img, goal, action):
        """
        Args:
            img:    (B, C, H, W) normalised image tensor
            goal:   (B, goal_dim)
            action: (B, n_actions)
        Returns:
            (B, 1) Q-value
        """
        features = self.encoder(img)
        x = T.cat([features, goal, action], dim=-1)
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
        self.load_state_dict(T.load(self.checkpoint_file + '_best',
                                    weights_only=True))
