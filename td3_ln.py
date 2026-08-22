"""TD3 with LayerNorm critics + optional primacy-bias resets.

A thin subclass of `td3.Agent` that swaps the plain critics for
`CriticNetworkLN`. Everything else — actor, exploration noise, PER, target
smoothing, delayed actor updates — is inherited unchanged, so this isolates
normalization as a single variable in the §9 comparison.

Both changes are training-only: critics never leave the host (§7) and the
actor architecture is byte-identical to `networks.ActorNetwork`, so the
deployed 8.4 KB INT8 artifact is unaffected.
"""

import torch as T

from td3 import Agent as TD3Agent
from networks_sac import CriticNetworkLN


class Agent(TD3Agent):
    def __init__(self, *args, layer_norm=True, **kwargs):
        super().__init__(*args, **kwargs)
        alpha_lr = self.critic_1.optimizer.param_groups[0]['lr']
        input_dims = self.actor.input_dims
        l1, l2 = self.critic_1.fc1_dims, self.critic_1.fc2_dims

        mk = lambda name: CriticNetworkLN(
            input_dims, self.n_actions, l1, l2, name,
            chkpt_dir=self.chkpt_dir, learning_rate=alpha_lr,
            layer_norm=layer_norm)
        self.critic_1, self.critic_2 = mk('critic_1'), mk('critic_2')
        self.target_critic_1 = mk('target_critic_1')
        self.target_critic_2 = mk('target_critic_2')
        self.update_network_parameters(tau=1)

    def reset_critic_heads(self):
        """Reinitialize critic output layers, keeping trunks and the buffer."""
        self.critic_1.reset_head()
        self.critic_2.reset_head()
        self.update_network_parameters(tau=1)
