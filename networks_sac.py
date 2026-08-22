"""Networks for the random-spawn algorithm comparison (SAC + LayerNorm critics).

Two additions over `networks.py`, both training-only:

  * `CriticNetworkLN` — critic with optional LayerNorm. Critics never leave the
    training host (§7), so normalization here costs nothing at deployment and
    targets the plasticity loss diagnosed in §9.3.
  * `GaussianActor` — SAC's tanh-squashed stochastic actor.

DEPLOYMENT INVARIANT: `GaussianActor` names its trunk `fc1`/`fc2` and its mean
head `output`, exactly matching `networks.ActorNetwork`. Dropping the
`log_std` head therefore yields a state_dict that loads straight into the
deployed 46->64->32->7 tanh actor, so the INT8/QAT pipeline and the 8.4 KB
artifact are unchanged regardless of which algorithm trained the weights.
"""

import os

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _as_int(dims):
    return dims[0] if isinstance(dims, tuple) else dims


class CriticNetworkLN(nn.Module):
    """Q(s, a) critic, optionally LayerNorm-regularized.

    Parameter names match `networks.CriticNetwork` when layer_norm=False, so
    existing TD3 critic checkpoints load without translation.
    """

    def __init__(self, input_dims, n_actions, fc1_dims=64, fc2_dims=32,
                 name='critic', chkpt_dir='./checkpoints/td3',
                 learning_rate=3e-4, layer_norm=False):
        super().__init__()
        input_dims = _as_int(input_dims)
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')
        self.layer_norm = layer_norm

        self.fc1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q1 = nn.Linear(fc2_dims, 1)
        if layer_norm:
            self.ln1 = nn.LayerNorm(fc1_dims)
            self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = self.fc1(T.cat([state, action], dim=1))
        x = F.relu(self.ln1(x) if self.layer_norm else x)
        x = self.fc2(x)
        x = F.relu(self.ln2(x) if self.layer_norm else x)
        return self.q1(x)

    def reset_head(self):
        """Reinitialize the output layer, keeping the trunk and the buffer.

        The primacy-bias remedy (Nikishin et al.): periodic partial resets
        restore plasticity after long training on self-similar data — the
        signature seen five times in the grasp campaign (§9.2).
        """
        self.q1.reset_parameters()

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Load, tolerating a warm start from a checkpoint without LayerNorm.

        Warm-starting a LayerNorm critic from a plain-critic checkpoint (e.g.
        the fixed-spawn or metric-0.355 artifacts, trained with `networks.py`)
        legitimately has no ln1/ln2 entries. Those params keep their fresh
        init (weight=1, bias=0) and the trunk is seeded as intended. Any OTHER
        missing/unexpected key is a real mismatch and still raises.
        """
        state = T.load(self.checkpoint_file)
        try:
            self.load_state_dict(state)
        except RuntimeError:
            missing, unexpected = self.load_state_dict(state, strict=False)
            stray = [k for k in list(missing) + list(unexpected)
                     if not k.startswith(("ln1.", "ln2."))]
            if stray:
                raise
            print(f"  [{self.name}] warm start without LayerNorm params "
                  f"({len(missing)} ln keys freshly initialized)", flush=True)


class GaussianActor(nn.Module):
    """Tanh-squashed Gaussian policy for SAC.

    Trunk/mean-head naming mirrors networks.ActorNetwork (see module docstring).
    """

    def __init__(self, input_dims, fc1_dims=64, fc2_dims=32, n_actions=7,
                 name='actor', chkpt_dir='./checkpoints/sac',
                 learning_rate=3e-4, max_action=1.0, init_log_std=-1.6):
        super().__init__()
        input_dims = _as_int(input_dims)
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_td3')
        self.max_action = max_action

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.output = nn.Linear(fc2_dims, n_actions)      # mean head
        self.log_std = nn.Linear(fc2_dims, n_actions)     # training-only
        # Start exploration at sigma ~= exp(-1.6) = 0.20, comparable to TD3's
        # fixed noise=0.1, and let the temperature adapt from there. The
        # default Linear init emits log_std ~= 0 (sigma ~= 1.0), which for
        # tanh-squashed actions in [-1,1] swamps the policy: measured sampled
        # |a| 0.559 vs deterministic |a| 0.095, i.e. noise ~6x the signal. That
        # destroys a WARM START outright (observed: 62% of episodes never
        # reached the object, success falling 44->16%) and would have made this
        # an initialization comparison rather than an algorithm comparison.
        nn.init.constant_(self.log_std.bias, init_log_std)
        nn.init.uniform_(self.log_std.weight, -1e-3, 1e-3)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _trunk(self, state):
        x = F.relu(self.fc1(state))
        return F.relu(self.fc2(x))

    def forward(self, state):
        """Deterministic action — tanh(mean). This is the deployed behaviour."""
        return T.tanh(self.output(self._trunk(state))) * self.max_action

    def sample(self, state):
        """Reparameterized sample with the tanh log-prob correction.

        Returns (action, log_prob). The correction term is the log-determinant
        of the tanh Jacobian; 1e-6 guards log(0) at saturation.
        """
        h = self._trunk(state)
        mu = self.output(h)
        log_std = T.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = T.distributions.Normal(mu, std)
        z = normal.rsample()
        squashed = T.tanh(z)
        action = squashed * self.max_action

        log_prob = normal.log_prob(z) - T.log(1 - squashed.pow(2) + 1e-6)
        return action, log_prob.sum(dim=1, keepdim=True)

    def deployment_state_dict(self):
        """state_dict for the deployed actor: trunk + mean head, no log_std.

        Loads directly into networks.ActorNetwork, keeping the QAT/INT8 export
        path (§7) identical across algorithms.
        """
        return {k: v.clone() for k, v in self.state_dict().items()
                if not k.startswith('log_std.')}

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        T.save(self.deployment_state_dict(),
               os.path.join(self.checkpoint_dir, self.name + '_deploy_td3'))

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
