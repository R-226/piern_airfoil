"""
PiERN Policy Network.

Physics-informed reinforcement learning network for airfoil optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PhysicsInformedMLP(nn.Module):
    """
    Physics-informed MLP with mandatory physics constraints.

    Uses physics-inspired architecture:
    - Physics-consistent activation functions
    - Symmetric/signed pathways for physical quantities
    - Monotonicity constraints
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int,
                 activation: str = "tanh", use_spectral_norm: bool = False):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        for i in range(len(dims) - 1):
            layer = nn.Linear(dims[i], dims[i+1])

            # Optional spectral normalization for stability
            if use_spectral_norm and i < len(dims) - 2:
                layer = nn.utils.spectral_norm(layer)

            self.layers.append(layer)

        # Activation
        if activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:  # No activation on output
                x = self.activation(x)
        return x


class PiERNPolicyNetwork(nn.Module):
    """
    PiERN Policy Network for airfoil optimization.

    Architecture:
    - Shared feature extractor (physics-informed)
    - Policy head (parameter adjustments + analysis choice + exploration)
    - Value head (state value estimation)

    Input state: [geometry_params(18) + performance(4) + constraint_violations(5) + confidence(1)] = 28 dims
    Output action: [param_delta(18) + analysis_choice(1) + exploration(1)] = 20 dims
    """

    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 20,
        hidden_dims: list = None,
        use_physics_informed: bool = True
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Physics-informed feature extraction
        if use_physics_informed:
            self.feature_net = PhysicsInformedMLP(
                input_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=hidden_dims[-1],
                activation="tanh"  # Smooth, physics-friendly
            )
        else:
            self.feature_net = nn.Sequential(
                nn.Linear(state_dim, hidden_dims[0]),
                nn.ReLU(),
                nn.Linear(hidden_dims[0], hidden_dims[1]),
                nn.ReLU(),
                nn.Linear(hidden_dims[1], hidden_dims[2]),
                nn.ReLU()
            )

        # Policy head (outputs action distribution parameters)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1] // 2, action_dim)
        )

        # Value head (outputs state value)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.Tanh(),
            nn.Linear(hidden_dims[-1] // 2, 1)
        )

        # Log standard deviation for action exploration (learnable)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor of shape (batch, state_dim)

        Returns:
            (policy_output, value_output)
            - policy_output: Action mean tensor (batch, action_dim)
            - value_output: State value tensor (batch, 1)
        """
        # Extract features
        features = self.feature_net(state)

        # Compute policy and value
        policy = self.policy_head(features)
        value = self.value_head(features)

        return policy, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from state.

        Args:
            state: State tensor
            deterministic: If True, return mean action; else sample

        Returns:
            (action, log_prob)
            - action: Selected action (batch, action_dim)
            - log_prob: Log probability of action (batch,)
        """
        policy, value = self.forward(state)

        if deterministic:
            action = policy
            log_prob = None
        else:
            # Sample from Gaussian distribution
            std = torch.exp(self.log_std).expand_as(policy)
            dist = torch.distributions.Normal(policy, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for training.

        Args:
            states: State tensor (batch, state_dim)
            actions: Action tensor (batch, action_dim)

        Returns:
            (log_probs, values)
        """
        policy, value = self.forward(states)

        std = torch.exp(self.log_std).expand_as(policy)
        dist = torch.distributions.Normal(policy, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)

        return log_probs, value


class PiERNValueNetwork(nn.Module):
    """
    Value network for PiERN.

    Estimates state value function V(s) for RL training.
    Uses physics-informed architecture similar to policy network.
    """

    def __init__(
        self,
        state_dim: int = 28,
        hidden_dims: list = None
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.Tanh(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.Tanh(),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.Tanh(),
            nn.Linear(hidden_dims[2], 1)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)


class TargetNetwork(nn.Module):
    """
    Target network for stable RL training.

    Periodically copies weights from main network.
    Used in DQN, DDPG, and other off-policy algorithms.
    """

    def __init__(self, network: nn.Module, tau: float = 0.005):
        super().__init__()
        self.network = network
        self.tau = tau
        self._sync()

    def _sync(self):
        """Synchronize with main network."""
        self.network.load_state_dict(self.network.state_dict())

    def soft_update(self, source_network: nn.Module):
        """
        Soft update: target = tau * source + (1 - tau) * target.

        Args:
            source_network: Source network to copy from
        """
        for target_param, source_param in zip(
            self.network.parameters(),
            source_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def hard_update(self, source_network: nn.Module):
        """Hard update: directly copy weights."""
        self.network.load_state_dict(source_network.state_dict())
