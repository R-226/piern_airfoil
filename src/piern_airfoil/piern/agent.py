"""
PiERN Agent for airfoil optimization.

Implements RL agent with PPO algorithm for physics-informed optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass
import copy

from .policy_network import PiERNPolicyNetwork, TargetNetwork
from .state_representation import DesignState, DesignAction


@dataclass
class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    capacity: int
    batch_size: int = 32

    def __post_init__(self):
        self.buffer = []
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int = None) -> Tuple:
        """Sample random batch from buffer."""
        batch_size = batch_size or self.batch_size
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) Agent for airfoil optimization.

    Uses clipped surrogate objective for stable policy updates.
    """

    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 20,
        hidden_dims: List[int] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        clip_grad: float = 0.5,
        use_gpu: bool = True
    ):
        """
        Initialize PPO Agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            clip_grad: Gradient clipping value
            use_gpu: Whether to use GPU
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # Networks
        self.policy = PiERNPolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.value_net = PiERNPolicyNetwork(state_dim, 1, hidden_dims).to(self.device)
        self.target_value = TargetNetwork(self.value_net, tau=0.01)

        # Optimizer
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )

        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.clip_grad = clip_grad

        # Replay buffer for DQN-style updates (optional)
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # Training statistics
        self.training_step = 0

    def select_action(
        self,
        state: DesignState,
        epsilon: float = 0.1,
        deterministic: bool = False
    ) -> Tuple[DesignAction, float]:
        """
        Select action given state.

        Args:
            state: Current design state
            epsilon: Exploration rate
            deterministic: If True, select greedy action

        Returns:
            (action, log_prob)
        """
        state_vec = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            policy, value = self.policy(state_vec)

        # Exploration: random action with probability epsilon
        if not deterministic and np.random.rand() < epsilon:
            action_vec = torch.randn(1, 20).to(self.device) * 0.1
            log_prob = None
        else:
            # Use mean policy
            action_vec = policy
            std = torch.exp(self.policy.log_std)
            dist = torch.distributions.Normal(policy, std)
            log_prob = dist.log_prob(action_vec).sum().item()

        action = DesignAction.from_vector(action_vec.squeeze().cpu().numpy())

        return action, log_prob or 0.0

    def update(
        self,
        states: List[DesignState],
        actions: List[DesignAction],
        rewards: List[float],
        next_states: List[DesignState],
        dones: List[bool],
        importance_weights: List[float] = None
    ):
        """
        Update policy using PPO.

        Args:
            states: List of states
            actions: List of actions
            rewards: List of rewards
            next_states: List of next states
            dones: List of done flags
            importance_weights: IS weights for off-policy correction
        """
        if not states:
            return

        # Convert to tensors
        state_tensors = torch.FloatTensor(np.array([s.to_vector() for s in states])).to(self.device)
        action_tensors = torch.FloatTensor(np.array([a.to_vector() for a in actions])).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Compute values
        with torch.no_grad():
            _, values = self.policy(state_tensors)
            values = values.squeeze()
            next_values, _ = self.policy(
                torch.FloatTensor(np.array([ns.to_vector() for ns in next_states])).to(self.device)
            )
            next_values = next_values.squeeze()

        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0

        for i in reversed(range(len(states))):
            delta = rewards_tensor[i] + self.gamma * next_values[i] * (1 - dones_tensor[i]) - values[i]
            gae = delta + self.gamma * gae * (1 - dones_tensor[i])
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(4):  # Multiple epochs
            # Get action probabilities
            log_probs, values_pred = self.policy.evaluate_actions(state_tensors, action_tensors)
            log_probs = log_probs.squeeze()

            # Importance sampling weights
            if importance_weights is not None:
                is_weights = torch.FloatTensor(importance_weights).to(self.device)
            else:
                is_weights = torch.ones_like(log_probs)

            # PPO clipped objective
            ratio = torch.exp(log_probs - log_probs.detach())
            surr1 = ratio * advantages * is_weights
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages * is_weights
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = self.value_coef * F.mse_loss(values_pred.squeeze(), returns)

            # Entropy bonus
            entropy = -torch.distributions.Normal(
                self.policy.policy_head,
                torch.exp(self.policy.log_std)
            ).entropy().mean()
            entropy_loss = -self.entropy_coef * entropy

            # Total loss
            loss = policy_loss + value_loss + entropy_loss

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.clip_grad)
            self.optimizer.step()

            self.training_step += 1

        # Update target network periodically
        if self.training_step % 100 == 0:
            self.target_value.soft_update(self.value_net)

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "value_state_dict": self.value_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_step": self.training_step
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_net.load_state_dict(checkpoint["value_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.training_step = checkpoint["training_step"]


class DDPGAgent:
    """
    DDPG (Deep Deterministic Policy Gradient) Agent.

    For continuous control with deterministic policy.
    """

    def __init__(
        self,
        state_dim: int = 28,
        action_dim: int = 20,
        hidden_dims: List[int] = None,
        lr_actor: float = 1e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        use_gpu: bool = True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")

        # Actor (policy)
        self.actor = PiERNPolicyNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic (Q-function)
        self.critic = PiERNPolicyNetwork(state_dim + action_dim, 1, hidden_dims).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau

        self.replay_buffer = ReplayBuffer(capacity=50000)

    def select_action(self, state: DesignState, noise_scale: float = 0.1) -> DesignAction:
        """Select action with exploration noise."""
        state_vec = torch.FloatTensor(state.to_vector()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_mean, _ = self.actor(state_vec)

        # Add noise for exploration
        noise = torch.randn_like(action_mean) * noise_scale
        action_vec = (action_mean + noise).cpu().numpy()

        action = DesignAction.from_vector(action_vec.squeeze())

        # Clip parameters to valid range
        action.param_delta = np.clip(action.param_delta, -0.5, 0.5)

        return action

    def update(self, batch_size: int = 64):
        """Update networks using sampled batch."""
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        state_tensors = torch.FloatTensor(np.array([s.to_vector() for s in states])).to(self.device)
        action_tensors = torch.FloatTensor(np.array([a.to_vector() for a in actions])).to(self.device)
        rewards_tensor = torch.FloatTensor(rewards).to(self.device)
        next_state_tensors = torch.FloatTensor(np.array([s.to_vector() for s in next_states])).to(self.device)
        dones_tensor = torch.FloatTensor(dones).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions, _ = self.actor_target(next_state_tensors)
            target_q = self.critic_target(torch.cat([next_state_tensors, next_actions], dim=-1))
            target_q = rewards_tensor.unsqueeze(1) + self.gamma * target_q * (1 - dones_tensor.unsqueeze(1))

        current_q = self.critic(torch.cat([state_tensors, action_tensors], dim=-1))
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actions_pred, _ = self.actor(state_tensors)
        actor_loss = -self.critic(torch.cat([state_tensors, actions_pred], dim=-1)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
