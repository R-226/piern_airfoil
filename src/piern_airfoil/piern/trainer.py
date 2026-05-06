"""
Trainer for PiERN agent.

Implements the training loop for physics-informed reinforcement learning.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
import time

from .agent import PPOAgent, DDPGAgent
from .state_representation import DesignState


@dataclass
class TrainingConfig:
    """Configuration for PiERN training."""
    algorithm: str = "PPO"           # "PPO" or "DDPG"
    n_episodes: int = 1000          # Number of training episodes
    max_steps_per_episode: int = 200  # Max steps per episode
    batch_size: int = 64            # Batch size for updates
    warmup_steps: int = 100         # Steps before training starts
    update_interval: int = 10       # Steps between updates
    target_update_interval: int = 100  # Steps between target updates
    save_interval: int = 500        # Steps between saves
    log_interval: int = 10          # Steps between logs
    eval_interval: int = 50         # Episodes between evaluations


class PiERNTrainer:
    """
    Trainer for PiERN agent.

    Handles the training loop, experience collection, and evaluation.
    """

    def __init__(
        self,
        agent: PPOAgent,
        parameterization: Parameterization,
        analyzer: Callable,  # Function that takes (params, conditions) -> AnalysisResult
        constraints: List[Callable] = None,
        config: TrainingConfig = None
    ):
        """
        Initialize trainer.

        Args:
            agent: RL agent to train
            parameterization: Parameterization method
            analyzer: Analysis function
            constraints: List of constraint functions
            config: Training configuration
        """
        self.agent = agent
        self.param = parameterization
        self.analyzer = analyzer
        self.constraints = constraints or []
        self.config = config or TrainingConfig()

        # Training state
        self.global_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        self.best_params = None

    def collect_experience(
        self,
        initial_params: np.ndarray,
        conditions: FlowConditions,
        max_steps: int,
        epsilon: float = 0.1
    ) -> Dict:
        """
        Collect experience for one episode.

        Args:
            initial_params: Starting parameters
            conditions: Flow conditions
            max_steps: Maximum steps to take
            epsilon: Exploration rate

        Returns:
            Dict with episode data
        """
        # Initialize
        params = initial_params.copy()
        geometry = self.param.params_to_geometry(params)
        result = self.analyzer(geometry, conditions)
        state = DesignState.from_geometry_and_result(geometry, result, self.constraints)

        # Episode data
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        episode_reward = 0

        for step in range(max_steps):
            # Select action
            action, log_prob = self.agent.select_action(state, epsilon=epsilon)

            # Apply action
            new_params = action.apply_to_params(params)

            # Check validity
            is_valid, _ = self.param.validate(new_params)
            if not is_valid:
                # Invalid action, give penalty and don't update
                reward = -10.0
                done = False
                new_state = state
            else:
                # Valid action, evaluate
                new_geometry = self.param.params_to_geometry(new_params)
                new_result = self.analyzer(new_geometry, conditions)
                new_state = DesignState.from_geometry_and_result(
                    new_geometry, new_result, self.constraints
                )

                # Compute reward
                reward = new_state.compute_reward(
                    target_CL=0.8,
                    constraint_weight=10.0
                )

                # Check if constraint satisfied (episode success)
                done = new_state.is_valid() and abs(new_result.CL - 0.8) < 0.05

            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(new_state)
            dones.append(done)

            # Update
            params = new_params
            geometry = new_geometry
            result = new_result
            state = new_state
            episode_reward += reward

            self.global_step += 1

            if done:
                break

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "episode_reward": episode_reward,
            "episode_length": len(states)
        }

    def train(
        self,
        initial_params: np.ndarray,
        conditions: FlowConditions,
        callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run training loop.

        Args:
            initial_params: Initial parameter guess
            conditions: Flow conditions
            callback: Called after each episode

        Returns:
            Training history
        """
        history = {
            "episode_rewards": [],
            "episode_lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "best_reward": []
        }

        for episode in range(self.config.n_episodes):
            # Collect experience
            episode_data = self.collect_experience(
                initial_params,
                conditions,
                self.config.max_steps_per_episode,
                epsilon=max(0.01, 0.5 - episode * 0.0005)  # Decaying epsilon
            )

            # Store episode reward
            self.episode_rewards.append(episode_data["episode_reward"])
            self.episode_lengths.append(episode_data["episode_length"])

            # Update agent (after warmup)
            if episode > 0 and episode % self.config.update_interval == 0:
                if self.config.algorithm == "PPO":
                    update_stats = self.agent.update(
                        episode_data["states"],
                        episode_data["actions"],
                        episode_data["rewards"],
                        episode_data["next_states"],
                        episode_data["dones"]
                    )
                    history["policy_losses"].append(update_stats.get("policy_loss", 0))
                    history["value_losses"].append(update_stats.get("value_loss", 0))
                elif self.config.algorithm == "DDPG":
                    # Add to replay buffer
                    for i in range(len(episode_data["states"])):
                        self.agent.replay_buffer.add(
                            episode_data["states"][i],
                            episode_data["actions"][i],
                            episode_data["rewards"][i],
                            episode_data["next_states"][i],
                            episode_data["dones"][i]
                        )
                    # Update
                    update_stats = self.agent.update(self.config.batch_size)
                    history["policy_losses"].append(update_stats.get("actor_loss", 0))
                    history["value_losses"].append(update_stats.get("critic_loss", 0))

            # Track best
            if episode_data["episode_reward"] > self.best_reward:
                self.best_reward = episode_data["episode_reward"]
                self.best_params = episode_data["states"][-1].to_vector()[:18] if episode_data["states"] else initial_params

            history["best_reward"].append(self.best_reward)

            # Logging
            if episode % self.config.log_interval == 0:
                print(f"Episode {episode}: "
                      f"reward={episode_data['episode_reward']:.2f}, "
                      f"length={episode_data['episode_length']}, "
                      f"best={self.best_reward:.2f}")

            # Callback
            if callback:
                callback(episode, episode_data)

        return history

    def evaluate(self, params: np.ndarray, conditions: FlowConditions, n_runs: int = 5) -> Dict:
        """
        Evaluate current policy.

        Args:
            params: Parameters to evaluate
            conditions: Flow conditions
            n_runs: Number of evaluation runs

        Returns:
            Evaluation metrics
        """
        rewards = []
        cl_values = []
        cd_values = []

        for _ in range(n_runs):
            geometry = self.param.params_to_geometry(params)
            result = self.analyzer(geometry, conditions)
            state = DesignState.from_geometry_and_result(geometry, result, self.constraints)
            reward = state.compute_reward(target_CL=0.8)

            rewards.append(reward)
            cl_values.append(result.CL)
            cd_values.append(result.CD)

        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_CL": np.mean(cl_values),
            "mean_CD": np.mean(cd_values)
        }


class CurriculumTrainer:
    """
    Curriculum learning trainer.

    Gradually increases task difficulty during training.
    """

    def __init__(
        self,
        agent: PPOAgent,
        parameterization: Parameterization,
        analyzer: Callable,
        constraints: List[Callable] = None
    ):
        self.agent = agent
        self.param = parameterization
        self.analyzer = analyzer
        self.constraints = constraints or []

        # Curriculum stages
        self.stages = [
            {"alpha_range": (0, 5), "Re_range": (1e6, 3e6), "complexity": "low"},
            {"alpha_range": (-5, 10), "Re_range": (1e6, 5e6), "complexity": "medium"},
            {"alpha_range": (-10, 15), "Re_range": (1e6, 10e6), "complexity": "high"}
        ]

    def train_with_curriculum(
        self,
        n_episodes_per_stage: int = 300
    ) -> Dict:
        """Train with curriculum learning."""
        history = {}

        for stage_idx, stage in enumerate(self.stages):
            print(f"\n=== Curriculum Stage {stage_idx + 1}: {stage['complexity']} ===")

            # Generate random conditions within stage range
            def get_conditions():
                alpha = np.random.uniform(*stage["alpha_range"])
                Re = np.random.uniform(*stage["Re_range"])
                return FlowConditions(alpha=alpha, Re=Re)

            # Initial params
            initial_params = self.param.random().to_array()

            # Create trainer for this stage
            trainer = PiERNTrainer(
                self.agent,
                self.param,
                self.analyzer,
                self.constraints
            )

            # Train
            stage_history = trainer.train(initial_params, get_conditions())

            history[f"stage_{stage_idx}"] = stage_history

            print(f"Stage {stage_idx + 1} complete: "
                  f"best reward = {max(stage_history['episode_rewards']):.2f}")

        return history
