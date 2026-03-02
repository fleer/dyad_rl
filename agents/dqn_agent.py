import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig

from agents.networks import MLPNetwork, CNNNetwork
from utils.replay_buffer import ReplayBuffer, Transition


class DQNAgent:
    """DQN agent with experience replay, target network, and action masking."""

    def __init__(
        self,
        obs_type: str,
        obs_shape: tuple[int, ...],
        num_actions: int,
        cfg: DictConfig,
        agent_cfg: DictConfig | None = None,
        device: torch.device | None = None,
    ):
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device or torch.device("cpu")

        # Use agent_cfg if provided (for dyad with different configs), else cfg.agent
        a_cfg = agent_cfg if agent_cfg is not None else cfg.agent

        # Hyperparameters from agent config
        self.batch_size = a_cfg.batch_size
        self.gamma = a_cfg.gamma
        self.eps_start = a_cfg.eps_start
        self.eps_end = a_cfg.eps_end
        self.eps_decay = a_cfg.eps_decay
        self.tau = a_cfg.tau
        self.lr = a_cfg.lr
        self.buffer_size = a_cfg.buffer_size

        # Build networks
        if a_cfg.type == "mlp":
            input_dim = int(np.prod(obs_shape))
            self.policy_net = MLPNetwork(
                input_dim, num_actions, list(a_cfg.hidden_sizes)
            ).to(self.device)
            self.target_net = MLPNetwork(
                input_dim, num_actions, list(a_cfg.hidden_sizes)
            ).to(self.device)
        elif a_cfg.type == "cnn":
            c, h, w = obs_shape
            self.policy_net = CNNNetwork(
                c, h, w, num_actions,
                list(a_cfg.conv_channels),
                list(a_cfg.conv_kernels),
                list(a_cfg.conv_strides),
                a_cfg.fc_hidden,
            ).to(self.device)
            self.target_net = CNNNetwork(
                c, h, w, num_actions,
                list(a_cfg.conv_channels),
                list(a_cfg.conv_kernels),
                list(a_cfg.conv_strides),
                a_cfg.fc_hidden,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown agent type: {a_cfg.type}")

        # Initialize target with policy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(self.buffer_size)
        self.steps_done = 0

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray | None = None,
        explore: bool = True,
    ) -> int:
        """Epsilon-greedy action selection with optional action masking."""
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -self.steps_done / self.eps_decay
        )
        self.current_epsilon = eps
        self.steps_done += 1

        if explore and np.random.random() < eps:
            # Random action from valid actions
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                return int(np.random.choice(valid))
            return int(np.random.randint(self.num_actions))

        # Greedy action with masking
        with torch.no_grad():
            state_t = torch.as_tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.policy_net(state_t)
            if action_mask is not None:
                mask_t = torch.as_tensor(
                    action_mask, dtype=torch.bool, device=self.device
                )
                q_values[0][~mask_t] = float("-inf")
            return int(q_values.argmax(dim=1).item())

    def optimize(self) -> float:
        """Sample a batch from replay buffer and perform one gradient step.

        Returns the loss value, or 0.0 if the buffer has insufficient samples.
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        states = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        dones = batch["dones"]

        # Q(s, a) from policy net
        q_values = self.policy_net(states).gather(1, actions)

        # max_a' Q_target(s', a') — no action masking on target for simplicity
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True).values
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return loss.item()

    def update_target_net(self) -> None:
        """Soft-update target network: theta_target = tau*theta_policy + (1-tau)*theta_target."""
        for tp, pp in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

    # --- Dyad support methods ---

    def compute_q_values(self, states: np.ndarray) -> torch.Tensor:
        """Compute Q-values for a batch of states using the policy network."""
        with torch.no_grad():
            states_t = torch.as_tensor(
                states, dtype=torch.float32, device=self.device
            )
            return self.policy_net(states_t)

    def compute_expected_return(self, transitions: list[Transition]) -> np.ndarray:
        """Evaluate transitions using own Q-network.

        For each transition, computes the expected return:
            Q(s, a) from own policy.
        Returns array of shape (len(transitions),).
        """
        states = np.array([t.state for t in transitions], dtype=np.float32)
        actions = np.array([t.action for t in transitions], dtype=np.int64)

        with torch.no_grad():
            states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
            actions_t = torch.as_tensor(
                actions, dtype=torch.long, device=self.device
            ).unsqueeze(1)
            q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        return q_values.cpu().numpy()

    def add_to_buffer(self, transitions: list[Transition]) -> None:
        """Add external transitions to the replay buffer."""
        self.replay_buffer.extend(transitions)

    # --- Serialization ---

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
