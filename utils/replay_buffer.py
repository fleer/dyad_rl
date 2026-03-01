import random
from collections import namedtuple

import numpy as np
import torch

Transition = namedtuple("Transition", [
    "state", "action", "reward", "next_state", "done",
    "state_discrete", "next_state_discrete",
    "state_rgb", "next_state_rgb",
])


class ReplayBuffer:
    """Fixed-size circular replay buffer storing transitions with both observation types."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        state_discrete: np.ndarray | None = None,
        next_state_discrete: np.ndarray | None = None,
        state_rgb: np.ndarray | None = None,
        next_state_rgb: np.ndarray | None = None,
    ) -> None:
        transition = Transition(
            state, action, reward, next_state, done,
            state_discrete, next_state_discrete,
            state_rgb, next_state_rgb,
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def extend(self, transitions: list[Transition]) -> None:
        for t in transitions:
            if len(self.buffer) < self.capacity:
                self.buffer.append(t)
            else:
                self.buffer[self.position] = t
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)

        states = torch.as_tensor(
            np.array([t.state for t in batch]), dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            np.array([t.action for t in batch]), dtype=torch.long, device=device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            np.array([t.reward for t in batch]), dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.array([t.next_state for t in batch]), dtype=torch.float32, device=device
        )
        dones = torch.as_tensor(
            np.array([t.done for t in batch]), dtype=torch.float32, device=device
        ).unsqueeze(1)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def sample_transitions(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)
