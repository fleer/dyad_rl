import random
from collections import namedtuple, deque

import numpy as np
import torch

Transition = namedtuple(
    "Transition",
    [
        "action",
        "reward",
        "done",
        "next_action_mask",
        "state_discrete",
        "next_state_discrete",
        "state_rgb",
        "next_state_rgb",
    ],
)


class ReplayBuffer:
    """Fixed-size circular replay buffer storing transitions with both observation types."""

    def __init__(self, capacity: int, observation_type: str | None = None) -> None:
        """Initialize Replay Buffer.

        Creates a fixed-size circular replay buffer.

        Args:
            capacity (int): Maximum number of stored transitions.
            observation_type (str | None): Type of observation stored ("puzzle_state",
                "rgb"). If None, no special handling is applied to next-action
                masks during sampling.

        Returns:
            None: Buffer state is initialized in place.
        """
        self.buffer = deque(maxlen=capacity)
        self.observation_type = observation_type

    def _trim_transition(self, transition: Transition) -> Transition:
        """Trim Transition Payload.

        Keeps only the observation modality needed by this buffer so replay
        storage does not retain duplicate state branches.

        Args:
            transition (Transition): Transition to normalize for buffer
                storage.

        Returns:
            Transition: Transition with unused observation branches removed.
        """
        if self.observation_type == "rgb":
            return Transition(
                action=transition.action,
                reward=transition.reward,
                done=transition.done,
                next_action_mask=transition.next_action_mask,
                state_discrete=None,
                next_state_discrete=None,
                state_rgb=transition.state_rgb,
                next_state_rgb=transition.next_state_rgb,
            )
        if self.observation_type == "puzzle_state":
            return Transition(
                action=transition.action,
                reward=transition.reward,
                done=transition.done,
                next_action_mask=transition.next_action_mask,
                state_discrete=transition.state_discrete,
                next_state_discrete=transition.next_state_discrete,
                state_rgb=None,
                next_state_rgb=None,
            )
        return transition

    # TODO: Remobe state and next_ste as it handles redundant data and only use
    # state_discrete and state_rgb
    def push(
        self,
        action: int,
        reward: float,
        done: bool,
        next_action_mask: np.ndarray | None = None,
        state_discrete: np.ndarray | None = None,
        next_state_discrete: np.ndarray | None = None,
        state_rgb: np.ndarray | None = None,
        next_state_rgb: np.ndarray | None = None,
    ) -> None:
        """Push Transition.

        Stores one transition in the circular replay buffer.

        Args:
            action (int): Action index.
            reward (float): Immediate reward.
            done (bool): Whether transition is terminal.
            next_action_mask (np.ndarray | None): Valid action mask for next
                state.
            state_discrete (np.ndarray | None): Optional discrete-state branch.
            next_state_discrete (np.ndarray | None): Optional next
                discrete-state branch.
            state_rgb (np.ndarray | None): Optional RGB-state branch.
            next_state_rgb (np.ndarray | None): Optional next RGB-state branch.

        Returns:
            None: Transition is inserted into storage.
        """
        transition = Transition(
            action,
            reward,
            done,
            next_action_mask,
            state_discrete,
            next_state_discrete,
            state_rgb,
            next_state_rgb,
            # [],
            # [],
        )
        self.buffer.append(self._trim_transition(transition))

    def extend(self, transitions: list[Transition]) -> None:
        """Extend Replay Buffer.

        Appends multiple transitions into the circular buffer.

        Args:
            transitions (list[Transition]): Transitions to insert.

        Returns:
            None: Transitions are inserted into storage.
        """
        for t in transitions:
            self.buffer.append(self._trim_transition(t))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> dict[str, torch.Tensor | None]:
        """Sample Transition Batch.

        Samples a random minibatch and converts fields to tensors.

        Args:
            batch_size (int): Number of transitions to sample.
            device (torch.device): Target device for tensor outputs.

        Returns:
            dict[str, torch.Tensor]: Tensor batch containing state, action,
            reward, next-state, done, and optional next-action masks.
        """
        batch = random.sample(self.buffer, batch_size)

        if self.observation_type == "rgb":
            # For RGB, stack along channel dimension
            raw_state_list = [t.state_rgb for t in batch]
            raw_next_state_list = [t.next_state_rgb for t in batch]
        elif self.observation_type == "puzzle_state":
            # For puzzle_state, stack along feature dimension
            raw_state_list = [t.state_discrete for t in batch]
            raw_next_state_list = [t.next_state_discrete for t in batch]
        else:
            raise ValueError(
                f"Unsupported observation_type for sampling: {self.observation_type}"
            )

        states = torch.as_tensor(
            np.array(raw_state_list), dtype=torch.float32, device=device
        )
        actions = torch.as_tensor(
            np.array([t.action for t in batch]), dtype=torch.long, device=device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            np.array([t.reward for t in batch]), dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            np.array(raw_next_state_list), dtype=torch.float32, device=device
        )
        dones = torch.as_tensor(
            np.array([t.done for t in batch]), dtype=torch.bool, device=device
        ).unsqueeze(1)
        next_action_masks_raw = [t.next_action_mask for t in batch]
        next_action_masks = None
        inferred_mask_size = next(
            (len(mask) for mask in next_action_masks_raw if mask is not None),
            None,
        )
        if inferred_mask_size is not None:
            next_action_masks = torch.as_tensor(
                np.array(
                    [
                        mask
                        if mask is not None
                        else np.ones(inferred_mask_size, dtype=bool)
                        for mask in next_action_masks_raw
                    ],
                    dtype=bool,
                ),
                dtype=torch.bool,
                device=device,
            )

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "next_action_masks": next_action_masks,
        }

    def __len__(self) -> int:
        """Get Buffer Length.

        Returns the number of transitions currently stored.

        Args:
            None: This method reads internal buffer state.

        Returns:
            int: Number of stored transitions.
        """
        return len(self.buffer)
