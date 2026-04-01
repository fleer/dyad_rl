import random
from collections import namedtuple
from typing import Callable

import numpy as np
import torch

Transition = namedtuple(
    "Transition",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "next_action_mask",
        "state_discrete",
        "next_state_discrete",
        "state_rgb",
        "next_state_rgb",
    ],
)

HERTransition = namedtuple(
    "HERTransition",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "next_action_mask",
        "goal",
        "next_goal",
        "state_discrete",
        "next_state_discrete",
        "state_rgb",
        "next_state_rgb",
    ],
)

Episode = namedtuple(
    "Episode",
    [
        "states",
        "actions",
        "goals",
        "achieved_goals",
        "states_discrete",
        "states_rgb",
    ],
)


class ReplayBuffer:
    """Fixed-size circular replay buffer storing transitions with both observation types."""

    def __init__(self, capacity: int):
        """Initialize Replay Buffer.

        Creates a fixed-size circular replay buffer.

        Args:
            capacity (int): Maximum number of stored transitions.

        Returns:
            None: Buffer state is initialized in place.
        """
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
        next_action_mask: np.ndarray | None = None,
        state_discrete: np.ndarray | None = None,
        next_state_discrete: np.ndarray | None = None,
        state_rgb: np.ndarray | None = None,
        next_state_rgb: np.ndarray | None = None,
    ) -> None:
        """Push Transition.

        Stores one transition in the circular replay buffer.

        Args:
            state (np.ndarray): Current observation.
            action (int): Action index.
            reward (float): Immediate reward.
            next_state (np.ndarray): Next observation.
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
            state,
            action,
            reward,
            next_state,
            done,
            next_action_mask,
            state_discrete,
            next_state_discrete,
            state_rgb,
            next_state_rgb,
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def extend(self, transitions: list[Transition]) -> None:
        """Extend Replay Buffer.

        Appends multiple transitions into the circular buffer.

        Args:
            transitions (list[Transition]): Transitions to insert.

        Returns:
            None: Transitions are inserted into storage.
        """
        for t in transitions:
            if len(self.buffer) < self.capacity:
                self.buffer.append(t)
            else:
                self.buffer[self.position] = t
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
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


class HERReplayBuffer:
    """Hindsight Experience Replay (HER) buffer for goal-conditioned RL.

    Paper: https://arxiv.org/abs/1707.01495
    Reference: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/her/her_replay_buffer.py

    Stores episodes and relabels transitions with alternative goals sampled from
    achieved states in the episode. This allows learning from sparse, binary rewards
    by treating failed attempts at reaching one goal as successful attempts at
    reaching an achieved state as a goal.
    """

    def __init__(
        self,
        capacity: int,
        reward_fn: Callable[[np.ndarray, np.ndarray], float],
        n_sampled_goal: int = 4,
        goal_selection_strategy: str = "future",
    ):
        """Initialize HER Replay Buffer.

        Creates a HER-enabled replay buffer with configurable goal relabeling.

        Args:
            capacity (int): Maximum number of transitions to store.
            reward_fn (Callable[[np.ndarray, np.ndarray], float]): Reward
                function over achieved and target goals.
            n_sampled_goal (int): Number of hindsight goals sampled per real
                transition.
            goal_selection_strategy (str): Strategy for selecting hindsight
                goals.
                - "future": sample goals from states after current step in episode (default)
                - "final": always use final achieved state as goal
                - "episode": sample uniformly from all achieved states in episode

        Returns:
            None: Buffer state is initialized in place.
        """
        self.capacity = capacity
        self.reward_fn = reward_fn
        self.n_sampled_goal = max(1, n_sampled_goal)
        # Her ratio represents the fraction of virtual transitions
        self.her_ratio = 1.0 - (1.0 / (self.n_sampled_goal + 1))
        self.goal_selection_strategy = goal_selection_strategy.lower()

        if self.goal_selection_strategy not in ["future", "final", "episode"]:
            raise ValueError(
                f"goal_selection_strategy must be 'future', 'final', or 'episode', "
                f"got {self.goal_selection_strategy}"
            )

        # Transition-level storage to remain drop-in compatible with ReplayBuffer.
        self.buffer: list[Transition] = []
        self.position = 0
        self._current_episode: list[Transition] = []

    def _append_transition(self, transition: Transition) -> None:
        """Append Transition Internally.

        Inserts one transition into circular storage.

        Args:
            transition (Transition): Transition to append.

        Returns:
            None: Transition is inserted into storage.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def _goal_from_transition(self, transition: Transition) -> np.ndarray:
        """Extract Achieved Goal.

        Extracts an achieved-goal representation from a transition using
        discrete, RGB, or generic next-state preference order.

        Args:
            transition (Transition): Transition containing next-state fields.

        Returns:
            np.ndarray: Achieved-goal representation.
        """
        if transition.next_state_discrete is not None:
            return np.asarray(transition.next_state_discrete)
        if transition.next_state_rgb is not None:
            return np.asarray(transition.next_state_rgb)
        return np.asarray(transition.next_state)

    def _sample_hindsight_goal(
        self,
        achieved_goals: list[np.ndarray],
        step_idx: int,
    ) -> np.ndarray:
        """Sample Hindsight Goal.

        Samples one hindsight goal according to the configured strategy.

        Args:
            achieved_goals (list[np.ndarray]): Achieved goals in an episode.
            step_idx (int): Current transition index.

        Returns:
            np.ndarray: Selected hindsight goal.
        """
        if self.goal_selection_strategy == "final":
            return achieved_goals[-1]
        if self.goal_selection_strategy == "episode":
            return achieved_goals[np.random.randint(0, len(achieved_goals))]
        # "future" strategy (inclusive)
        return achieved_goals[np.random.randint(step_idx, len(achieved_goals))]

    def _augment_current_episode(self) -> None:
        """Augment Episode With HER.

        Generates HER virtual transitions from the currently buffered episode.

        Args:
            None: This method uses internal episode state.

        Returns:
            None: Virtual transitions are appended to storage.
        """
        if not self._current_episode:
            return

        achieved_goals = [self._goal_from_transition(t) for t in self._current_episode]

        for step_idx, transition in enumerate(self._current_episode):
            achieved_goal = achieved_goals[step_idx]
            for _ in range(self.n_sampled_goal):
                hindsight_goal = self._sample_hindsight_goal(achieved_goals, step_idx)
                her_reward = float(self.reward_fn(achieved_goal, hindsight_goal))
                her_transition = Transition(
                    state=transition.state,
                    action=transition.action,
                    reward=her_reward,
                    next_state=transition.next_state,
                    done=transition.done,
                    next_action_mask=transition.next_action_mask,
                    state_discrete=transition.state_discrete,
                    next_state_discrete=transition.next_state_discrete,
                    state_rgb=transition.state_rgb,
                    next_state_rgb=transition.next_state_rgb,
                )
                self._append_transition(her_transition)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
        state_discrete: np.ndarray | None = None,
        next_state_discrete: np.ndarray | None = None,
        state_rgb: np.ndarray | None = None,
        next_state_rgb: np.ndarray | None = None,
    ) -> None:
        """Push HER Transition.

        Stores one transition and, at episode end, augments with HER relabeled
        transitions.

        Args:
            state (np.ndarray): Current observation.
            action (int): Action index.
            reward (float): Immediate reward.
            next_state (np.ndarray): Next observation.
            done (bool): Whether transition is terminal.
            next_action_mask (np.ndarray | None): Valid action mask for next
                state.
            state_discrete (np.ndarray | None): Optional discrete-state branch.
            next_state_discrete (np.ndarray | None): Optional next
                discrete-state branch.
            state_rgb (np.ndarray | None): Optional RGB-state branch.
            next_state_rgb (np.ndarray | None): Optional next RGB-state branch.

        Returns:
            None: Transition and optional HER augmentations are stored.
        """
        transition = Transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            next_action_mask=next_action_mask,
            state_discrete=state_discrete,
            next_state_discrete=next_state_discrete,
            state_rgb=state_rgb,
            next_state_rgb=next_state_rgb,
        )
        self._append_transition(transition)
        self._current_episode.append(transition)

        if done:
            self._augment_current_episode()
            self._current_episode.clear()

    def extend(self, transitions: list[Transition]) -> None:
        """Extend HER Buffer.

        Inserts pre-built transitions while preserving HER episode handling.

        Args:
            transitions (list[Transition]): Transitions to insert.

        Returns:
            None: Transitions are inserted into storage.
        """
        for t in transitions:
            self.push(
                state=t.state,
                action=t.action,
                reward=t.reward,
                next_state=t.next_state,
                done=t.done,
                next_action_mask=t.next_action_mask,
                state_discrete=t.state_discrete,
                next_state_discrete=t.next_state_discrete,
                state_rgb=t.state_rgb,
                next_state_rgb=t.next_state_rgb,
            )

    def push_episode(
        self,
        states: list[np.ndarray],
        actions: list[int],
        goals: list[np.ndarray],
        achieved_goals: list[np.ndarray],
        states_discrete: list[np.ndarray] | None = None,
        states_rgb: list[np.ndarray] | None = None,
    ) -> None:
        """Push Full Episode.

        Stores a full episode by converting sequential states to transitions.

        Args:
            states (list[np.ndarray]): Observations at each step.
            actions (list[int]): Actions at each step.
            goals (list[np.ndarray]): Original goals at each step.
            achieved_goals (list[np.ndarray]): Achieved goals at each step.
            states_discrete (list[np.ndarray] | None): Optional discrete
                observations.
            states_rgb (list[np.ndarray] | None): Optional RGB observations.

        Returns:
            None: Episode transitions are stored via ``push``.
        """
        if not states:
            return

        if len(states) < 2:
            # Need at least 2 states to form a transition
            return

        if states_discrete is None:
            states_discrete = [None] * len(states)
        if states_rgb is None:
            states_rgb = [None] * len(states)

        episode_len = len(states)
        for i in range(episode_len - 1):
            done = i == episode_len - 2
            reward = float(
                self.reward_fn(np.asarray(achieved_goals[i]), np.asarray(goals[i]))
            )
            self.push(
                state=states[i],
                action=int(actions[i]),
                reward=reward,
                next_state=states[i + 1],
                done=done,
                next_action_mask=None,
                state_discrete=states_discrete[i],
                next_state_discrete=states_discrete[i + 1],
                state_rgb=states_rgb[i],
                next_state_rgb=states_rgb[i + 1],
            )

    def _select_hindsight_goal(self, episode: Episode, current_step: int) -> np.ndarray:
        """Select Hindsight Goal.

        Selects a hindsight goal from achieved states in an episode.

        Args:
            episode (Episode): Episode data source.
            current_step (int): Current step index.

        Returns:
            np.ndarray: Selected hindsight goal.
        """
        ep_len = len(episode.achieved_goals)

        if self.goal_selection_strategy == "final":
            # Always use the final achieved state goal
            return episode.achieved_goals[-1]

        elif self.goal_selection_strategy == "future":
            # Sample from current step onwards (inclusive of current step)
            # This allows the agent to replay with any future state as goal
            selected_step = np.random.randint(current_step, ep_len)
            return episode.achieved_goals[selected_step]

        elif self.goal_selection_strategy == "episode":
            # Sample uniformly from all achieved goals in episode
            selected_step = np.random.randint(0, ep_len)
            return episode.achieved_goals[selected_step]

    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Sample HER Batch.

        Samples a minibatch from storage and converts fields to tensors.

        Args:
            batch_size (int): Number of transitions to sample.
            device (torch.device): Target device for tensors.

        Returns:
            dict[str, torch.Tensor]: Tensor batch with transition fields and
            optional modality branches.
        """
        if not self.buffer:
            raise RuntimeError("Cannot sample from empty buffer")

        batch = random.sample(self.buffer, batch_size)

        all_states = np.array([t.state for t in batch])
        all_actions = np.array([t.action for t in batch])
        all_rewards = np.array([t.reward for t in batch])
        all_next_states = np.array([t.next_state for t in batch])
        all_goals = np.zeros((batch_size, 1), dtype=np.float32)

        all_states_discrete = [t.state_discrete for t in batch]
        all_next_states_discrete = [t.next_state_discrete for t in batch]
        all_states_rgb = [t.state_rgb for t in batch]
        all_next_states_rgb = [t.next_state_rgb for t in batch]
        next_action_masks_raw = [t.next_action_mask for t in batch]

        # Convert to tensors
        states = torch.as_tensor(all_states, dtype=torch.float32, device=device)
        actions = torch.as_tensor(
            all_actions, dtype=torch.long, device=device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            all_rewards, dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.as_tensor(
            all_next_states, dtype=torch.float32, device=device
        )
        goals = torch.as_tensor(all_goals, dtype=torch.float32, device=device)

        dones = torch.as_tensor(
            np.array([t.done for t in batch]), dtype=torch.bool, device=device
        ).unsqueeze(1)
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
            "goals": goals,
            "states_discrete": all_states_discrete,
            "next_states_discrete": all_next_states_discrete,
            "states_rgb": all_states_rgb,
            "next_states_rgb": all_next_states_rgb,
        }

    def _sample_transitions(
        self, batch_size: int, use_hindsight: bool
    ) -> dict[str, np.ndarray | list]:
        """Sample Transition Arrays.

        Internal helper that samples transitions and returns numpy-backed fields.

        Args:
            batch_size (int): Number of transitions to sample.
            use_hindsight (bool): Placeholder flag for hindsight relabeling
                behavior.

        Returns:
            dict[str, np.ndarray | list]: Sampled transition fields.
        """
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_goals = []
        batch_states_discrete = []
        batch_next_states_discrete = []
        batch_states_rgb = []
        batch_next_states_rgb = []

        sampled = random.sample(self.buffer, batch_size)
        for t in sampled:
            batch_states.append(t.state)
            batch_actions.append(t.action)
            batch_rewards.append(t.reward)
            batch_next_states.append(t.next_state)
            batch_goals.append(np.zeros(1, dtype=np.float32))
            batch_states_discrete.append(t.state_discrete)
            batch_next_states_discrete.append(t.next_state_discrete)
            batch_states_rgb.append(t.state_rgb)
            batch_next_states_rgb.append(t.next_state_rgb)

        return {
            "states": np.array(batch_states),
            "actions": np.array(batch_actions),
            "rewards": np.array(batch_rewards),
            "next_states": np.array(batch_next_states),
            "goals": np.array(batch_goals),
            "states_discrete": batch_states_discrete,
            "next_states_discrete": batch_next_states_discrete,
            "states_rgb": batch_states_rgb,
            "next_states_rgb": batch_next_states_rgb,
        }

    def sample_transitions(self, batch_size: int) -> list[HERTransition]:
        """Sample HER Transition Objects.

        Samples transitions and packages them as ``HERTransition`` objects.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list[HERTransition]: Sampled transition objects.
        """
        if not self.buffer:
            raise RuntimeError("Cannot sample from empty buffer")

        transitions = []
        sampled = random.sample(self.buffer, batch_size)
        for t in sampled:
            transitions.append(
                HERTransition(
                    state=t.state,
                    action=t.action,
                    reward=t.reward,
                    next_state=t.next_state,
                    done=t.done,
                    next_action_mask=t.next_action_mask,
                    goal=np.zeros(1, dtype=np.float32),
                    next_goal=np.zeros(1, dtype=np.float32),
                    state_discrete=t.state_discrete,
                    next_state_discrete=t.next_state_discrete,
                    state_rgb=t.state_rgb,
                    next_state_rgb=t.next_state_rgb,
                )
            )
        return transitions

    def _sample_single_transition(self, use_hindsight: bool) -> HERTransition:
        """Sample Single Transition.

        Samples one transition and packages it as ``HERTransition``.

        Args:
            use_hindsight (bool): Placeholder flag for hindsight relabeling
                behavior.

        Returns:
            HERTransition: Sampled transition object.
        """
        transition = random.choice(self.buffer)
        return HERTransition(
            state=transition.state,
            action=transition.action,
            reward=transition.reward,
            next_state=transition.next_state,
            done=transition.done,
            next_action_mask=transition.next_action_mask,
            goal=np.zeros(1, dtype=np.float32),
            next_goal=np.zeros(1, dtype=np.float32),
            state_discrete=transition.state_discrete,
            next_state_discrete=transition.next_state_discrete,
            state_rgb=transition.state_rgb,
            next_state_rgb=transition.next_state_rgb,
        )

    def __len__(self) -> int:
        """Get Buffer Length.

        Returns the number of transitions currently stored.

        Args:
            None: This method reads internal buffer state.

        Returns:
            int: Number of stored transitions.
        """
        return len(self.buffer)
