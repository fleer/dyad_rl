import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
import gymnasium as gym

from agents.networks import MLPNetwork, CNNNetwork
from utils.replay_buffer import ReplayBuffer, Transition

# By default, PyTorch 2.0+ may use 'medium' precision for matmul on float32 tensors,
# which can cause instability in RL training.
# Setting to 'high' ensures full float32 precision for matmul operations, improving stability at the cost of some performance.
torch.set_float32_matmul_precision("high")


class DQNAgent:
    """DQN agent with experience replay, target network, and action masking."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        num_actions: int,
        cfg: DictConfig,
        agent_cfg: DictConfig | None = None,
        device: torch.device | None = None,
        
    ):
        """Initialize DQN Agent.

        Initializes network architecture, replay buffer, optimizer, and
        exploration schedule for a DQN agent.

        Args:
            obs_shape (tuple[int, ...]): Shape of the processed observation
                expected by the policy network.
            num_actions (int): Number of discrete actions.
            cfg (DictConfig): Root Hydra configuration.
            agent_cfg (DictConfig | None): Optional agent-specific override
                configuration. If ``None``, ``cfg.agent`` is used.
            device (torch.device | None): Torch device for tensors and models.
                Defaults to CPU when ``None``.

        Returns:
            None: The constructor initializes the instance in place.
        """
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device or torch.device("cpu")
        

        # Use agent_cfg if provided (for dyad with different configs), else cfg.agent
        a_cfg = agent_cfg if agent_cfg is not None else cfg.agent

        assert a_cfg.buffer_size >= a_cfg.learning_starts, (
            "Buffer size must be larger than learning_starts to allow for initial population before training."
        )

        # Hyperparameters from agent config — names mirror SB3 DQN.__init__
        self.batch_size = a_cfg.batch_size
        self.gamma = a_cfg.gamma
        self.exploration_initial_eps = float(a_cfg.exploration_initial_eps)
        self.exploration_final_eps = float(a_cfg.exploration_final_eps)
        self.exploration_decay = max(1.0, float(a_cfg.exploration_decay))
        self.learning_rate = a_cfg.learning_rate
        self.buffer_size = a_cfg.buffer_size
        self.max_grad_norm = float(a_cfg.max_grad_norm)
        self.target_update_interval = int(a_cfg.target_update_interval)
        self._n_calls = 0
        self.obs_type = a_cfg.obs_type

        # Target tracking:
        #   - "standard": ordinary Double-DQN training
        #   - "sgt2": symmetric gradient target tracking (both networks train)
        #
        # Target update method:
        #   - "hard": copy policy -> target every target_update_interval calls
        #   - "polyak": soft update target <- (1-tau)*target + tau*policy
        self.target_tracking_type = str(a_cfg.get("target_tracking_type", "standard")).lower()
        if self.target_tracking_type not in {"standard", "sgt2"}:
            raise ValueError(
                "target_tracking_type must be 'standard' or 'sgt2', "
                f"got {self.target_tracking_type!r}"
            )

        self.target_update_method = str(
            a_cfg.get("target_update_method", "hard")
        ).lower()
        if self.target_update_method not in {"hard", "polyak"}:
            raise ValueError(
                "target_update_method must be 'hard' or 'polyak', "
                f"got {self.target_update_method!r}"
            )

        # Accept either `polyak_tau` or the shorter `tau` config name.
        self.polyak_tau = float(a_cfg.get("polyak_tau", a_cfg.get("tau", 0.005)))
        if not 0.0 < self.polyak_tau <= 1.0:
            raise ValueError(
                f"polyak_tau/tau must be in (0, 1], got {self.polyak_tau}"
            )

        # SGT2 coupling strength.
        self.beta = float(a_cfg.get("beta", 1.0))
        if self.beta < 0.0:
            raise ValueError(f"beta must be >= 0, got {self.beta}")

        # Build networks
        if a_cfg.type == "mlp":
            input_dim = int(np.prod(obs_shape))
            # Priority: net_arch (SB3 naming) → hidden_size+num_layers (sweep compat) → hidden_sizes (legacy)
            net_arch = getattr(a_cfg, "net_arch", None)
            hidden_sizes = list(net_arch)
            policy_net = MLPNetwork(input_dim, num_actions, hidden_sizes).to(
                self.device
            )
            target_net = MLPNetwork(input_dim, num_actions, hidden_sizes).to(
                self.device
            )
        elif a_cfg.type == "cnn":
            c, h, w = obs_shape
            policy_net = CNNNetwork(
                c,
                h,
                w,
                num_actions,
                list(a_cfg.conv_channels),
                list(a_cfg.conv_kernels),
                list(a_cfg.conv_strides),
                a_cfg.fc_hidden,
            ).to(self.device)
            target_net = CNNNetwork(
                c,
                h,
                w,
                num_actions,
                list(a_cfg.conv_channels),
                list(a_cfg.conv_kernels),
                list(a_cfg.conv_strides),
                a_cfg.fc_hidden,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown agent type: {a_cfg.type}")

        # SGT2 optimizes both Q networks. Standard DQN keeps the target frozen.
        if self.target_tracking_type == "sgt2":
            for p in policy_net.parameters():
                p.requires_grad = True
            for p in target_net.parameters():
                p.requires_grad = True
        else:
            for p in policy_net.parameters():
                p.requires_grad = True
            for p in target_net.parameters():
                p.requires_grad = False

        # Compile networks with TorchDynamo for speedup.
        # fullgraph=True is intentionally omitted: on ROCm (AMD GPU) it triggers
        # a whole-graph HIP/hipcc compilation that can consume 10–80+ GB of RAM.
        # The default mode with graph-break fallbacks is safe and far cheaper.
        # self.policy_net = torch.compile(policy_net)
        # self.target_net = torch.compile(target_net)
        self.policy_net = policy_net
        self.target_net = target_net


        # Initialize target with policy weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # SGT2 uses one optimizer for both networks because its symmetric
        # objective contains trainable policy and target Q-values.
        if self.target_tracking_type == "sgt2":
            self.optimizer = optim.AdamW(
                list(self.policy_net.parameters()) + list(self.target_net.parameters()),
                lr=self.learning_rate,
                amsgrad=True,
            )
        else:
            self.optimizer = optim.AdamW(
                self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True
            )
        self.loss_fn = nn.SmoothL1Loss()

        self.replay_buffer = ReplayBuffer(self.buffer_size, a_cfg.obs_type)
        self.steps_done = 0
        self.current_epsilon = self.exploration_initial_eps
        self.last_optimize_stats = {
            "loss": 0.0,
            "non_zero_reward_frac": 0.0,
            "terminal_frac": 0.0,
        }

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray | None = None,
        explore: bool = True,
    ) -> int:
        """Select Action.

        Selects an action using epsilon-greedy exploration with optional action
        masking for invalid actions.

        Args:
            state (np.ndarray): Current observation in the agent-specific
                format.
            action_mask (np.ndarray | None): Boolean mask of valid actions.
                ``True`` indicates valid actions. If ``None``, all actions are
                considered valid.
            explore (bool): Whether to apply epsilon-greedy exploration.

        Returns:
            int: Selected discrete action index.
        """
        # Exponential decay:
        eps = self.exploration_final_eps + (
            self.exploration_initial_eps - self.exploration_final_eps
        ) * math.exp(-1.0 * self.steps_done / self.exploration_decay)
        eps = max(self.exploration_final_eps, min(self.exploration_initial_eps, eps))
        self.current_epsilon = eps
        if explore:
            self.steps_done += 1

        if action_mask is not None:
            action_mask = np.asarray(action_mask, dtype=bool)
            if action_mask.shape != (self.num_actions,):
                raise ValueError(
                    f"action_mask must have shape ({self.num_actions},), "
                    f"got {action_mask.shape}"
                )
            if not np.any(action_mask):
                raise ValueError("action_mask contains no valid actions.")

        if explore and (np.random.random() < eps):
            # Random action from valid actions
            if action_mask is not None:
                valid = np.flatnonzero(action_mask)
                return int(np.random.choice(valid))
            return int(np.random.randint(self.num_actions))

        # Greedy action with masking
        state_t = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        q_values = self.policy_net(state_t)
        if action_mask is not None:
            mask_t = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            # Use masked_fill (non-in-place) to avoid modifying the compiled
            # model's output buffer, which can cause issues with torch.compile.
            q_values = q_values.masked_fill(~mask_t, float("-inf"))
        return int(q_values.argmax(dim=1).item())

    @torch.no_grad()
    def _td_target(
        self, 
        reward: torch.Tensor, 
        next_states: torch.Tensor, # Renamed to avoid variable name collision
        done: torch.Tensor,
        next_action_masks: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute TD Target using a fully masked Double DQN mechanism."""
        # 1. Get Q-values for next states from the ONLINE policy network
        next_state_policy_q = self.policy_net(next_states)
        
        # 2. MASK the online Q-values so it only picks valid actions
        if next_action_masks is not None:
            next_state_policy_q = next_state_policy_q.masked_fill(~next_action_masks, float("-inf"))
            
        # 3. Online network decides WHICH action is best
        best_actions = torch.argmax(next_state_policy_q, dim=1)
        
        # 4. TARGET network evaluates the value of that chosen action
        next_state_target_q = self.target_net(next_states)
        
        # Pure PyTorch device-safe gathering
        batch_indices = torch.arange(self.batch_size, device=self.device)
        next_values = next_state_target_q[batch_indices, best_actions]
        
        # 5. Calculate standard Bellman equation target
        return (next_values * self.gamma * (~done).float() + reward).float()

    
    def _td_estimate(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute TD Estimate.

        Computes the TD estimate for a batch of transitions.

        Args:
            state (torch.Tensor): Tensor of shape ``(batch_size, *obs_shape)``
            containing batch of states.
            action (torch.Tensor): Tensor of shape ``(batch_size,)`` containing
            action indices.
        Returns:
            torch.Tensor: TD estimate values with shape ``(batch_size,)``.
        """
        state_action_values = self.policy_net(state)[
            np.arange(0, self.batch_size), action
        ]
        return state_action_values.float()

    def optimize(self) -> float:
        """Optimize Policy Network.

        Samples a minibatch from replay memory and performs one DQN gradient
        update step using the Huber loss.

        Args:
            None: This method uses internal agent state and replay buffer.

        Returns:
            float: Scalar loss value for the update step, or ``0.0`` when there
            are not enough samples to train.
        """
        if len(self.replay_buffer) < self.batch_size:
            self.last_optimize_stats = {
                "loss": 0.0,
                "non_zero_reward_frac": 0.0,
                "terminal_frac": 0.0,
            }
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size, self.device)
        states = batch["states"]
        actions = batch["actions"].squeeze()
        rewards = batch["rewards"].squeeze()
        next_states = batch["next_states"]
        dones = batch["dones"].squeeze()
        next_action_masks = batch["next_action_masks"] 

        # Current-state action values from both networks.
        batch_indices = torch.arange(self.batch_size, device=self.device)
        q_policy_current = self.policy_net(states)[batch_indices, actions].float()
        q_target_current = self.target_net(states)[batch_indices, actions].float()

        if self.target_tracking_type == "sgt2":
            # SGT2: each network has its own Bellman target, while the two
            # Q-functions are coupled symmetrically by beta.
            #
            # The Bellman targets are detached: gradients are taken only
            # through the current-state Q estimates.
            with torch.no_grad():
                next_policy_q = self.policy_net(next_states)
                next_target_q = self.target_net(next_states)

                if next_action_masks is not None:
                    next_policy_q = next_policy_q.masked_fill(
                        ~next_action_masks, float("-inf")
                    )
                    next_target_q = next_target_q.masked_fill(
                        ~next_action_masks, float("-inf")
                    )

                next_max_policy = next_policy_q.max(dim=1).values
                next_max_target = next_target_q.max(dim=1).values

                y_policy = (
                    rewards
                    + self.gamma * next_max_target * (~dones).float()
                ).float()
                y_target = (
                    rewards
                    + self.gamma * next_max_policy * (~dones).float()
                ).float()

            # Symmetric Gradient Target Tracking (SGT2).
            loss_policy = 0.5 * torch.mean(
                (y_policy - q_policy_current) ** 2
                + self.beta * (q_target_current - q_policy_current) ** 2
            )
            loss_target = 0.5 * torch.mean(
                (y_target - q_target_current) ** 2
                + self.beta * (q_policy_current - q_target_current) ** 2
            )
            loss = loss_policy + loss_target

        else:
            # Standard Double DQN: online policy selects the action and the
            # frozen target network evaluates it.
            next_state_values = self._td_target(
                rewards, next_states, dones, next_action_masks
            )
            loss = self.loss_fn(q_policy_current, next_state_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip all trainable network gradients.
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.max_grad_norm
        )
        if self.target_tracking_type == "sgt2":
            torch.nn.utils.clip_grad_norm_(
                self.target_net.parameters(), self.max_grad_norm
            )

        self.optimizer.step()

        # Optional soft target tracking. This is deliberately performed after
        # the gradient step and without building an autograd graph.
        if self.target_update_method == "polyak":
            self._polyak_update()

        self.last_optimize_stats = {
            "loss": float(loss.item()),
            "non_zero_reward_frac": float((rewards.abs() > 0).float().mean().item()),
            "terminal_frac": float(dones.float().mean().item()),
        }

        return loss.item()

    @torch.no_grad()
    def _polyak_update(self) -> None:
        """Soft-update target parameters using Polyak averaging.

        target <- (1 - tau) * target + tau * policy
        """
        tau = self.polyak_tau
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.mul_(1.0 - tau)
            target_param.add_(policy_param, alpha=tau)

    @torch.no_grad()
    def update_target_net(self) -> None:
        """Update the target network according to ``target_update_method``.

        ``hard`` copies the policy network every ``target_update_interval``
        calls. ``polyak`` performs the soft update after each optimization
        step; this method is retained for compatibility with training loops
        that explicitly call ``update_target_net()``.
        """
        if self.target_update_method == "polyak":
            self._polyak_update()
            return

        self._n_calls += 1
        if self._n_calls % self.target_update_interval != 0:
            return

        self.target_net.load_state_dict(self.policy_net.state_dict())

    # --- Dyad support methods ---
    @torch.no_grad()
    def compute_q_values(self, states: np.ndarray) -> torch.Tensor:
        """Compute Q Values.

        Runs a forward pass of the policy network to compute Q-values for a
        batch of observations.

        Args:
            states (np.ndarray): Batch of states in agent-specific observation
                format.

        Returns:
            torch.Tensor: Tensor of shape ``(batch_size, num_actions)``
            containing Q-values.
        """
        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        return self.policy_net(states_t)

    @torch.no_grad()
    def compute_expected_return(self, transitions: list[Transition]) -> np.ndarray:
        """Compute Expected Return.

        Computes per-transition action values ``Q(s, a)`` from the current
        policy network.

        Args:
            transitions (list[Transition]): Transition batch containing state
                and action fields.

        Returns:
            np.ndarray: Expected returns with shape ``(len(transitions),)``.
        """
        actions = np.array([t.action for t in transitions], dtype=np.int64)
        if self.obs_type == "rgb":
            # For RGB, stack along channel dimension
            states = np.array([t.state_rgb for t in transitions], dtype=np.float32)
        elif self.obs_type == "puzzle_state":
            # For puzzle_state, stack along feature dimension
            states = np.array([t.state_discrete for t in transitions], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported observation type: {self.obs_type!r}")

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(
            actions, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(1)

        return q_values.cpu().numpy()

    def add_to_buffer(self, transitions: list[Transition]) -> None:
        """Add Transitions To Replay Buffer.

        Appends externally provided transitions to the replay buffer.

        Args:
            transitions (list[Transition]): Transitions to append.

        Returns:
            None: Transitions are added to internal replay storage.
        """
        self.replay_buffer.extend(transitions)

    # --- Serialization ---

    def save(self, path: str) -> None:
        """Save Agent Checkpoint.

        Saves policy and target network weights, optimizer state, and training
        step counter to disk.

        Args:
            path (str): Destination path for the checkpoint file.

        Returns:
            None: Checkpoint is written to disk.
        """
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
                "target_tracking_type": self.target_tracking_type,
                "target_update_method": self.target_update_method,
                "polyak_tau": self.polyak_tau,
                "beta": self.beta,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load Agent Checkpoint.

        Loads policy and target network weights, optimizer state, and training
        step counter from a saved checkpoint.

        Args:
            path (str): Source path of the checkpoint file.

        Returns:
            None: Model and optimizer states are restored in place.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = int(checkpoint.get("steps_done", 0))

        saved_tracking_type = checkpoint.get("target_tracking_type")
        if (
            saved_tracking_type is not None
            and str(saved_tracking_type).lower() != self.target_tracking_type
        ):
            raise ValueError(
                "Checkpoint uses target_tracking_type="
                f"{saved_tracking_type!r}, but the current agent uses "
                f"{self.target_tracking_type!r}."
            )

        saved_update_method = checkpoint.get("target_update_method")
        if (
            saved_update_method is not None
            and str(saved_update_method).lower() != self.target_update_method
        ):
            raise ValueError(
                "Checkpoint uses target_update_method="
                f"{saved_update_method!r}, but the current agent uses "
                f"{self.target_update_method!r}."
            )
