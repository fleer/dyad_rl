# Dyad RL — Comprehensive Research Document

## Table of Contents

1. [Research Question & Motivation](#1-research-question--motivation)
2. [Conceptual Foundation: Dyad Learning](#2-conceptual-foundation-dyad-learning)
3. [Environment: Simon Tatham's Puzzle Collection](#3-environment-simon-tathams-puzzle-collection)
4. [Agent Architecture](#4-agent-architecture)
5. [Training Infrastructure](#5-training-infrastructure)
6. [The Dyad Protocol: Cross-Modality Experience Sharing](#6-the-dyad-protocol-cross-modality-experience-sharing)
7. [Configuration & Hyperparameter Management](#7-configuration--hyperparameter-management)
8. [Evaluation & Metrics](#8-evaluation--metrics)
9. [Experimental Design](#9-experimental-design)
10. [Key Design Decisions & Rationale](#10-key-design-decisions--rationale)
11. [Technical Implementation Details](#11-technical-implementation-details)
12. [Code Architecture Summary](#12-code-architecture-summary)

---

## 1. Research Question & Motivation

**Primary Research Question:**

> *Is learning in a dyad beneficial for the learning process of a reinforcement learning agent?*

The experiment investigates whether two RL agents that periodically share and rate each other's experiences can learn more effectively than either agent training alone. This is inspired by the educational psychology concept of **dyad learning** — the observation that two learners who collaborate, compare strategies, and critique each other's approaches often outperform individual learners.

The key insight is that the two agents are not identical: they see the same puzzle through **different observation modalities** — one operates on the abstract discrete game state (MLP agent), the other on raw RGB pixel observations (CNN agent). This asymmetry creates a natural opportunity for complementary learning, analogous to two human learners who bring different cognitive strategies to the same problem.

---

## 2. Conceptual Foundation: Dyad Learning

### 2.1 What Is a "Dyad" in This Context?

A dyad consists of exactly two DQN agents:

| Agent | Observation Type | Network | Strengths |
|-------|-----------------|---------|-----------|
| **Agent A** | Discrete internal game state (flattened) | MLP (Multi-Layer Perceptron) | Direct access to game semantics; compact representation; fast learning on small puzzles |
| **Agent B** | RGB pixel observations (128×128×3) | CNN (Convolutional Neural Network) | Generalizable visual features; modality-agnostic; closer to real-world perception |

### 2.2 The Dyad Hypothesis

The experiment tests whether cross-modality experience sharing produces a synergistic effect:

- Agent A may discover efficient strategies quickly (due to compact state representation) and share those "good trajectories" with Agent B, accelerating B's learning from pixels.
- Agent B may discover strategies that A misses (due to visual patterns not encoded in the discrete state) and vice versa.
- The **rating mechanism** acts as a quality filter — only transitions that are genuinely surprising or valuable to the receiving agent are accepted.

### 2.3 Control vs. Treatment

- **Baseline (Control):** Each agent architecture trained independently (Experiments 1 & 2)
- **Treatment:** Both agents trained simultaneously with the dyad sharing protocol (Experiment 3)
- The comparison measures whether dyad-trained agents achieve higher win rates, faster convergence, or better generalization than their solo-trained counterparts.

---

## 3. Environment: Simon Tatham's Puzzle Collection

### 3.1 Overview

The environment wraps [Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/) — a set of 40 single-player logic puzzles with the following properties critical for RL research:

- **Deterministic**: Same seed → same puzzle instance, same actions → same outcomes
- **Guaranteed solvable**: Every generated puzzle has a solution reachable without guessing
- **Scalable difficulty**: Puzzle size/complexity is parameterizable (e.g., `2x3b1`, `3x3b1`)
- **Diverse reasoning**: Different puzzles require different logical strategies

The collection is implemented in C and integrated via `ctypes` FFI through the `rlp` Python package, with `pygame` as the 2D rendering backend.

### 3.2 The Target Puzzle: Netslide

All experiments in this project focus on the **Netslide** puzzle:

- **Netslide** is a sliding-tile network puzzle where the player must connect all cells in a network by sliding rows and columns of tiles.
- Two difficulty levels are used:
  - **`2x3b1`**: 2 columns × 3 rows with 1 barrier — a simpler variant suitable for initial experimentation.
  - **`3x3b1`**: 3 columns × 3 rows with 1 barrier — a harder variant for scalability testing.
- The parameter format `NxMbB` encodes width N, height M, and B barriers.

### 3.3 State Space

The environment provides two observation modalities:

#### RGB Pixel Observation
- Shape: $(3, W, H)$ where $W = H = 128$ (configurable: 64, 128, 256)
- Data type: `uint8`, values in $[0, 255]$
- Obtained by rendering the puzzle via pygame and extracting the surface pixel array
- Transposed to channels-first format (CHW) for CNN compatibility

#### Discrete Internal Game State
- A puzzle-specific dictionary of arrays and scalars extracted from the C backend's `GameState` struct
- For Netslide, this includes grid values, cursor position, network connectivity, puzzle dimensions
- Flattened to a 1-D `float32` vector via Gymnasium's `FlattenObservation` wrapper for MLP input
- The state can optionally include the cursor position (`include_cursor_in_state_info`)

#### Dual Observation Mode
- For dyad training, the environment operates in `obs_type="dual"` mode
- Returns **both** modalities simultaneously in a dict: `{"puzzle_state": ..., "pixels": ...}`
- The `puzzle_state` is pre-flattened to a 1-D float32 vector
- This ensures that every transition captures both observation types, enabling cross-modality sharing

### 3.4 Action Space

- **Type**: `Discrete(N)` where $N \in [5, 14]$ depending on puzzle
- **Actions are keyboard-based**:
  - **Cursor movement**: Up / Down / Left / Right (arrow keys)
  - **Game state modification**: Enter digit, toggle state, confirm selection, etc.
  - **Modifiers**: CTRL / SHIFT toggle (for puzzles that use modified inputs)
- **Action masking**: The environment provides `action_masks()` returning a boolean array indicating which actions would actually change the game state. This filters out no-op moves.
- **Key property**: Action cardinality is independent of puzzle size — the same action space works for `2x3` and `3x3` variants, enabling train-small/test-large generalization.

### 3.5 Reward Function

The environment uses a **sparse reward** scheme:

$$r(s, a) = \begin{cases} +100 & \text{if puzzle solved (game\_status = +1)} \\ -100 & \text{if puzzle failed/stuck (game\_status = -1)} \\ 0 & \text{otherwise (game ongoing)} \end{cases}$$

This is deliberately sparse — the agent receives no intermediate reward signal and must discover the entire solution path through exploration. The `+100` / `-100` magnitude provides a strong signal that propagates through the Q-network via the discount factor.

### 3.6 Episode Termination

Three termination conditions:

1. **Natural termination**: Puzzle solved ($\text{status} = +1$) or irreversibly failed ($\text{status} = -1$)
2. **State-repeat truncation**: If the same state is visited more than `max_state_repeats` times (default: 200), the episode is truncated — this prevents the agent from endlessly cycling
3. **Step limit**: Episodes can be bounded by `max_steps` (default: 5,000 in training config) to enforce finite episodes during training

The state-repeat mechanism works by maintaining a histogram of state hashes (`state_histogram`). Each step, the current state is hashed and its count incremented. This is a practical anti-cycling mechanism since logic puzzles with sparse rewards are highly susceptible to looping behavior.

### 3.7 Environment Registration

All 40 puzzles are registered as Gymnasium environments via `rlp/__init__.py`:
- Generic: `rlp/Puzzle-v0` (accepts `puzzle=` parameter)
- Specific: `rlp/BlackBox-v0`, `rlp/Bridges-v0`, ..., `rlp/Untangle-v0`

The project uses `rlp/Puzzle-v0` with `puzzle="netslide"` exclusively.

---

## 4. Agent Architecture

### 4.1 DQN Agent (`agents/dqn_agent.py`)

All agents in the project are instances of the `DQNAgent` class, a standard DQN implementation with the following components:

#### Core DQN Components

| Component | Implementation |
|-----------|---------------|
| **Policy network** | MLP or CNN (determined by `agent.type` config) |
| **Target network** | Identical architecture, soft-updated from policy |
| **Replay buffer** | Circular buffer with capacity `buffer_size` (default: 50,000) |
| **Optimizer** | AdamW |
| **Loss function** | Smooth L1 Loss (Huber loss) |
| **Exploration** | Epsilon-greedy with exponential decay |
| **Action masking** | Applied during both exploration and exploitation |

#### Epsilon-Greedy Schedule

The exploration rate $\epsilon$ decays exponentially:

$$\epsilon(t) = \epsilon_{\text{end}} + (\epsilon_{\text{start}} - \epsilon_{\text{end}}) \cdot e^{-t / \epsilon_{\text{decay}}}$$

With default values: $\epsilon_{\text{start}} = 0.9$, $\epsilon_{\text{end}} = 0.05$, $\epsilon_{\text{decay}} = 100{,}000$.

The `steps_done` counter is only incremented during exploration (not during evaluation).

#### Action Selection with Masking

During action selection:
1. **Random action** (with probability $\epsilon$): Choose uniformly from valid actions (where `action_mask == True`)
2. **Greedy action** (with probability $1 - \epsilon$): Compute Q-values, set invalid action Q-values to $-\infty$, then `argmax`

This ensures the agent never selects provably useless actions, dramatically reducing the effective action space.

#### Q-Value Update (Bellman Equation)

The standard DQN update:

$$Q(s, a) \leftarrow r + \gamma \cdot \max_{a'} Q_{\text{target}}(s', a') \cdot (1 - \text{done})$$

Note: Action masking is **not** applied to the target network's max computation. This simplifies the implementation but means the target includes Q-values for potentially invalid next-state actions. In practice, this is acceptable because the target network's role is to provide stable value estimates, not optimal action selection.

#### Soft Target Update

After every optimization step:

$$\theta_{\text{target}} \leftarrow \tau \cdot \theta_{\text{policy}} + (1 - \tau) \cdot \theta_{\text{target}}$$

Default $\tau = 0.1$ — this is notably higher than many DQN implementations (which use $\tau \approx 0.005$), resulting in faster target network updates. The Optuna sweeps search over $\tau \in [10^{-4}, 0.05]$.

#### Gradient Clipping

Gradient values are clipped element-wise to $[-100, 100]$ via `nn.utils.clip_grad_value_`, preventing catastrophic gradient explosions.

### 4.2 MLP Network (`agents/networks.py`)

**Purpose**: Q-value estimation from flattened discrete game state.

**Architecture**:
```
Input(flattened_state_dim) → Linear(hidden[0]) → ReLU → Linear(hidden[1]) → ReLU → Linear(num_actions)
```

- Default hidden sizes: `[64, 64]`
- Configurable via `agent.hidden_sizes`
- Input dimension computed as `np.prod(obs_shape)` from the flattened observation space
- No batch normalization, no dropout — a deliberately simple architecture

### 4.3 CNN Network (`agents/networks.py`)

**Purpose**: Q-value estimation from RGB pixel observations.

**Architecture**:
```
Conv2d(3→32, 8×8, stride=4) → ReLU →
Conv2d(32→64, 4×4, stride=2) → ReLU →
Conv2d(64→64, 3×3, stride=1) → ReLU →
Flatten →
Linear(flat_size→512) → ReLU →
Linear(512→num_actions)
```

- This is the classic "Nature DQN" architecture (Mnih et al., 2015), well-established for visual RL
- The flattened convolutional feature size is computed dynamically by passing a dummy tensor through the conv stack
- Default config: `conv_channels=[32, 64, 64]`, `conv_kernels=[8, 4, 3]`, `conv_strides=[4, 2, 1]`, `fc_hidden=512`

### 4.4 Dyad-Specific Agent Methods

The `DQNAgent` class includes three methods specifically for the dyad protocol:

1. **`compute_q_values(states)`**: Forward-pass Q-value computation for a batch of states (used internally)
2. **`compute_expected_return(transitions)`**: For a list of `Transition` namedtuples, computes $Q(s, a)$ from the policy network — this is the rater's assessment of how good each (state, action) pair is
3. **`add_to_buffer(transitions)`**: Injects external transitions (from the other agent's rated experience) into the agent's own replay buffer

---

## 5. Training Infrastructure

### 5.1 Single-Agent Training Loop (`training/train_single.py`)

The standard training loop follows the classic DQN pattern:

```
for episode in 1..total_episodes:
    1. Reset environment
    2. For each step (up to max_steps):
        a. Get action mask from environment
        b. Select action via epsilon-greedy with masking
        c. Execute action, observe (next_obs, reward, terminated, truncated)
        d. Store transition in replay buffer
        e. Sample batch & perform gradient step (optimize)
        f. Soft-update target network
    3. Log episode metrics
    4. Periodically evaluate and checkpoint
```

#### Buffer Prefilling

Before training begins, the buffer is filled with `buffer_size` random transitions:
- Random actions are selected from valid actions (using `action_masks()`)
- This ensures the agent has enough data for meaningful batch sampling from the start
- Controlled by `training.prefill_buffer` config flag (default: `true`)

#### Observation Processing

The `_process_obs()` function handles observation routing:
- **Standard mode**: Returns the raw observation (already flattened for MLP, raw pixels for CNN)
- **Dual mode**: Extracts both `puzzle_state` and `pixels` from the dict observation
- RGB observations are normalized to `[0, 1]` by dividing by 255.0

#### Checkpointing Strategy

- **Best model**: Saved whenever evaluation win rate improves (`best_model.pt`)
- **Periodic checkpoints**: Saved every `checkpoint_interval` episodes (default: 5,000)
- **Final model**: Always saved at the end of training (`final_model.pt`)
- Checkpoints include: policy network weights, target network weights, optimizer state, and `steps_done` counter

### 5.2 Dyad Training Loop (`training/train_dyad.py`)

The dyad loop is an extension of the single-agent loop with interleaved experience sharing:

```
for episode in 1..total_episodes:
    1. Agent A trains for 1 episode on env_a
    2. Agent B trains for 1 episode on env_b
    3. If episode % share_interval == 0:
        a. Agent A runs 1 greedy eval episode → trajectory_A
        b. Agent B runs 1 greedy eval episode → trajectory_B
        c. Agent A rates trajectory_B → accepted transitions for A
        d. Agent B rates trajectory_A → accepted transitions for B
        e. Inject accepted transitions into respective replay buffers
    4. Log metrics for both agents
    5. Periodically evaluate and checkpoint both agents
```

Key differences from single-agent training:
- **Two separate environments** (`env_a`, `env_b`), both in dual-obs mode
- **Two separate replay buffers** — each agent maintains its own
- **Two separate MetricsLoggers** — enabling independent analysis
- **Alternating episodes** — A and B each get one episode per iteration
- **Separate checkpointing** — `checkpoints/dyad/agent_a/` and `checkpoints/dyad/agent_b/`

### 5.3 The Replay Buffer (`utils/replay_buffer.py`)

#### Transition Schema

Each transition is stored as a `Transition` namedtuple with **9 fields**:

| Field | Type | Description |
|-------|------|-------------|
| `state` | `np.ndarray` | Agent-specific observation (used for training) |
| `action` | `int` | Action taken |
| `reward` | `float` | Reward received |
| `next_state` | `np.ndarray` | Next agent-specific observation |
| `done` | `bool` | Whether episode ended |
| `state_discrete` | `np.ndarray \| None` | Flattened puzzle state (for cross-modality sharing) |
| `next_state_discrete` | `np.ndarray \| None` | Next flattened puzzle state |
| `state_rgb` | `np.ndarray \| None` | RGB pixels (for cross-modality sharing) |
| `next_state_rgb` | `np.ndarray \| None` | Next RGB pixels |

The extra `state_discrete`/`state_rgb` fields exist specifically for the dyad protocol. In standard training, they may be `None`. In dual-obs mode, every transition captures both modalities, enabling seamless cross-modality experience transfer.

#### Buffer Mechanics

- **Circular buffer**: Fixed capacity, oldest transitions are overwritten when full
- **Uniform random sampling**: `random.sample()` for batch selection
- **Batch conversion**: Sampled transitions are converted to tensors on-device (`states`, `actions`, `rewards`, `next_states`, `dones`)
- **`extend()`**: Accepts a list of `Transition` objects — used by the dyad sharing mechanism to inject rated external experience

---

## 6. The Dyad Protocol: Cross-Modality Experience Sharing

This is the core novelty of the experiment. The dyad protocol is implemented in `_share_experience()` within `training/train_dyad.py`.

### 6.1 Protocol Overview

Every `share_interval` episodes (default: 50), the following sequence occurs:

```
1. Both agents pause training
2. Agent A runs 1 greedy (no exploration) evaluation episode → trajectory_A
3. Agent B runs 1 greedy evaluation episode → trajectory_B
4. Agent A rates trajectory_B:
   - A sees B's trajectory through A's own observation modality (discrete state)
   - A computes Q_A(s, a) for each transition
   - A compares this to the actual discounted return from the trajectory
   - Transitions where the actual return exceeded A's expectation are accepted
5. Agent B rates trajectory_A:
   - B sees A's trajectory through B's own observation modality (RGB pixels)
   - Same rating process as above, but using B's Q-network
6. Accepted transitions are injected into the respective agent's replay buffer
```

### 6.2 The Rating Mechanism (Critical Algorithm)

For a trajectory from the provider and a rater agent, the rating process works as follows:

#### Step 1: Observation Translation

The rater extracts its own compatible observations from the provider's trajectory:
- If rater is Agent A (MLP): uses `state_discrete` / `next_state_discrete` fields
- If rater is Agent B (CNN): uses `state_rgb` / `next_state_rgb` fields, normalized to $[0, 1]$

This works because dual-obs mode captures both modalities in every transition.

#### Step 2: Compute Rater's Expected Value

For each transition $i$ in the trajectory:

$$\hat{Q}_{\text{rater}}(s_i, a_i) = Q_{\text{rater}}^{\pi}(s_i, a_i)$$

This is the rater's prediction of the value of taking the provider's action in the provider's state, as evaluated by the rater's own policy network.

#### Step 3: Compute Actual Discounted Return

The **actual cumulative discounted return** from each point in the trajectory is computed backward:

$$G_i = r_i + \gamma \cdot G_{i+1}$$

with $G_{T} = r_T$ for the terminal step. This represents what actually happened in the provider's episode.

#### Step 4: Compute Rating

$$\text{rating}_i = G_i - \hat{Q}_{\text{rater}}(s_i, a_i)$$

This is the **temporal difference advantage** — the difference between what actually happened and what the rater expected.

#### Step 5: Accept/Reject

A transition is accepted if:

$$\text{rating}_i > \text{rating\_threshold}$$

Default `rating_threshold = 0.0`, meaning: **accept transitions where the provider's actual outcome was better than what the rater expected.**

This is a principled filtering mechanism:
- **High positive rating**: "The provider achieved something here that I wouldn't have expected — this is valuable to learn from"
- **Near-zero rating**: "This is about what I'd expect — nothing special to learn"
- **Negative rating**: "The provider did worse than I'd expect — not worth incorporating"

### 6.3 Transition Conversion for Cross-Modality

When transitions are accepted, they are stored using the **rater's observation format**:
- The `state` and `next_state` fields in the `Transition` are set to the rater-compatible observations
- This means Agent A (MLP) receives transitions with discrete state observations, even though they came from Agent B's episode
- And Agent B (CNN) receives transitions with RGB pixel observations from Agent A's episode
- The `action`, `reward`, and `done` fields are shared directly

### 6.4 Design Implications

| Aspect | Implication |
|--------|-------------|
| **Asymmetry** | The two agents see different things, so their ratings will differ. A may accept different transitions from B's trajectory than B would accept from A's |
| **Quality filtering** | The threshold prevents flooding the replay buffer with low-quality or redundant transitions |
| **Surprise-based learning** | By accepting only transitions that exceed expectations, the mechanism focuses on novel information |
| **Gradual adaptation** | As agents improve, their Q-value estimates become more accurate, so the bar for "surprising" transitions rises — a natural curriculum effect |
| **Buffer composition** | Over time, each agent's replay buffer becomes a mixture of self-generated and cross-agent transitions, creating a richer training distribution |

### 6.5 Sharing Metrics

The training loop logs:
- Number of accepted transitions per sharing event (vs. total trajectory length)
- Cumulative total shared transitions to each agent
- This enables analysis of how sharing frequency and acceptance rates evolve during training

---

## 7. Configuration & Hyperparameter Management

### 7.1 Hydra Framework

All configuration is managed via [Hydra](https://hydra.cc/) with a hierarchical YAML structure. The config groups are:

```
config/
├── default.yaml          # Top-level defaults (seed, device, mode)
├── agent/
│   ├── mlp.yaml          # MLP hyperparameters
│   └── cnn.yaml          # CNN hyperparameters
├── env/
│   ├── netslide_2x3.yaml
│   └── netslide_3x3.yaml
├── experiment/
│   ├── baseline_mlp.yaml  # Experiment 1 composition
│   ├── baseline_cnn.yaml  # Experiment 2 composition
│   └── dyad.yaml          # Experiment 3 composition
├── training/
│   ├── dqn.yaml           # Standard DQN training params
│   └── dyad.yaml          # Dyad training params (adds share_interval, rating_threshold)
├── sweep_mlp.yaml         # Optuna sweep for MLP baseline
├── sweep_cnn.yaml         # Optuna sweep for CNN baseline
└── sweep_dyad.yaml        # Optuna sweep for dyad
```

### 7.2 Default Hyperparameters

#### Agent (MLP)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_sizes` | `[64, 64]` | Two hidden layers, 64 units each |
| `batch_size` | `32` | Replay sampling batch size |
| `gamma` | `0.99` | Discount factor |
| `eps_start` | `0.9` | Initial exploration rate |
| `eps_end` | `0.05` | Terminal exploration rate |
| `eps_decay` | `100,000` | Exponential decay constant (in steps) |
| `tau` | `0.1` | Soft target update rate |
| `lr` | `1e-4` | AdamW learning rate |
| `buffer_size` | `50,000` | Replay buffer capacity |

#### Agent (CNN)
Same as MLP plus:
| Parameter | Value |
|-----------|-------|
| `conv_channels` | `[32, 64, 64]` |
| `conv_kernels` | `[8, 4, 3]` |
| `conv_strides` | `[4, 2, 1]` |
| `fc_hidden` | `512` |

#### Training (DQN)
| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_episodes` | `10,000` | Total training episodes |
| `max_steps` | `5,000` | Max steps per episode |
| `eval_interval` | `1,000` | Evaluate every N episodes |
| `eval_episodes` | `100` | Episodes per evaluation |
| `checkpoint_interval` | `5,000` | Save checkpoint every N episodes |
| `log_interval` | `50` | Log stats every N episodes |
| `prefill_buffer` | `true` | Pre-fill replay buffer with random transitions |
| `dyad` | `false` | Whether to use dyad training |

#### Training (Dyad) — additional parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| `share_interval` | `50` | Share experience every N episodes |
| `rating_threshold` | `0.0` | Minimum advantage for transition acceptance |
| `dyad` | `true` | Enables dyad training mode |

### 7.3 Optuna Hyperparameter Sweeps

Each experiment variant has a corresponding sweep configuration using the **TPE (Tree-structured Parzen Estimator)** sampler:

| Parameter | Search Range | Scale |
|-----------|-------------|-------|
| `lr` | $[10^{-5}, 10^{-3}]$ | Log |
| `gamma` | $[0.95, 0.999]$ | Linear |
| `batch_size` | $\{32, 64, 128, 256\}$ | Categorical |
| `eps_decay` | $\{5000, 10000, 20000, 50000\}$ | Categorical |
| `tau` | $[10^{-4}, 0.05]$ | Log |
| `buffer_size` | $\{10000, 50000, 100000\}$ | Categorical |

The CNN sweep additionally searches over `fc_hidden` $\in \{128, 256, 512\}$.

The dyad sweep searches over **independent hyperparameters for each agent** plus the dyad-specific parameters:
- `training.share_interval` $\in \{10, 25, 50, 100\}$
- `training.rating_threshold` $\in [-1.0, 1.0]$

All sweeps run 50 trials and maximize the return value from the training function (best evaluation win rate).

### 7.4 Experiment Composition

The experiment configs use Hydra's override mechanism:

- **`baseline_mlp`**: `env=netslide_2x3` + `agent=mlp` + `training=dqn` + `obs_type=puzzle_state`
- **`baseline_cnn`**: `env=netslide_2x3` + `agent=cnn` + `training=dqn` + `obs_type=rgb`
- **`dyad`**: `env=netslide_2x3` + `training=dyad` + inline `agent_a` (MLP) + inline `agent_b` (CNN)

Any parameter can be overridden from the command line, e.g.:
```bash
python experiment.py +experiment=baseline_mlp training.lr=5e-4 env=netslide_3x3
```

---

## 8. Evaluation & Metrics

### 8.1 Evaluation Protocol (`evaluation/evaluate.py`)

Evaluation runs the agent with **greedy policy** (no exploration, $\epsilon = 0$):

```python
action = agent.select_action(obs, action_mask, explore=False)
```

Each evaluation phase runs `eval_episodes` (default: 100) episodes and reports:

| Metric | Description |
|--------|-------------|
| `avg_return` | Mean total reward across evaluation episodes |
| `win_rate` | Fraction of episodes where the puzzle was solved ($\text{reward} > 0$) |
| `avg_length` | Mean episode length (steps) across all episodes |
| `avg_success_length` | Mean episode length among successful episodes only |
| `std_length` | Standard deviation of episode lengths |

### 8.2 Trajectory Collection for Dyad Sharing

`collect_eval_trajectory()` runs a single greedy episode and returns the complete trajectory as a list of dicts, each containing:
- `state`, `action`, `reward`, `next_state`, `done` (agent-specific format)
- `state_discrete`, `next_state_discrete` (always present in dual mode)
- `state_rgb`, `next_state_rgb` (always present in dual mode)

This trajectory is what gets passed to the rating function.

### 8.3 Training Metrics (`utils/metrics.py`)

The `MetricsLogger` records two types of data:

#### Per-Episode Records
Saved to `results/<experiment_name>/<agent_name>_training.csv`:
- `episode`, `total_return`, `length`, `success`, `epsilon`, `loss`

#### Evaluation Records
Saved to `results/<experiment_name>/<agent_name>_eval.json`:
- `episode`, `avg_return`, `win_rate`, `avg_length`, `avg_success_length`, `std_length`

#### Rolling Statistics
`get_recent_stats(window)` computes over the last `window` episodes:
- `avg_return`, `avg_length`, `win_rate`

Used for logging progress during training.

### 8.4 Analysis & Visualization (`evaluation/analyze.py`)

The analysis module provides:

1. **`plot_learning_curves()`**: Plots smoothed training return and win rate for multiple experiments on the same axes — enables direct visual comparison of baselines vs. dyad
2. **`plot_eval_comparison()`**: Plots evaluation win rate over training episodes — shows convergence speed differences

Both use configurable moving-average smoothing (default window: 100 episodes).

---

## 9. Experimental Design

### 9.1 Experiment 1: Baseline MLP (Discrete State)

**Purpose**: Establish the performance baseline for a DQN agent operating on the flattened discrete game state.

**Configuration**:
- Agent: MLP with `[64, 64]` hidden layers
- Observation: Flattened puzzle state (1-D float32 vector)
- Environment: Netslide `2x3b1` (and optionally `3x3b1`)
- Training: 10,000 episodes, max 5,000 steps per episode

**Command**: `python experiment.py +experiment=baseline_mlp`

### 9.2 Experiment 2: Baseline CNN (RGB Pixels)

**Purpose**: Establish the performance baseline for a DQN agent operating on raw pixel observations.

**Configuration**:
- Agent: CNN (Nature DQN architecture)
- Observation: 128×128×3 RGB pixels, normalized to [0, 1]
- Environment: Same as Experiment 1
- Training: Same parameters

**Command**: `python experiment.py +experiment=baseline_cnn`

### 9.3 Experiment 3: Dyad Learning

**Purpose**: Test whether cross-modality experience sharing improves learning for either or both agents.

**Configuration**:
- Agent A: MLP on discrete state (same as Experiment 1)
- Agent B: CNN on RGB pixels (same as Experiment 2)
- Both environments use `obs_type="dual"` to capture both modalities
- Sharing every 50 episodes
- Rating threshold: 0.0 (accept transitions that beat expectations)

**Command**: `python experiment.py +experiment=dyad`

### 9.4 Comparison Framework

The experiment is designed for **controlled comparison**:

| Variable | Baseline MLP | Baseline CNN | Dyad MLP | Dyad CNN |
|----------|-------------|-------------|----------|----------|
| Network | MLP [64,64] | CNN [32,64,64] | MLP [64,64] | CNN [32,64,64] |
| Obs type | Discrete | RGB | Discrete | RGB |
| Training episodes | 10,000 | 10,000 | 10,000 | 10,000 |
| Experience sharing | No | No | Yes (rated) | Yes (rated) |
| Buffer content | Self only | Self only | Self + filtered B | Self + filtered A |

The key comparison pairs:
- **Dyad MLP vs. Baseline MLP**: Does receiving CNN-rated experience help the MLP agent?
- **Dyad CNN vs. Baseline CNN**: Does receiving MLP-rated experience help the CNN agent?
- **Dyad (best) vs. Baseline (best)**: Does the dyad setup produce a better overall agent?

### 9.5 Reproducibility

- Global seed (default: 42) applied to numpy, torch, and environment
- All configs saved by Hydra in `outputs/` with timestamps
- Device auto-detection with fallback to CPU
- Smoke test (`smoke_test.py`) verifies the full pipeline end-to-end

---

## 10. Key Design Decisions & Rationale

### 10.1 Why DQN (Not PPO, SAC, etc.)?

DQN is chosen for its simplicity and the discrete action space of the puzzle environment. The value-based approach also naturally supports the dyad rating mechanism — Q-values provide a direct measure of "expected value" that can be compared against actual returns. Policy-gradient methods would require different sharing mechanisms (e.g., distillation).

### 10.2 Why Separate Environments for Dyad Agents?

Each agent has its own environment instance (`env_a`, `env_b`). This ensures:
- Different puzzle instances per episode (different seeds)
- No information leakage through shared environment state
- Independent episode trajectories for genuine cross-validation

### 10.3 Why Rating Instead of Direct Buffer Sharing?

Simply copying all transitions from one agent to another would be problematic:
- The transitions might be low-quality (from random exploration phases)
- The receiving agent's observation modality is different, so raw states can't be shared directly
- Unfiltered mixing could destabilize training by adding noisy/irrelevant data

The rating mechanism solves all three issues: it selects only high-quality transitions, converts them to the receiver's modality, and provides a principled acceptance criterion.

### 10.4 Why `rating_threshold = 0.0`?

A threshold of 0.0 means: "accept transitions where the provider did at least as well as I expected." This is the neutral point — the rater adds to its buffer only transitions that contain positive surprise. The Optuna sweep searches $[-1.0, 1.0]$, allowing:
- Negative threshold: Accept even slightly disappointing transitions (more permissive)
- Positive threshold: Only accept significantly surprising transitions (more selective)

### 10.5 Why Dual Observation Mode?

Rather than maintaining separate environments or post-hoc rendering, the environment captures both modalities simultaneously in every transition. This ensures:
- Exact state correspondence between discrete and pixel observations
- No need for replay or re-rendering
- Simple indexing into the trajectory for cross-modality access

### 10.6 Why Sparse Rewards?

Sparse rewards ($\{-100, 0, +100\}$) test the agents' ability to perform long-horizon credit assignment — a core challenge in RL. If intermediate rewards were provided, the learning signal would be too easy and the benefit (or lack thereof) of dyad sharing might be masked. Sparse rewards make the problem harder, amplifying any advantage from the sharing mechanism.

### 10.7 Why Netslide Specifically?

Netslide is a good testbed because:
- It has a relatively small but non-trivial action space
- The `2x3b1` configuration is solvable in a manageable number of steps
- It scales naturally to harder configurations (`3x3b1`)
- The sliding-tile mechanics create a rich state space even at small sizes
- It requires sequential planning, not just pattern matching

### 10.8 Why AdamW (Not Adam)?

AdamW applies weight decay correctly (decoupled from the gradient update), which generally provides better generalization than L2 regularization with Adam. Given the small network sizes, the effect is subtle but follows modern best practices.

---

## 11. Technical Implementation Details

### 11.1 Environment Factory (`utils/env_factory.py`)

Two factory functions create properly wrapped environments:

- **`make_env(cfg)`**: Creates a single-modality environment
  - For `puzzle_state`: wraps with `FlattenObservation` to produce a 1-D vector
  - Always wraps with `ActionMaskWrapper` for `action_masks()` accessibility
- **`make_dual_obs_env(cfg)`**: Creates a dual-modality environment
  - Forces `obs_type="dual"` regardless of config
  - Does NOT flatten (the environment returns a dict with pre-flattened `puzzle_state`)
  - Wraps with `ActionMaskWrapper`

### 11.2 Observation Processing Pipeline

The `_process_obs()` function (in both `train_single.py` and `evaluate.py`) handles the observation routing:

```
Raw env observation → _process_obs() → (agent_obs, state_discrete, state_rgb)
```

- **Standard mode**: Returns obs directly; `state_discrete` and `state_rgb` are `None`
- **Dual mode**: Extracts `obs["puzzle_state"]` and `obs["pixels"]`, selects the appropriate one for the agent, and returns all three
- RGB normalization: pixel values divided by 255.0 to get $[0, 1]$

### 11.3 C Backend Integration

The puzzle logic runs in compiled C code from Simon Tatham's collection:

1. `puzzle_env/puzzles/*.c` — original C source files (40 puzzles)
2. Compiled via CMake into shared libraries: `puzzle_env/rlp/lib/lib{puzzle}.so`
3. Python `ctypes` bindings in `puzzle_env/rlp/api.py` and `puzzle_env/rlp/specific_api.py`
4. `Puzzle` class in `puzzle_env/rlp/puzzle.py` manages the lifecycle: creation, input, drawing, state extraction
5. Drawing callbacks render to a pygame surface; state queries read C struct fields

### 11.4 State Hashing for Cycle Detection

Each step, the environment:
1. Extracts the puzzle state dict via `puzzle.get_puzzle_state()`
2. Hashes it via `rp.api.make_hash(state_dict)`
3. Maintains a histogram of hash occurrences (`state_histogram`)
4. Truncates the episode if any state has been seen more than `max_state_repeats` times

This is critical for puzzles with sparse rewards, where naive exploration tends to revisit the same states endlessly.

### 11.5 Smoke Test (`smoke_test.py`)

An end-to-end verification script that tests:
1. **Networks**: MLPNetwork and CNNNetwork forward passes with correct output shapes
2. **Replay Buffer**: Push/sample operations with expected tensor shapes
3. **Dual Obs Environment**: Dict observation with both `puzzle_state` and `pixels` keys, correct dtypes
4. **Agent Lifecycle**: Full episode with action selection, buffer storage, optimization, target update, save/load

Note: The C backend crashes if an environment is closed and recreated, so the tests avoid calling `env.close()` between test functions.

### 11.6 Hydra Output Management

Hydra automatically creates output directories in `outputs/<date>/<time>/` for each run, containing:
- The resolved config (`.hydra/config.yaml`)
- Overrides (`.hydra/overrides.yaml`)
- Logs from the run

Results and checkpoints are stored separately in `results/` and `checkpoints/` relative to the working directory.

### 11.7 Device Management

The `_resolve_device()` function in `experiment.py`:
- `"auto"`: Checks `torch.cuda.is_available()` → CUDA if available, else CPU
- Explicit options: `"cpu"`, `"cuda"`, `"rocm"`
- The install script targets AMD ROCm 7.1 by default (PyTorch nightly with ROCm support)

---

## 12. Code Architecture Summary

```
experiment.py                  # Hydra entry point: routes to train/eval/dyad
├── config/                    # YAML configuration hierarchy
│   ├── default.yaml           # Global defaults (seed, device, mode)
│   ├── agent/{mlp,cnn}.yaml   # Network architecture configs
│   ├── env/{2x3,3x3}.yaml    # Environment configs
│   ├── experiment/*.yaml      # Experiment compositions
│   ├── training/{dqn,dyad}.yaml # Training loop configs
│   └── sweep_*.yaml           # Optuna sweep configs
│
├── agents/
│   ├── dqn_agent.py           # DQNAgent: policy/target nets, optimize, select_action,
│   │                          #           dyad rating methods, save/load
│   └── networks.py            # MLPNetwork, CNNNetwork (nn.Module subclasses)
│
├── training/
│   ├── train_single.py        # Single-agent DQN loop: prefill, episode, eval, checkpoint
│   └── train_dyad.py          # Dyad loop: interleaved training + _share_experience()
│
├── evaluation/
│   ├── evaluate.py            # evaluate(): greedy eval + collect_eval_trajectory()
│   └── analyze.py             # Matplotlib plotting: learning curves, eval comparison
│
├── utils/
│   ├── env_factory.py         # make_env(), make_dual_obs_env(), ActionMaskWrapper
│   ├── replay_buffer.py       # ReplayBuffer with Transition namedtuple (9 fields)
│   └── metrics.py             # MetricsLogger, EpisodeRecord, EvalRecord
│
├── puzzle_env/                # C-backed puzzle environment (DO NOT MODIFY)
│   ├── rlp/
│   │   ├── __init__.py        # Gymnasium env registration (40 puzzles)
│   │   ├── puzzle.py          # Puzzle class: C backend interface
│   │   ├── api.py             # ctypes FFI bindings
│   │   ├── specific_api.py    # Puzzle-specific state/action helpers
│   │   ├── envs/
│   │   │   ├── puzzle_env.py  # PuzzleEnv (Gymnasium.Env): step, reset, obs, masks
│   │   │   └── observation_spaces.py  # Per-puzzle observation space definitions
│   │   └── lib/*.so           # Compiled C puzzle libraries
│   └── puzzles/*.c            # Simon Tatham's C source code
│
├── smoke_test.py              # End-to-end pipeline verification
├── checkpoints/               # Saved model weights
├── results/                   # Metrics CSVs, eval JSONs, visualizations
└── logs/                      # Experiment logs
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SINGLE AGENT TRAINING                         │
│                                                                      │
│  Environment ──obs──► _process_obs() ──► Agent.select_action()       │
│       │                                        │                     │
│       │◄──────────── action ◄──────────────────┘                     │
│       │                                                              │
│  (obs, reward, done) ──► ReplayBuffer.push()                         │
│                                │                                     │
│                         ReplayBuffer.sample() ──► Agent.optimize()   │
│                                                      │               │
│                                           Agent.update_target_net()  │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                         DYAD SHARING PROTOCOL                        │
│                                                                      │
│  Agent A (eval)──trajectory_A──► Agent B rates ──► B.add_to_buffer() │
│                                     │                                │
│                        _share_experience():                          │
│                        1. Extract B-compatible obs from A's traj     │
│                        2. Q_B(s,a) for each transition               │
│                        3. Compute actual return G_i (backward)       │
│                        4. rating_i = G_i - Q_B(s_i, a_i)            │
│                        5. Accept if rating_i > threshold             │
│                                                                      │
│  Agent B (eval)──trajectory_B──► Agent A rates ──► A.add_to_buffer() │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Summary

The dyad_rl experiment is a carefully controlled study of whether collaborative learning between two structurally different RL agents—one reasoning over abstract game state, the other over visual perception—can outperform either agent training independently. The core mechanism is a **surprise-based experience rating system**: each agent evaluates the other's demonstrations through its own value function and selectively incorporates transitions that exceeded its expectations. This creates a principled, asymmetric information exchange that respects each agent's unique perspective while enriching its training distribution with cross-modality experience.
