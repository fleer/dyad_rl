# Reinforcement Learning Experiment Instructions

## 1. Overall Experiment Design

### 1.1 Objectives

- **Primary Goal:** Is learning in a dyad beneficial for the learning process of a reinforcement learning agent?

The aim of the experiment is to study the influence of collaborative learning - to be more specific "dyad learning" - on the learning process of reinforcement agents.


### 1.2 Environment

- **Environment Type:** 
  - Custom puzzle environment based on **Simon Tatham's Portable Puzzle Collection**, comprising 40 diverse logic puzzles (blackbox, bridges, cube, dominosa, fifteen, filling, flip, flood, galaxies, guess, inertia, keen, lightup, loopy, magnets, map, mines, mosaic, net, netslide, palisade, pattern, pearl, pegs, range, rect, samegame, signpost, singles, sixteen, slant, solo, tents, towers, tracks, twiddle, undead, unequal, unruly, untangle).
  - Integrated with **Gymnasium** (Farama) RL framework for standardized environment interface.
  - Underlying mechanics built on **Pygame** for 2D rendering and **C backend** from original puzzle collection for puzzle logic.
  - Visual representation: deterministic, two-dimensional play area with no upper bound on episode steps.

- **State Space:** 
  - Two observation types available:
    1. **RGB Pixel Observation**: Integer matrix of shape $ (3, W, H) $ where $W, H \in \{64, 128, 256\}$ (default 128×128), values in range [0, 255]. Consistent representation across all puzzles, similar to Atari benchmark.
    2. **Discrete Internal Game State**: Puzzle-specific dict containing arrays and scalars representing puzzle logic state (e.g., grid values, cursor position, game metadata). Composition differs per puzzle but all provided in the `info["puzzle_state"]` dict.
  - State also includes:
    - Cursor position: `(int, int)` coordinates indicating selected game object
    - Game metadata: puzzle dimensions, configuration parameters
    - State hash: for tracking state repetitions and early termination detection

- **Action Space:** 
  - **Type**: Discrete (keyboard-based, fixed-size)
  - **Cardinality**: 5-14 actions depending on puzzle (most puzzles use 5-6 valid actions)
  - **Action Types**:
    1. **Cursor Movement**: Up/Down/Left/Right directional inputs to select game objects (cells, edges, regions)
    2. **Game State Changes**: Actions applied to selected object (enter digit, draw edge, toggle state, etc.)
  - **Action Masking**: Environment provides action mask indicating which actions change the current state, filtering no-op moves
  - **Note**: Action cardinality is independent of puzzle size, enabling training on small instances and testing on larger ones

- **Reward Function:**
  - **Default (Sparse) Reward**:
    $$r(t) = \begin{cases} 
    +100 & \text{if puzzle solved (status = +1)} \\
    -100 & \text{if puzzle failed/stuck (status = -1)} \\
    0 & \text{if episode ongoing (status = 0)}
    \end{cases}$$
  - **Custom Rewards**: Environment allows wrapping with Gymnasium `Wrapper` interface to implement puzzle-specific intermediate rewards:
    - Negative rewards for rule-breaking actions
    - Positive rewards for partial solution progress
    - Rewards based on puzzle state information available in `info["puzzle_state"]`
  - Encourages solving puzzles with minimum number of steps

- **Environment Dynamics:**
  - **State Transitions**: Deterministic; only occur after valid user input (keyboard action)
  - **Deterministic**: Same puzzle seed produces identical puzzle instances
  - **Solvability**: All puzzles designed to be solvable without guessing (pure algorithmic reasoning)
  - **Episode Termination**:
    1. **Natural**: When puzzle is solved or agent irreversibly fails
    2. **Early Termination** (optional): When agent revisits same state >N times (default N=200, configurable), prevents agent from cycling uselessly
    3. **Timeout**: Episodes can be truncated at max_steps (default 10,000) to bound training
  - **Scalability**: Puzzles support configurable difficulty and size parameters, enabling:
    - Curriculum learning (easy → hard progression)
    - Generalization testing (train on small, test on large sizes) 

### 1.3 Agent & Algorithm
- **Algorithm:** DQN
- **Agent Architecture:** 
    - General agent architecture: A DQN agent with a policy network that takes either the discrete internal game state or the RGB pixel observation as input and outputs Q-values for each action. The architecture can be adapted based on the input modality. Following features should be implemented:
      - Experience replay buffer to store transitions and sample mini-batches for training.
      - Target network to stabilize training by providing consistent Q-value targets.
    - Agent with _discrete internal game state_: A multi-layer perceptron
        Start with following structure:
        ```python
        nn.Linear(),
        nn.ReLU(),
        nn.Linear(),
        nn.ReLU(),
        nn.Linear()
        ```
    - Agent with _RGB Pixel Observation_: A Convolutional Neural Network
        Start with following structure:
        ```python
        nn.Conv2d(),
        nn.ReLU(),
        nn.Conv2d(),
        nn.ReLU(),
        nn.Conv2d(),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(),
        nn.ReLU(),
        nn.Linear()
        ```
- **Optimizer**: AdamW Optimizer
- **Hyperparameters:** 
    - BATCH_SIZE is the number of transitions sampled from the replay buffer
    - GAMMA is the discount factor as mentioned in the previous section
    - EPS_START is the starting value of epsilon
    - EPS_END is the final value of epsilon
    - EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
    - TAU is the update rate of the target network
    - LR is the learning rate of the `AdamW` optimizer

- **Training Configuration:**
The training configuration is defined in the `config/` directory and can be easily modified. The following parameters are the default values for the training process:
  - Episode length: Until puzzle is solved, agent fails, or max_steps (10,000) is reached.
  - Total training episodes: 200000
  - Evaluation interval: 1000 episodes

### 1.4 Experiment Metrics & Evaluation
- **Success Criteria:** [Define the primary metric used to compare dyad vs. solo agents, e.g., average return or success rate.]
- **Metrics to Track:** Track and report:
  - Average return per episode
  - Success rate (solved / total)
  - Episode length (steps) for successful episodes
  - Training loss and Q-value statistics (mean, variance)
  - Action mask usage rate (fraction of invalid actions filtered)
- **Logging Cadence:** Log every `e_log` episodes and save to CSV for later analysis.
- **Evaluation Protocol:** Every `e_eval` episodes run `n_eval` test episodes and report average return, success rate, and episode length. Store results in JSON or CSV for comparisons.
- **Baseline/Comparisons:** Baseline Experiments are described in "Executed Experiments" below. Compare dyad learning against single-agent baselines.

### 1.5 Reproducibility
- **Seeds:** Specify a global seed per run and log it in results. Suggested: `seed` for environment + `seed` for numpy + `seed` for torch.
- **Determinism:** [Specify whether deterministic torch and gym flags are enabled.]
- **Puzzle Parameters:** Log the puzzle name and generation parameters for each run.
- **Versioning:** Log git commit hash and package versions in results.

### 1.6 Dyad Data Exchange Protocol
- **Storage Location:** [e.g., results/dyad/ or logs/dyad/]
- **Format:** [e.g., NPZ, Parquet, JSONL]
- **Transition Schema:**
  - `obs_rgb`: uint8 array, shape (3, W, H)
  - `obs_state`: dict of puzzle-specific arrays/scalars
  - `action`: int
  - `reward`: float
  - `next_obs_rgb`: uint8 array, shape (3, W, H)
  - `next_obs_state`: dict of puzzle-specific arrays/scalars
  - `terminated`: bool
  - `truncated`: bool
  - `info`: dict (include `puzzle_state`, `state_hash`, `cursor_pos`)
- **Rating Rule:** [Define how agent B computes expected return vs. observed return, and the acceptance threshold.]
- **Merge Rule:** [Define how experiences are combined and filtered before adding to replay buffers.]

---

## 2. Code Architecture

### 2.1 Project Structure
```
dyad_rl/
├── experiment.py       # Script for training/evaluation
├── config/              # Configuration management
├── puzzle_env           # Environment implementation
├── agents/                # Agent implementations
├── utils/                 # Utility functions
├── training/              # Training loops and procedures
├── evaluation/            # Evaluation scripts
├── logs/                  # Experiment logs and checkpoints
├── results/               # Results, metrics, visualizations
└── experiment.ipynb       # Jupyter notebook for training/evaluation
```

### 2.2 Key Components & Responsibilities

**Environment Module:**
- The `puzzle_env` directory contains the implementation of the custom puzzle environment, including the integration with Gymnasium and Pygame for rendering and puzzle logic. It should not be modified, but can be wrapped with Gymnasium `Wrapper` for custom reward shaping.

**Agent Module:**
- The `agents` directory contains the implementation of various agent architectures, including DQN agents with different input modalities (discrete state, RGB pixels). Each agent class defines the policy network, action selection, and learning methods.
- It is important that the implementation of the training loop is independent of the agent architecture, so that different agents can be easily swapped in and out for experimentation.

**Training Module:**
- Define the DQN training loop, including:
  - Action selection (epsilon-greedy)
  - Replay buffer sampling
  - Target network updates (soft update with TAU)
  - Loss computation via Huber loss
  - Gradient clipping (if used)
- Specify where action masking is applied (before action selection or via invalid-action penalty).

**Utilities:**
- [Common functions, logging, metrics tracking]

### 2.4 Configuration Management
- **Config Format:** YAML

The configuration and hyperparameter are managed via the `hydra` package. 

---

## 3. Handling of package dependencies

- A virtual environment was already initialized in `@[DIR].venv`
- All packages are already installed 
- If a package seems to be missing, notify me but don't install it by yourself


### 3.1 Core Dependencies
- **Python Version:** 3.11

### 3.2 RL/ML Frameworks
- **Primary Framework:** PyTorch

### 3.3 Environment & Simulation
- **Environment Library:** Gymnasium
- **Additional Tools:** Puzzle Environment `puzzle_env`

### 3.4 Development & Utilities

- **Visualization**: matplotlib
- **jupyter support**: jupyter
- **hyperparameter configuration and optimization**:  
    ```
    hydra-core
    hydra-optuna-sweeper
    optuna
    optuna-dashboard
    ```

### 3.5 GPU Support
Pytorch is installed with support for AMD GPUs (rocm7.1)

---

## 4. Execution & Testing


### 4.1 Training & Evaluation
```bash
python experiment.py --config [config_file] --exp-name [experiment_name]
```
All experiments are also executable and well documented in `experiment.ipynb`

### 4.2 Execution Checklist
- [ ] Confirm `config_file` includes puzzle name + parameters (netslide 2x3b1, 3x3b1)
- [ ] Set `obs_type` to `rgb` or `puzzle_state` per experiment
- [ ] Set `action_masking` to on/off
- [ ] Set `seed`, `e_log`, `e_eval`, `n_eval`
- [ ] Confirm output paths for checkpoints, logs, and results

---

## 5. Expected Outputs

- **Checkpoints:** Stored in `checkpoints/`, named as `{agent}_{environment}_{timestamp}`
- **Logs:** Stored in `logs/`
- **Results:** Stored in `results/`, including metrics and visualizations
- **Visualizations:** Training curves, evaluation results, etc., saved in `results/visualizations/`
- **Documentation:** Update the `experiment.ipynb` notebook with training and evaluation results, including visualizations and analysis.


## 6. Executed Experiments

Following experiments should be executed on the `netslide` puzzle with the parameters 
- 2x3b1
- 3x3b1

## 6.1 Experiment 1: Baseline DQN Agent with Discrete State Input
- **Objective:** Train a DQN agent using the discrete internal game state as input to establish a baseline performance on the `netslide` puzzle.
- **Configuration:**
  - Agent Architecture: MLP
  - Hyperparameters: Optimized via Optuna
- **Expected Outcome:** Baseline performance metrics (average return, win rate) for the DQN agent on the `netslide` puzzle with discrete state input.

## 6.2 Experiment 2: DQN Agent with RGB Pixel Input
- **Objective:** Train a DQN agent using RGB pixel observations as input to evaluate the impact of high-dimensional visual input on learning performance.
- **Configuration:**
  - Agent Architecture: CNN
  - Hyperparameters: Optimized via Optuna
- **Expected Outcome:** Performance metrics for the DQN agent on the `netslide` puzzle with RGB pixel input, compared to the baseline from Experiment 1.

## 6.3 Experiment 3: Dyad Learning (Most important Experiment)
- **Objective:** Train two DQN agents simultaneously in a dyad learning setup, where they can share information or learn from each other's experiences, to assess the benefits of collaborative learning. The process of sharing experience is defined as follows:
After each `n` episodes (with `n` being a configurable parameter), each agent performs one evaluation run on the environment using its current policy. The states, actions, rewards, and next states from these evaluation runs are stored. **It is important that for each tuple both available state observations - the RGB Pixel Input AND the Discrete State Input - are stored so both agents can evaluate the experience.** 
The other agent then reads these stored experiences and rates them based on its own policy.
Therefore, the agent computes the expected return of the other agent's experience using its own value function and compares it to the actual return received by the other agent. This allows each agent to assess the quality of the other agent's experiences and learn from them.
After this evaluation and rating process, the experience from both agents is combined, and then filtered based on the ratings. The combined experience is then stored in both agents' replay buffers.

- **Configuration:**
  - Agent Architecture: MLP or CNN (based on previous experiments)
  - Hyperparameters: Optimized via Optuna
- **Expected Outcome:** Performance metrics for the dyad learning setup on the `netslide` puzzle, compared to individual agent performance.


---

## Checklist for Implementation

- [ ] Environment implemented and tested
- [ ] Agent/Algorithm implemented
- [ ] Training loop functional
- [ ] Evaluation metrics tracked
- [ ] Configuration system working
- [ ] Logging and checkpointing implemented
- [ ] Documentation complete
- [ ] Initial experiments run and results recorded
