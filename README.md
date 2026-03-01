# Dyad RL — Collaborative Reinforcement Learning on Logic Puzzles

Investigating whether **dyad learning** (two agents sharing experiences) improves reinforcement learning on [Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/), integrated as a Gymnasium environment.

## Research Question

> Is learning in a dyad beneficial for the learning process of a reinforcement learning agent?

Two DQN agents — one operating on discrete game state (MLP), the other on RGB pixels (CNN) — periodically share and rate each other's experiences, and the filtered transitions are merged into both replay buffers.

## Project Structure

```
dyad_rl/
├── experiment.py          # Main entry point (train / eval / sweep)
├── experiment.ipynb       # Jupyter notebook for interactive runs
├── config/                # Hydra YAML configs
│   ├── default.yaml
│   ├── sweep_{mlp,cnn,dyad}.yaml
│   ├── agent/             # mlp.yaml, cnn.yaml
│   ├── env/               # netslide_2x3.yaml, netslide_3x3.yaml
│   ├── experiment/        # baseline_mlp.yaml, baseline_cnn.yaml, dyad.yaml
│   └── training/          # dqn.yaml, dyad.yaml
├── agents/                # DQN agent & network definitions (MLP / CNN)
├── training/              # Training loops (single agent & dyad)
├── evaluation/            # Evaluation & analysis utilities
├── utils/                 # Env factory, replay buffer, metrics logger
├── puzzle_env/            # C-backed puzzle environment (do not modify)
├── checkpoints/           # Saved model weights
├── results/               # Metrics CSVs & visualizations
├── logs/                  # Experiment logs
└── smoke_test.py          # End-to-end pipeline verification
```

## Setup

**Prerequisites:** Python 3.11, CMake, a C compiler, and pygame dependencies.

```bash
# Clone and install
./install.sh
```

The install script will:
1. Create a `.venv` virtual environment (Python 3.11)
2. Build the C puzzle libraries via CMake
3. Install the `rlp` puzzle-env package, PyTorch (ROCm 7.1), and Python dependencies

> **Note:** PyTorch is installed with AMD ROCm support by default. For NVIDIA CUDA, replace the `--index-url` in `install.sh` with the appropriate PyTorch wheel URL.

## Experiments

All experiments target the **netslide** puzzle at two difficulty levels: `2x3b1` and `3x3b1`.

### Experiment 1 — Baseline MLP (Discrete State)

DQN with an MLP policy network operating on the flattened internal game state.

```bash
# Train on netslide 2x3
python experiment.py +experiment=baseline_mlp

# Train on netslide 3x3
python experiment.py +experiment=baseline_mlp env=netslide_3x3
```

### Experiment 2 — Baseline CNN (RGB Pixels)

DQN with a CNN policy network operating on 128×128 RGB pixel observations.

```bash
python experiment.py +experiment=baseline_cnn

python experiment.py +experiment=baseline_cnn env=netslide_3x3
```

### Experiment 3 — Dyad Learning

Two agents (MLP on discrete state + CNN on RGB) train simultaneously and periodically share rated experiences. After every `share_interval` episodes, each agent runs an evaluation episode; the other agent rates those transitions using its own value function, and filtered experiences are merged into both replay buffers.

```bash
python experiment.py +experiment=dyad

python experiment.py +experiment=dyad env=netslide_3x3
```

### Hyperparameter Sweeps (Optuna)

Each experiment has an Optuna sweep config that searches over learning rate, gamma, batch size, epsilon decay, tau, and buffer size (50 trials by default).

```bash
python experiment.py --multirun --config-name=sweep_mlp
python experiment.py --multirun --config-name=sweep_cnn
python experiment.py --multirun --config-name=sweep_dyad

# Override environment for any sweep
python experiment.py --multirun --config-name=sweep_mlp env=netslide_3x3
```

### Evaluation

Evaluate a trained checkpoint:

```bash
python experiment.py +experiment=baseline_mlp mode=eval experiment_name=<exp_name>
```

The model is loaded from `checkpoints/<exp_name>/best_model.pt`.

## Configuration

Managed via [Hydra](https://hydra.cc/). Key parameters:

| Group | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| `env` | `puzzle` | `netslide` | Puzzle name |
| `env` | `params` | `2x3b1` | Puzzle generation parameters |
| `env` | `obs_type` | `puzzle_state` | `puzzle_state` or `rgb` |
| `training` | `total_episodes` | `10000` | Training episodes |
| `training` | `max_steps` | `10000` | Max steps per episode |
| `training` | `eval_interval` | `1000` | Episodes between evaluations |
| `training` | `eval_episodes` | `100` | Episodes per evaluation |
| `training` | `lr` | `1e-4` | Learning rate (AdamW) |
| `training` | `gamma` | `0.99` | Discount factor |
| `training` | `batch_size` | `32` / `128` | Replay batch size |
| `training` | `eps_start` / `eps_end` / `eps_decay` | `0.9` / `0.05` / `100000` | Epsilon-greedy schedule |
| `training` | `tau` | `0.005` | Target network soft-update rate |
| `training` | `share_interval` | `50` | Dyad experience sharing interval |

Override any parameter from the command line:

```bash
python experiment.py +experiment=baseline_mlp training.lr=5e-4 training.total_episodes=50000
```

## Metrics

Tracked during training and saved to `results/<experiment_name>/`:

- Average return per episode
- Success rate (puzzles solved / total)
- Episode length for successful episodes
- Training loss and Q-value statistics
- Action mask usage rate

## Smoke Test

Verify the full pipeline works end-to-end:

```bash
python smoke_test.py
```