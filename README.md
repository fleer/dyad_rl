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

## Visualization — Interactive Episode Playback

Run a trained agent on a single puzzle instance and watch it solve (or attempt to solve) the puzzle in real-time. The `visualize_episode.py` script automates checkpoint loading, config inference, and episode execution.

### Quick Start

```bash
# Visualize the best MLP model agent
source .venv/bin/activate.fish
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --device cpu

# Visualize a CNN agent
python visualize_episode.py --checkpoint checkpoints/exp2_cnn_2x3/best_model.pt --device cpu
```

### Features

- **Auto-Configuration Detection**: Automatically loads the experiment's Hydra config from `outputs/` or infers defaults
- **Agent Type Inference**: Detects whether the agent is MLP (puzzle_state) or CNN (rgb) from checkpoint name
- **Observation Handling**: Properly formats observations for each agent modality (flattened discrete state vs. RGB pixels)
- **Robust Checkpoint Loading**: Auto-detects the correct number of actions from the checkpoint file
- **Reproducibility**: Set seed for deterministic episode runs
- **Frame Saving**: Optionally save frames for video creation or analysis
- **Device Flexibility**: Run on CPU, CUDA, or ROCm

### Usage Examples

#### Basic Visualization (Auto Device)

```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt
```

This will:
1. Auto-detect CUDA if available, otherwise use CPU
2. Load config from `outputs/2026-03-31/11-24-50/.hydra/config.yaml` (or infer defaults)
3. Identify the agent as MLP (discrete state observations)
4. Run one greedy episode and report results

**Sample Output:**
```
__main__ INFO Agent loaded: MLPNetwork
__main__ INFO Starting episode...
__main__ INFO Agent type: MLPNetwork
__main__ INFO Environment: netslide 2x3b1
__main__ INFO Episode finished!
__main__ INFO   Total return: 100.0
__main__ INFO   Steps: 15
__main__ INFO   Success: True
```

#### Force CPU Device (Recommended)

```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --device cpu
```

Use this if you encounter GPU memory issues or crashes.

#### Test Generalization to Harder Puzzles

```bash
# Train on 2x3, test on 3x3 (if the agent generalizes)
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt \
  --params "3x3b1" --device cpu
```

#### Save Episode Frames for Video Creation

```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt \
  --save-frames --frames-dir my_episode_frames --device cpu
```

This saves frames to `my_episode_frames/frame_0000.png`, `frame_0001.png`, etc.

You can then create a video manually (requires `ffmpeg`):

```bash
ffmpeg -framerate 10 -pattern_type glob -i 'my_episode_frames/*.png' \
  -c:v libx264 -pix_fmt yuv420p episode.mp4
```

#### Automatically Create MP4 Video (Recommended)

Just specify `--mp4-output` and the script handles the rest:

```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt \
  --mp4-output my_episode.mp4 --device cpu
```

This will:
1. Run the episode and capture frames automatically
2. Create `my_episode.mp4` using ffmpeg (requires ffmpeg installed)
3. Display progress and results in the console

**Requirements:** `ffmpeg` must be installed:
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **macOS**: `brew install ffmpeg`
- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org) or `choco install ffmpeg`

Play the video:

```bash
mpv my_episode.mp4          # Using mpv
vlc my_episode.mp4          # Using VLC
ffplay my_episode.mp4       # Using ffplay
```

Adjust video framerate if needed:

```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt \
  --mp4-output my_episode.mp4 --framerate 15 --device cpu
```

### Workflow Examples

**No output files:**
```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --device cpu
# Output: Console logs only, no files created
```

**Frames only (for manual video creation):**
```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --save-frames --device cpu
# Output: PNG frames saved to episode_frames/
```

**MP4 video (recommended):**
```bash
python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt --mp4-output demo.mp4 --device cpu
# Output: Frames in episode_frames/ + MP4 file demo.mp4
```

#### Run Multiple Episodes with Different Seeds

```bash
for seed in 42 123 456; do
  python visualize_episode.py --checkpoint checkpoints/exp1_mlp_2x3/best_model.pt \
    --seed $seed --device cpu
done
```

### Command-Line Options

```
--checkpoint CHECKPOINT       (required) Path to checkpoint file
                              Example: checkpoints/exp1_mlp_2x3/best_model.pt

--params PARAMS              (optional) Override puzzle parameters
                              Example: 3x3b1
                              Default: Uses config value (usually 2x3b1)

--seed SEED                  (optional) Random seed for reproducibility
                              Default: 42

--device DEVICE              (optional) Compute device (auto, cpu, cuda, rocm)
                              Default: auto (auto-detects CUDA)

--max-steps MAX_STEPS        (optional) Maximum steps per episode
                              Default: 1000

--save-frames                (optional) Save rendered frames as PNG images
                              Default: Disabled

--frames-dir FRAMES_DIR      (optional) Directory for saved frames
                              Default: episode_frames (only used with --save-frames)

--mp4-output MP4_OUTPUT      (optional) Output MP4 filename. If specified, automatically
                              saves frames and creates video (requires ffmpeg).
                              Automatically enables frame saving.
                              Default: None (video not created unless specified)

--framerate FRAMERATE        (optional) Video framerate in fps
                              Default: 10 (only used with --mp4-output)
```

### Understanding the Output

A successful episode run produces:

| Metric | Meaning |
|--------|---------|
| **Total return** | Sum of all step rewards. +100 = solved, -100 = failed, 0 = incomplete |
| **Steps** | Number of actions taken by the agent |
| **Success** | `True` if puzzle solved (total return > 0), `False` otherwise |

### How It Works

1. **Config Loading**: The script searches `outputs/` for a saved `.hydra/config.yaml` matching the checkpoint's experiment name. If not found, it infers reasonable defaults.

2. **Observation Shape Inference**: For MLP agents, the script uses hardcoded sizes (e.g., 34D for netslide 2x3). For CNN agents, it uses the window dimensions (e.g., 3×128×128 for RGB).

3. **Checkpoint Auto-Detection**: If num_actions doesn't match, the script parses the checkpoint to extract the correct value and rebuilds the agent.

4. **Episode Execution**: The agent runs in greedy (no exploration) mode, selecting actions with ε=0.

5. **Frame Rendering**: Uses `render_mode="rgb_array"` (pixel rendering is more stable than pygame window mode) and optionally saves frames to disk.

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

## Plotting Results

After running experiments, generate comparison plots across all three experiment variants using the standalone plotting script:

```bash
python plot_results.py
```

This script loads training CSVs and evaluation JSONs from the results directories and generates:

- **Training Return curves** (`training_return.png`) — Smoothed total reward over episodes for each experiment
- **Training Win Rate curves** (`training_win_rate.png`) — Smoothed success rate over episodes
- **Evaluation Win Rate comparison** (`eval_comparison.png`) — Evaluation checkpoints with uncertainty bands (SEM) for each agent

Plots are saved separately for each puzzle size:
- `results/visualizations/2x3/` — Plots for netslide 2x3b1
- `results/visualizations/3x3/` — Plots for netslide 3x3b1

The evaluation plots include shaded uncertainty bands around the win-rate curves, reflecting the standard error of the mean (SEM) computed during each evaluation run across multiple episodes.

## Resuming Training from a Checkpoint

`continue_training.py` picks up an existing experiment from the latest checkpoint and trains for additional episodes. It is useful for extending a run that finished too early or for fine-tuning with adjusted hyperparameters.

### Quick Start

```bash
# Resume from the latest checkpoint in a directory
python continue_training.py checkpoints/exp1_mlp_2x3 --n-episodes 5000

# Resume from a specific checkpoint file
python continue_training.py checkpoints/exp1_mlp_2x3/checkpoint_20000.pt --n-episodes 5000

# Specify output name and device
python continue_training.py checkpoints/exp1_mlp_2x3 --n-episodes 5000 \
    --experiment-name exp1_mlp_2x3_ft --device cpu

# Override any config key
python continue_training.py checkpoints/exp1_mlp_2x3 --n-episodes 5000 \
    agent.learning_rate=5e-5 training.eval_interval=500
```

### Checkpoint Selection

When a directory is passed the script selects the checkpoint automatically:

1. **Highest-numbered** `checkpoint_N.pt` (e.g. `checkpoint_40000.pt`)
2. `final_model.pt` — if no numbered checkpoints exist
3. `best_model.pt` — final fallback

### Epsilon Continuity

The epsilon schedule continues seamlessly from where training left off. The script reads `steps_done` from the checkpoint and sets the exploration horizon to `steps_done + n_new_steps`, so epsilon never resets to 1.0 mid-training.

### Output Directories

Results and new checkpoints are written to separate directories to preserve the original run:

| Original | Resumed |
|----------|---------|
| `results/exp1_mlp_2x3/` | `results/exp1_mlp_2x3_resumed/` |
| `checkpoints/exp1_mlp_2x3/` | `checkpoints/exp1_mlp_2x3_resumed/` |

Use `--experiment-name` to choose a custom output name.

### Command-Line Options

```
checkpoint            (required) Path to a .pt file or a checkpoint directory

--n-episodes N        (required) Number of additional training episodes

--device DEV          (optional) auto | cpu | cuda | rocm  (default: from config)

--experiment-name NAME (optional) Override output directory name
                       Default: <original_name>_resumed

key=value ...         (optional) Config overrides, e.g. agent.learning_rate=5e-5
```

## Smoke Test

Verify the full pipeline works end-to-end:

```bash
python smoke_test.py
```