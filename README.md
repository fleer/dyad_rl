# Dyad RL — Collaborative Reinforcement Learning on Logic Puzzles

Investigating whether **dyad learning** (two agents sharing experiences) improves reinforcement learning on [Simon Tatham's Portable Puzzle Collection](https://www.chiark.greenend.org.uk/~sgtatham/puzzles/), integrated as a Gymnasium environment.

## Research Question

> Is learning in a dyad beneficial for the learning process of a reinforcement learning agent?

Two DQN agents (MLP on flattened discrete puzzle state and/or CNN on RGB pixels) periodically share and rate each other's experiences, and the filtered transitions are merged into both replay buffers.

## Project Structure

```
dyad_rl/
├── experiment.py          # Main entry point (train / eval / sweep)
├── experiment.ipynb       # Jupyter notebook for interactive runs
├── continue_training.py   # Resume training from a checkpoint
├── visualize_episode.py   # Render a trained agent solving one episode
├── plot_results.py        # Compare *_eval.json curves across runs
├── plot_dyad_sharing.py   # Plot dyad sharing acceptance rates
├── average_runs.py        # Average metric JSONs across repeated runs
├── config/                # Hydra YAML configs
│   ├── default.yaml
│   ├── sweep_mlp.yaml, sweep_cnn.yaml
│   ├── agent/             # mlp.yaml, cnn.yaml
│   ├── env/               # netslide_{2x3,3x3}.yaml, samegame_{2x3c3s2,5x5c3s2}.yaml
│   ├── experiment/        # per-puzzle single-agent & dyad experiment presets
│   └── training/          # dqn.yaml, dyad.yaml
├── agents/                # DQN agent & network definitions (MLP / CNN)
├── training/              # Training loops (single agent & dyad)
├── evaluation/            # Evaluation & analysis utilities
├── utils/                 # Env factory, replay buffer, metrics logger
├── puzzle_env/            # C-backed puzzle environment (do not modify)
├── checkpoints/           # Saved model weights
├── results/               # Metrics JSONs & visualizations
└── smoke_test.py          # End-to-end pipeline verification
```

## Setup

**Prerequisites:** Python 3.11, CMake, a C compiler, and pygame dependencies.

```bash
# Clone and install
./install.sh
```

The install script will:
1. Verify `uv` is installed
2. Create a `.venv` virtual environment (Python 3.11)
3. Build the C puzzle libraries via CMake
4. Prompt you to choose a PyTorch backend (`cpu`, `amd`/ROCm, or `cuda`), then run `uv sync --extras <choice>` to install the `rlp` puzzle-env package, PyTorch, and all other Python dependencies

> **Note:** The PyTorch backend defaults to `cpu` if you press Enter without a choice. You can also install non-interactively by running `uv sync --extras cpu|amd|cuda` directly.

## Experiments

Experiment presets live in `config/experiment/`. Each preset selects an `env`, an `agent` (or `agent_a` + `agent_b` for dyads), and a `training` config (`dqn` or `dyad`). Puzzles currently shipped:

- `netslide` at `2x3b1` and `3x3b1`
- `samegame` at `2x3c3s2` and `5x5c3s2`

The default composition (see `config/default.yaml`) is `env=netslide_2x3`, `agent=mlp`, `training=dqn`. Add `+experiment=<preset>` to swap in a full preset.

### Single-agent DQN (baseline)

MLP over the flattened discrete puzzle state:

```bash
# netslide
python experiment.py +experiment=netslide_2x3_mlp

# samegame
python experiment.py +experiment=samegame_2x3c3s2_mlp
python experiment.py +experiment=samegame_5x5c3s2_mlp
```

CNN over rendered RGB pixels:

```bash
python experiment.py +experiment=samegame_2x3c3s2_cnn
python experiment.py +experiment=samegame_5x5c3s2_cnn
```

A masked-random control (no learning, ε=1) is available as:

```bash
python experiment.py +experiment=samegame_2x3c3s2_mlp_all_random
```

### Dyad learning

Two agents train in parallel and periodically share rated experiences. After every `share_interval` episodes, each agent rolls out an evaluation episode; the other agent rates those transitions via its own value function and passing transitions are merged into both replay buffers. Presets cover the three dyad compositions:

```bash
# MLP + MLP dyad (samegame)
python experiment.py +experiment=samegame_2x3c3s2_mlp_dyad
python experiment.py +experiment=samegame_5x5c3s2_mlp_dyad

# MLP + CNN dyad
python experiment.py +experiment=samegame_2x3c3s2_cnn_mlp_dyad
python experiment.py +experiment=netslide_2x3_cnn_mlp_dyad

# CNN + CNN dyad
python experiment.py +experiment=samegame_2x3c3s2_cnn_cnn_dyad

# Accept-all ablation (no rating filter — training.share_all=true)
python experiment.py +experiment=samegame_2x3c3s2_mlp_dyad_accept_all
```

### Overriding parts of a preset

Any field can be overridden from the command line via Hydra:

```bash
# Swap the env inside an existing preset
python experiment.py +experiment=netslide_2x3_mlp env=netslide_3x3

# Tune hyperparameters
python experiment.py +experiment=samegame_2x3c3s2_mlp agent.learning_rate=5e-4 training.total_episodes=30000
```

### Hyperparameter sweeps (Optuna)

Two Optuna sweep configs are shipped, both targeting `samegame_5x5c3s2` and searching over learning rate, gamma, batch size, buffer size, target-update interval, epsilon schedule, `train_freq`, and `gradient_steps` (80 trials by default):

```bash
python experiment.py --multirun --config-name=sweep_mlp
python experiment.py --multirun --config-name=sweep_cnn

# Point a sweep at a different environment
python experiment.py --multirun --config-name=sweep_mlp env=samegame_2x3c3s2
```

### Evaluation

Evaluate a trained checkpoint (loads `checkpoints/<exp_name>/best_model.pt`):

```bash
python experiment.py +experiment=netslide_2x3_mlp mode=eval experiment_name=<exp_name>
```

## Visualization — Interactive Episode Playback

Run a trained agent on a single puzzle instance and watch it solve (or attempt to solve) the puzzle in real-time. The `visualize_episode.py` script automates checkpoint loading, config inference, and episode execution.

### Quick Start

```bash
# Visualize a trained MLP agent
source .venv/bin/activate.fish
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt --device cpu

# Visualize a CNN agent
python visualize_episode.py --checkpoint checkpoints/exp_cnn_samegame2x3c3s2/best_model.pt --device cpu
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
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt
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
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt --device cpu
```

Use this if you encounter GPU memory issues or crashes.

#### Test Generalization to Harder Puzzles

```bash
# Train on 2x3, test on 3x3 (if the agent generalizes)
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt \
  --params "3x3b1" --device cpu
```

#### Save Episode Frames for Video Creation

```bash
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt \
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
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt \
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
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt \
  --mp4-output my_episode.mp4 --framerate 15 --device cpu
```

### Workflow Examples

**No output files:**
```bash
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt --device cpu
# Output: Console logs only, no files created
```

**Frames only (for manual video creation):**
```bash
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt --save-frames --device cpu
# Output: PNG frames saved to episode_frames/
```

**MP4 video (recommended):**
```bash
python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt --mp4-output demo.mp4 --device cpu
# Output: Frames in episode_frames/ + MP4 file demo.mp4
```

#### Run Multiple Episodes with Different Seeds

```bash
for seed in 42 123 456; do
  python visualize_episode.py --checkpoint checkpoints/exp_mlp_netslide_2x3/best_model.pt \
    --seed $seed --device cpu
done
```

### Command-Line Options

```
--checkpoint CHECKPOINT       (required) Path to checkpoint file
                              Example: checkpoints/exp_mlp_netslide_2x3/best_model.pt

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

Managed via [Hydra](https://hydra.cc/). The default composition (`config/default.yaml`) is `env=netslide_2x3`, `agent=mlp`, `training=dqn`. Key parameters:

| Group | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| `env` | `puzzle` | `netslide` | Puzzle name (`netslide`, `samegame`, …) |
| `env` | `params` | `2x3b1` | Puzzle generation parameters |
| `env` | `render_mode` | `rgb_array` | Renderer used for RGB observations |
| `env` | `window_width` / `window_height` | `64` / `64` | RGB render resolution (env-dependent) |
| `env` | `max_state_repeats` | `200` | Episode truncated after this many repeats |
| `agent` | `type` | `mlp` | `mlp` or `cnn` |
| `agent` | `obs_type` | `puzzle_state` | `puzzle_state` (MLP) or `rgb` (CNN) |
| `agent` | `net_arch` | `[128, 128]` | MLP hidden sizes |
| `agent` | `conv_channels` / `conv_kernels` / `conv_strides` / `fc_hidden` | `[32,64,64]` / `[8,4,3]` / `[4,2,1]` / `512` | CNN backbone |
| `agent` | `learning_rate` | `1e-4` | AdamW learning rate |
| `agent` | `gamma` | `0.99` | Discount factor |
| `agent` | `batch_size` | `64` | Replay batch size |
| `agent` | `buffer_size` | `1_000_000` | Replay capacity (presets often lower this to `500_000`) |
| `agent` | `learning_starts` | `100` | Env steps before optimization begins |
| `agent` | `train_freq` | `4` | Optimize every N environment steps |
| `agent` | `gradient_steps` | `1` | Gradient updates per train call |
| `agent` | `target_update_interval` | `10000` | Hard-copy target network every N steps |
| `agent` | `exploration_initial_eps` / `exploration_final_eps` | `1.0` / `0.05` | Epsilon schedule endpoints |
| `agent` | `exploration_decay` | `1_000_000` | Exponential ε decay constant (env steps) |
| `agent` | `max_grad_norm` | `10` | Gradient-norm clipping threshold |
| `training` | `total_episodes` | `60000` | Training episodes |
| `training` | `max_steps` | `1000` | Max steps per episode |
| `training` | `eval_interval` | `1000` | Episodes between evaluations |
| `training` | `eval_episodes` | `100` | Episodes per evaluation |
| `training` | `checkpoint_interval` | `1000` | Episodes between checkpoint saves |
| `training` | `log_interval` | `100` | Episodes between metric log flushes |
| `training` | `dyad` | `false` | Enables dyad training loop when true |
| `training` | `share_interval` | `1` | Episodes between dyad sharing rounds (dyad only) |
| `training` | `rating_threshold` | `0.0` | Minimum rating for a shared transition (dyad only) |
| `training` | `share_all` | `false` | Skip the rating filter and share every transition (dyad only) |

Override any parameter from the command line:

```bash
python experiment.py +experiment=netslide_2x3_mlp agent.learning_rate=5e-4 training.total_episodes=30000
```

## Metrics

Tracked during training and saved to `results/<experiment_name>/` as JSON files:

- Average return per episode
- Success rate (puzzles solved / total)
- Episode length for successful episodes
- Training loss and Q-value statistics
- Action mask usage rate

## Plotting Results

After running experiments, generate evaluation comparison plots with the standalone plotting script.

The `--experiments` argument is required and must be a list of paths to `*_eval.json` files (relative or absolute):

```bash
python plot_results.py --experiments results/exp1/run1_eval.json ./results/exp2/run2_eval.json
```

Useful options:

```bash
# Compare a selected set of eval JSON files
python plot_results.py --experiments results/exp1/run1_eval.json ./results/exp2/run2_eval.json

# Write plots to a custom output root instead of results/visualizations/
python plot_results.py --experiments results/exp1/run1_eval.json ./results/exp2/run2_eval.json --output-dir comparison_plots
```

The script groups selected files by inferred board size and compares all provided series that have valid `*_eval.json` inputs.

It generates:

- **Evaluation Win Rate comparison** (`eval_comparison.png`) — Evaluation checkpoints with uncertainty bands (SEM) for each agent
- **Evaluation Episode Length comparison** (`eval_length_comparison.png`) — Evaluation episode-length curves across checkpoints

Plots are grouped by inferred puzzle size, with one subdirectory per board size under the chosen output root. With the default settings, for example:
- `results/visualizations/2x3/` — Plots for netslide 2x3b1 comparisons
- `results/visualizations/3x3/` — Plots for netslide 3x3b1 comparisons
- `comparison_plots/2x3/` — Same output structure when `--output-dir comparison_plots` is used

The evaluation plots include shaded uncertainty bands around the win-rate curves, reflecting the standard error of the mean (SEM) computed during each evaluation run across multiple episodes.

### Plotting Dyad Sharing Rates

For dyad runs that produce a `sharing_stats.json` file, generate a dedicated sharing-rate plot with:

```bash
python plot_dyad_sharing.py results/exp_dyad_mlp_samegame2x3c3s2_averaged/sharing_stats.json
```

Or write the figure to a custom directory:

```bash
python plot_dyad_sharing.py results/exp_dyad_mlp_samegame2x3c3s2_averaged/sharing_stats.json --output-dir comparison_plots
```

The script reads per-episode dyad sharing statistics and plots:

- **Agent A acceptance rate** — `accepted_for_a / traj_b_len`
- **Agent B acceptance rate** — `accepted_for_b / traj_a_len`

The output figure, `dyad_sharing_acceptance_rates.png`, includes both the raw per-episode acceptance-rate curves and 1000-episode moving-average overlays for each agent.

### Averaging Multiple Runs

Use `average_runs.py` to average JSON metrics across multiple enumerated experiment folders such as `exp_mlp_samegame_1`, `exp_mlp_samegame_2`, and `exp_mlp_samegame_3`.

```bash
python average_runs.py results
```

The script:

- groups folders whose names end in `_<N>` by their shared base name
- loads each JSON array found in those folders
- aligns records by the first key in each JSON object, typically `episode`
- averages numeric fields across runs
- writes the aggregated outputs to a new `<base_name>_averaged/` directory

For example, averaging:

- `results/exp_dyad_mlp_samegame2x3c3s2_1/`
- `results/exp_dyad_mlp_samegame2x3c3s2_2/`
- `results/exp_dyad_mlp_samegame2x3c3s2_3/`

produces:

- `results/exp_dyad_mlp_samegame2x3c3s2_averaged/`

This is useful before plotting when you want one aggregated `*_eval.json` file or `sharing_stats.json` derived from several repeated runs.

## Resuming Training from a Checkpoint

`continue_training.py` picks up an existing experiment from the latest checkpoint and trains for additional episodes. It is useful for extending a run that finished too early or for fine-tuning with adjusted hyperparameters.

### Quick Start

```bash
# Resume from the latest checkpoint in a directory
python continue_training.py checkpoints/exp_mlp_netslide_2x3 --n-episodes 5000

# Resume from a specific checkpoint file
python continue_training.py checkpoints/exp_mlp_netslide_2x3/checkpoint_20000.pt --n-episodes 5000

# Specify output name and device
python continue_training.py checkpoints/exp_mlp_netslide_2x3 --n-episodes 5000 \
    --experiment-name exp_mlp_netslide_2x3_ft --device cpu

# Override any config key
python continue_training.py checkpoints/exp_mlp_netslide_2x3 --n-episodes 5000 \
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
| `results/exp_mlp_netslide_2x3/` | `results/exp_mlp_netslide_2x3_resumed/` |
| `checkpoints/exp_mlp_netslide_2x3/` | `checkpoints/exp_mlp_netslide_2x3_resumed/` |

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