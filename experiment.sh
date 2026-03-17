#!/usr/bin/env bash
# =============================================================================
# experiment.sh — Dyad RL: Full Experiment Pipeline
# =============================================================================
#
# PURPOSE
# -------
# Runs all three experiments of the Dyad RL project sequentially:
#
#   Exp 1 — Baseline MLP   : DQN agent on discrete puzzle state (MLP network)
#   Exp 2 — Baseline CNN   : DQN agent on raw RGB pixels (CNN network)
#   Exp 3 — Dyad Learning  : MLP + CNN agents sharing experience via rated
#                            cross-modality trajectory exchange
#
# Each experiment is run on both puzzle difficulty levels:
#   • 2x3b1 — 2 columns × 3 rows, 1 barrier  (smaller, faster to train)
#   • 3x3b1 — 3 columns × 3 rows, 1 barrier  (larger, harder, more compute)
#
# OUTPUTS
# -------
# For each run, the following files are written:
#
#   results/<experiment_name>/
#     <agent>_training.csv    — per-episode: return, length, success, epsilon, loss
#     <agent>_eval.json       — periodic evaluation: win_rate, avg_return, avg_length
#
#   checkpoints/<experiment_name>/
#     best_model.pt           — weights at highest eval win rate
#     final_model.pt          — weights at end of training
#     checkpoint_<N>.pt       — periodic checkpoint every 5 000 episodes
#
# DEPENDENCIES
# ------------
#   • Python ≥ 3.10 (with hydra-core, torch, gymnasium, pygame, optuna, omegaconf)
#   • Compiled puzzle C libraries under puzzle_env/rlp/lib/
#   • Run install.sh once before using this script
#
# USAGE
# -----
#   # Run all experiments with defaults:
#   bash experiment.sh
#
#   # Run only a subset (comment out unwanted blocks below)
#
#   # Override any Hydra parameter from the environment, e.g.:
#   EXTRA_ARGS="training.total_episodes=5000 seed=123" bash experiment.sh
#
# =============================================================================

set -euo pipefail  # Exit on error, unset variable, or pipe failure

# ── Runtime configuration ────────────────────────────────────────────────────
#
# PYTHON
#   Path to the Python interpreter. Defaults to whatever `python` resolves to
#   in the current environment. Override if you use a specific venv:
#     PYTHON=/home/user/venvs/dyad/bin/python bash experiment.sh
PYTHON="${PYTHON:-python}"

# EXTRA_ARGS
#   Additional Hydra overrides appended to every experiment command.
#   Use this to globally change parameters without editing the script.
#   Examples:
#     EXTRA_ARGS="seed=0"                         — change random seed
#     EXTRA_ARGS="training.total_episodes=1000"   — shorter run for testing
#     EXTRA_ARGS="device=cpu"                     — force CPU
#     EXTRA_ARGS="device=cuda"                    — force CUDA GPU
#     EXTRA_ARGS="seed=7 training.total_episodes=500 device=cpu"  — combine
EXTRA_ARGS="${EXTRA_ARGS:-}"

# SKIP_3X3
#   Set to "1" to skip the larger 3x3b1 puzzle runs (saves significant time).
#     SKIP_3X3=1 bash experiment.sh
SKIP_3X3="${SKIP_3X3:-0}"

# SKIP_SWEEPS
#   Set to "0" to also run the Optuna hyperparameter sweeps before training.
#   Sweeps run 50 Optuna trials each and can take many hours.
#     SKIP_SWEEPS=0 bash experiment.sh
SKIP_SWEEPS="${SKIP_SWEEPS:-1}"

# ── Helper functions ──────────────────────────────────────────────────────────

log() {
    # Print a timestamped section header to stderr so it stays visible even
    # when stdout is redirected to a log file.
    echo "" >&2
    echo "══════════════════════════════════════════════════════" >&2
    echo "  $(date '+%Y-%m-%d %H:%M:%S')  $*" >&2
    echo "══════════════════════════════════════════════════════" >&2
    echo "" >&2
}

run_exp() {
    # Usage: run_exp <description> <hydra_arg> [<hydra_arg> ...]
    #
    # Launches experiment.py with the provided Hydra arguments.
    # The first positional argument is a human-readable label printed to the
    # log; all remaining arguments are forwarded verbatim to experiment.py.
    #
    # The function appends EXTRA_ARGS at the end so user overrides take
    # precedence over experiment defaults.
    local desc="$1"; shift
    log "Starting: $desc"
    # shellcheck disable=SC2086
    "$PYTHON" experiment.py "$@" $EXTRA_ARGS
    log "Finished: $desc"
}

# =============================================================================
# SECTION 0 — Hyperparameter Sweeps (optional, off by default)
# =============================================================================
#
# The Optuna sweeps perform automated hyperparameter search BEFORE the main
# experiments. Each sweep runs up to 50 trials using the TPE sampler and
# maximises the best evaluation win rate over 2 000 short training episodes.
#
# After a sweep completes, inspect multirun/<date>/<time>/optimization_results.yaml
# and update config/agent/mlp.yaml, config/agent/cnn.yaml, or
# config/experiment/dyad.yaml with the best-found values before running the
# main experiments.
#
# Sweep search spaces (see config/sweep_*.yaml for full details):
#
#   sweep_mlp  — searches: learning_rate, gamma, batch_size, hidden_size,
#                           num_layers, tau, buffer_size
#
#   sweep_cnn  — same as MLP plus: fc_hidden (FC layer width after conv stack)
#
#   sweep_dyad — independent agent_a / agent_b params plus:
#                training.share_interval ∈ {10, 25, 50, 100}
#                training.rating_threshold ∈ [-1.0, 1.0]
#
# HOW TO MODIFY:
#   • Change trial count:   add `hydra.sweeper.n_trials=20`
#   • Change env target:    add `env=netslide_3x3`
#   • Run in parallel (N jobs): add `hydra.sweeper.n_jobs=4`
#   • Resume:               Optuna stores results in SQLite; set
#                           `hydra.sweeper.storage=sqlite:///sweep.db`

if [ "${SKIP_SWEEPS}" = "0" ]; then

    # ── Sweep 1: MLP baseline ─────────────────────────────────────────────────
    # Finds optimal DQN hyperparameters for the MLP agent on discrete state.
    # Results saved to multirun/...
    run_exp "Optuna sweep — MLP hyperparameters" \
        --multirun \
        --config-name=sweep_mlp

    # ── Sweep 2: CNN baseline ─────────────────────────────────────────────────
    # Same as above but for the CNN agent on RGB pixels. Includes fc_hidden in
    # the search space since the fully-connected head width matters more for
    # pixel-based observations.
    run_exp "Optuna sweep — CNN hyperparameters" \
        --multirun \
        --config-name=sweep_cnn

    # ── Sweep 3: Dyad configuration ───────────────────────────────────────────
    # Searches over independent agent_a / agent_b hyperparameters and the
    # dyad-specific sharing parameters (share_interval, rating_threshold).
    # The rating_threshold controls how "surprising" a transition must be to be
    # accepted into the partner's replay buffer (0.0 = accept anything better
    # than expected; positive values are more selective).
    run_exp "Optuna sweep — Dyad sharing hyperparameters" \
        --multirun \
        --config-name=sweep_dyad

fi

# =============================================================================
# SECTION 1 — Experiment 1: Baseline MLP (Discrete State)
# =============================================================================
#
# Agent:       DQN with MLP network  [64 → 64 → 64 → num_actions]
# Observation: Flattened discrete puzzle state (1-D float32 vector, normalised
#              to [-1, 1])
# Purpose:     Establishes performance baseline for state-based reasoning.
#              The MLP has direct access to game semantics (tile positions,
#              connectivity flags, cursor position) so it tends to learn faster
#              than the CNN on small puzzles.
#
# Key config files:
#   config/experiment/baseline_mlp.yaml   — sets agent=mlp, obs_type=puzzle_state
#   config/agent/mlp.yaml                 — network architecture + DQN params
#   config/training/dqn.yaml              — total_episodes, eval_interval, etc.
#   config/env/netslide_2x3.yaml          — puzzle="netslide", params="2x3b1"
#   config/env/netslide_3x3.yaml          — puzzle="netslide", params="3x3b1"
#
# HOW TO MODIFY THIS EXPERIMENT:
#   • Change total training length:
#       add  training.total_episodes=5000
#   • Adjust MLP hidden layer sizes:
#       add  agent.net_arch=[128,128]
#   • Change learning rate:
#       add  agent.learning_rate=5e-4
#   • Disable greedy replay-buffer pre-fill (learning_starts controls this):
#       add  agent.learning_starts=0
#   • Change exploration schedule:
#       add  agent.exploration_fraction=0.2 agent.exploration_final_eps=0.01
#   • Use a different random seed:
#       add  seed=7
#
# OUTPUT:
#   results/exp1_mlp_2x3/exp1_mlp_2x3_training.csv
#   results/exp1_mlp_2x3/exp1_mlp_2x3_eval.json
#   checkpoints/exp1_mlp_2x3/best_model.pt
#   checkpoints/exp1_mlp_2x3/final_model.pt

log "=== EXPERIMENT 1: Baseline MLP ==="

# ── 1a. Netslide 2×3  (small) ────────────────────────────────────────────────
# The 2×3 variant (2 columns, 3 rows, 1 barrier) is the simplest configuration
# and typically converges within a few thousand episodes for the MLP agent.
run_exp "Exp 1 — Baseline MLP on Netslide 2x3b1" \
    +experiment=baseline_mlp \
    experiment_name=exp1_mlp_2x3 \
    env=netslide_2x3

# ── 1b. Netslide 3×3  (medium) ───────────────────────────────────────────────
# The 3×3 variant (3 columns, 3 rows, 1 barrier) has a significantly larger
# state space and requires more training episodes to converge.
# HOW TO SKIP: set SKIP_3X3=1 before running the script.
if [ "${SKIP_3X3}" = "0" ]; then
    run_exp "Exp 1 — Baseline MLP on Netslide 3x3b1" \
        +experiment=baseline_mlp \
        experiment_name=exp1_mlp_3x3 \
        env=netslide_3x3
fi

# =============================================================================
# SECTION 2 — Experiment 2: Baseline CNN (RGB Pixels)
# =============================================================================
#
# Agent:       DQN with CNN network  (Nature DQN architecture)
#              Conv2d(3→32, 8×8, stride=4) → ReLU →
#              Conv2d(32→64, 4×4, stride=2) → ReLU →
#              Conv2d(64→64, 3×3, stride=1) → ReLU →
#              Flatten → Linear(→512) → ReLU → Linear(→num_actions)
# Observation: 128×128×3 RGB pixel array, normalised to [0, 1] (channels-first)
# Purpose:     Establishes the pixel-based learning baseline. The CNN must
#              infer all game semantics purely from raw visual appearance,
#              making it harder but more modality-agnostic than the MLP.
#
# Key config files:
#   config/experiment/baseline_cnn.yaml   — sets agent=cnn, obs_type=rgb
#   config/agent/cnn.yaml                 — conv architecture + DQN params
#   config/training/dqn.yaml              — shared training loop parameters
#
# HOW TO MODIFY THIS EXPERIMENT:
#   • Change CNN architecture (convolutional layers):
#       add  agent.conv_channels=[32,64]  agent.conv_kernels=[8,4]  agent.conv_strides=[4,2]
#   • Change the fully-connected head width:
#       add  agent.fc_hidden=256
#   • Change image resolution (must match window_width/height in env config):
#       add  env.window_width=64  env.window_height=64
#   • Use a larger replay buffer (pixels use more memory):
#       add  agent.buffer_size=50000     (default is 50 000 due to memory)
#   • Increase target network update frequency:
#       add  agent.target_update_interval=5000
#
# OUTPUT:
#   results/exp2_cnn_2x3/exp2_cnn_2x3_training.csv
#   results/exp2_cnn_2x3/exp2_cnn_2x3_eval.json
#   checkpoints/exp2_cnn_2x3/best_model.pt
#   checkpoints/exp2_cnn_2x3/final_model.pt

log "=== EXPERIMENT 2: Baseline CNN ==="

# ── 2a. Netslide 2×3  (small) ────────────────────────────────────────────────
run_exp "Exp 2 — Baseline CNN on Netslide 2x3b1" \
    +experiment=baseline_cnn \
    experiment_name=exp2_cnn_2x3 \
    env=netslide_2x3

# ── 2b. Netslide 3×3  (medium) ───────────────────────────────────────────────
if [ "${SKIP_3X3}" = "0" ]; then
    run_exp "Exp 2 — Baseline CNN on Netslide 3x3b1" \
        +experiment=baseline_cnn \
        experiment_name=exp2_cnn_3x3 \
        env=netslide_3x3
fi

# =============================================================================
# SECTION 3 — Experiment 3: Dyad Learning
# =============================================================================
#
# Agents:      Two simultaneous DQN agents — Agent A (MLP) and Agent B (CNN)
# Observation: BOTH agents use obs_type="dual" — every transition stores the
#              discrete puzzle state AND the RGB pixels, enabling cross-modality
#              experience exchange.
# Purpose:     Tests whether cross-modality experience sharing between agents
#              with complementary perceptual biases improves learning for either
#              or both agents compared to solo training (Exp 1 & 2).
#
# THE DYAD PROTOCOL (runs every share_interval episodes):
#   1. Agent A runs 1 greedy episode  → trajectory_A
#   2. Agent B runs 1 greedy episode  → trajectory_B
#   3. Agent A RATES trajectory_B:
#      - Extracts Agent A-compatible obs (discrete state) from B's trajectory
#      - Computes Q_A(s_i, a_i) for each step
#      - Computes actual discounted return G_i backward from rewards
#      - Accepts transition i if  G_i - Q_A(s_i, a_i) > rating_threshold
#      - Accepted transitions injected into Agent A's replay buffer
#   4. Agent B RATES trajectory_A  (symmetric, using RGB obs + Q_B)
#
# Key config files:
#   config/experiment/dyad.yaml     — agent_a (MLP) + agent_b (CNN) params
#   config/training/dyad.yaml       — total_episodes=100000, share_interval=50,
#                                     rating_threshold=0.0
#
# Key dyad-specific parameters:
#   training.share_interval      (default 50)
#     How frequently (in episodes) agents share experiences.
#     Lower → more frequent sharing, more influence of cross-agent data.
#     Higher → less frequent, agents develop more independently first.
#     Try: 10, 25, 50, 100
#
#   training.rating_threshold    (default 0.0)
#     Minimum "surprise advantage" (G_i - Q_rater) for a transition to be
#     accepted. 0.0 = accept anything that exceeded expectations.
#     Negative (e.g. -1.0) = accept even slightly disappointing transitions.
#     Positive (e.g. 5.0)  = only accept strongly surprising transitions.
#
#   agent_a.* / agent_b.*
#     All DQN hyperparameters can be set independently per agent.
#     Example: agent_a.learning_rate=1e-3  agent_b.learning_rate=5e-5
#
# HOW TO MODIFY THIS EXPERIMENT:
#   • Increase sharing frequency:
#       add  training.share_interval=10
#   • Make acceptance criterion stricter:
#       add  training.rating_threshold=2.0
#   • Run fewer episodes:
#       add  training.total_episodes=20000
#   • Tune only Agent B's learning rate:
#       add  agent_b.learning_rate=1e-3
#   • Disable HER replay buffer and use standard experience replay:
#       add  agent_a.replay_buffer_type=standard  agent_b.replay_buffer_type=standard
#   • Change puzzle difficulty:
#       add  env=netslide_3x3
#
# OUTPUT:
#   results/exp3_dyad_2x3/agent_a_training.csv   — Agent A (MLP) per-episode log
#   results/exp3_dyad_2x3/agent_b_training.csv   — Agent B (CNN) per-episode log
#   results/exp3_dyad_2x3/agent_a_eval.json
#   results/exp3_dyad_2x3/agent_b_eval.json
#   checkpoints/exp3_dyad_2x3/agent_a/best_model.pt
#   checkpoints/exp3_dyad_2x3/agent_b/best_model.pt

log "=== EXPERIMENT 3: Dyad Learning ==="

# ── 3a. Netslide 2×3  (small) ────────────────────────────────────────────────
run_exp "Exp 3 — Dyad Learning on Netslide 2x3b1" \
    +experiment=dyad \
    experiment_name=exp3_dyad_2x3 \
    env=netslide_2x3

# ── 3b. Netslide 3×3  (medium) ───────────────────────────────────────────────
if [ "${SKIP_3X3}" = "0" ]; then
    run_exp "Exp 3 — Dyad Learning on Netslide 3x3b1" \
        +experiment=dyad \
        experiment_name=exp3_dyad_3x3 \
        env=netslide_3x3
fi

# =============================================================================
# Done
# =============================================================================

log "All experiments completed successfully."
echo ""
echo "Results are saved under:  results/"
echo "Checkpoints are saved in: checkpoints/"
echo ""
echo "To visualise results, open experiment.ipynb and run Section 6."
