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
#   • Python ≥ 3.10 (with hydra-core, torch, gymnasium, pygame, omegaconf)
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

set -euo pipefail # Exit on error, unset variable, or pipe failure

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
  local desc="$1"
  shift
  log "Starting: $desc"
  # shellcheck disable=SC2086
  "$PYTHON" experiment.py "$@" $EXTRA_ARGS
  log "Finished: $desc"
}

# =============================================================================
# SECTION 1 — Experiment 1: Baseline MLP (Discrete State)
# =============================================================================

log "=== EXPERIMENT 1: Baseline MLP ==="
for i in {1..6}; do
  run_exp "Exp ${i} — Baseline MLP on Samegame 2x3c3s2" \
    +experiment=samegame_2x3c3s2_mlp \
    experiment_name="exp_mlp_samegame2x3c3s2_${i}"
done

for i in {1..6}; do
  run_exp "Exp ${i} — Baseline MLP on Netslide 2x3" \
    +experiment=samegame_2x3c3s2_mlp_all_random \
    experiment_name="exp_mlp_samegame_2x3c3s2_all_random_${i}"
done


# =============================================================================
# SECTION 2 — Experiment 2: Baseline CNN (RGB Pixels)
# =============================================================================
#
log "=== EXPERIMENT 2: Baseline CNN ==="
#
# ── 2a. Samegame 2×3  (small) ────────────────────────────────────────────────
run_exp "Exp ${i} — Baseline CNN on Samegame 2x3c3s2" \
  +experiment=samegame_2x3c3s2_cnn \
  experiment_name="exp_cnn_samegame2x3c3s2"

# =============================================================================
# SECTION 3 — Experiment 3: Dyad Learning
# =============================================================================

# log "=== EXPERIMENT 3: Dyad Learning ==="
#
# # ── 3a. Samegame 2x3c3s2 (small) ────────────────────────────────────────────────

# Accept transitions with rating above 0.0 (default behavior)
for i in {1..6}; do
  run_exp "Exp ${i} — MLP vs. MLP Dyad Learning on SameGame 2x3c3s2" \
    +experiment=samegame_2x3c3s2_mlp_dyad \
    experiment_name="exp_dyad_mlp_samegame2x3c3s2_${i}"
done

# Accept all trajectories regardless of rating (ablation to test importance of sharing threshold)
for i in {1..6}; do
  run_exp "Exp ${i} — MLP vs. MLP Dyad Learning on SameGame 2x3c3s2" \
    +experiment=samegame_2x3c3s2_mlp_dyad_accept_all \
    experiment_name="exp_dyad_mlp_accept_all_samegame2x3c3s2_${i}"
done

run_exp "Exp ${i} — Dyad Learning CNN vs. MLP on SameGame 2x3c3s2" \
  +experiment=samegame_2x3c3s2_cnn_mlp_dyad \
  experiment_name="exp_dyad_cnn_mlp_samegame2x3c3s2"

# =============================================================================
# Done
# =============================================================================

# log "All experiments completed successfully."
# echo ""
# echo "Results are saved under:  results/"
# echo "Checkpoints are saved in: checkpoints/"
# echo ""

# echo ""
# echo "Average returns for all runs (see results/ folder):"
# python average_runs.py results

# echo "Plotting learning curves for all runs (see results/ folder):"
# python plot_results.py --experiments results/exp_mlp_samegame2x3c3s2_averaged/exp_mlp_samegame2x3c3s2_eval.json results/exp_dyad_mlp_samegame2x3c3s2_averaged/agent_a_eval.json results/exp_dyad_mlp_samegame2x3c3s2_averaged/agent_b_eval.json

# python plot_dyad_sharing.py results/exp_dyad_mlp_samegame2x3c3s2_averaged/sharing_stats.json
