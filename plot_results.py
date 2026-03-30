"""Generate and save all result plots produced in experiment.ipynb.

This script reproduces the analysis plotting cells from the notebook and saves:
1) Helper-generated comparison plots from evaluation.analyze
2) Inline notebook plots as explicit image files

Usage:
    python plot_results.py
"""

from __future__ import annotations

import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from evaluation.analyze import (
    load_eval_json,
    load_training_csv,
    plot_eval_comparison,
    plot_learning_curves,
    smooth,
)


def _existing(paths: Dict[str, str]) -> Dict[str, str]:
    """Filter mapping to files that exist."""
    return {name: path for name, path in paths.items() if os.path.exists(path)}


def _save_inline_learning_curves(
    experiments: Dict[str, str],
    output_path: str,
    puzzle_label: str,
    window: int = 100,
) -> None:
    """Save the notebook-style inline learning curves figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, csv_path in experiments.items():
        data = load_training_csv(csv_path)
        episodes = data["episode"]

        if len(episodes) < window:
            continue

        ret_smooth = smooth(data["total_return"], window)
        win_smooth = smooth([1.0 if s else 0.0 for s in data["success"]], window)

        axes[0].plot(episodes[window - 1 :], ret_smooth, label=name, alpha=0.8)
        axes[1].plot(episodes[window - 1 :], win_smooth, label=name, alpha=0.8)

    axes[0].set(xlabel="Episode", ylabel="Avg Return", title=f"Training Return ({puzzle_label})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set(xlabel="Episode", ylabel="Win Rate", title=f"Training Win Rate ({puzzle_label})")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_inline_eval_winrate(
    experiments: Dict[str, str],
    output_path: str,
    puzzle_label: str,
) -> None:
    """Save the notebook-style inline eval win-rate figure."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, json_path in experiments.items():
        evals = load_eval_json(json_path)
        episodes = [e["episode"] for e in evals]
        win_rates = [e["win_rate"] for e in evals]
        sem_win_rates = [float(e.get("sem_win_rate", 0.0)) for e in evals]
        line = ax.plot(episodes, win_rates, marker="o", label=name, alpha=0.85)[0]
        color = line.get_color()
        lower = np.clip(np.array(win_rates) - np.array(sem_win_rates), 0.0, 1.0)
        upper = np.clip(np.array(win_rates) + np.array(sem_win_rates), 0.0, 1.0)
        ax.fill_between(episodes, lower, upper, color=color, alpha=0.18)

    ax.set(xlabel="Episode", ylabel="Win Rate", title=f"Eval Win Rate ({puzzle_label})")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    training_csvs_2x3 = {
        "Exp1 MLP": "results/exp1_mlp_2x3/exp1_mlp_2x3_training.csv",
        "Exp2 CNN": "results/exp2_cnn_2x3/exp2_cnn_2x3_training.csv",
        "Exp3 Dyad (Agent A / MLP)": "results/exp3_dyad_2x3/agent_a_training.csv",
        "Exp3 Dyad (Agent B / CNN)": "results/exp3_dyad_2x3/agent_b_training.csv",
    }
    eval_jsons_2x3 = {
        "Exp1 MLP": "results/exp1_mlp_2x3/exp1_mlp_2x3_eval.json",
        "Exp2 CNN": "results/exp2_cnn_2x3/exp2_cnn_2x3_eval.json",
        "Exp3 Dyad (Agent A / MLP)": "results/exp3_dyad_2x3/agent_a_eval.json",
        "Exp3 Dyad (Agent B / CNN)": "results/exp3_dyad_2x3/agent_b_eval.json",
    }

    training_csvs_3x3 = {
        "Exp1 MLP": "results/exp1_mlp_3x3/exp1_mlp_3x3_training.csv",
        "Exp2 CNN": "results/exp2_cnn_3x3/exp2_cnn_3x3_training.csv",
        "Exp3 Dyad (Agent A / MLP)": "results/exp3_dyad_3x3/agent_a_training.csv",
        "Exp3 Dyad (Agent B / CNN)": "results/exp3_dyad_3x3/agent_b_training.csv",
    }
    eval_jsons_3x3 = {
        "Exp1 MLP": "results/exp1_mlp_3x3/exp1_mlp_3x3_eval.json",
        "Exp2 CNN": "results/exp2_cnn_3x3/exp2_cnn_3x3_eval.json",
        "Exp3 Dyad (Agent A / MLP)": "results/exp3_dyad_3x3/agent_a_eval.json",
        "Exp3 Dyad (Agent B / CNN)": "results/exp3_dyad_3x3/agent_b_eval.json",
    }

    vis_2x3 = "results/visualizations/2x3"
    vis_3x3 = "results/visualizations/3x3"
    os.makedirs(vis_2x3, exist_ok=True)
    os.makedirs(vis_3x3, exist_ok=True)

    available_train_2x3 = _existing(training_csvs_2x3)
    available_eval_2x3 = _existing(eval_jsons_2x3)
    available_train_3x3 = _existing(training_csvs_3x3)
    available_eval_3x3 = _existing(eval_jsons_3x3)

    if available_train_2x3:
        plot_learning_curves(available_train_2x3, vis_2x3, window=100)
        _save_inline_learning_curves(
            available_train_2x3,
            os.path.join(vis_2x3, "learning_curves_inline.png"),
            puzzle_label="2x3b1",
            window=100,
        )
        print(f"Saved learning curves (2x3): {vis_2x3}")
    else:
        print("Skipped 2x3 learning curves: no training CSV files found.")

    if available_eval_2x3:
        plot_eval_comparison(available_eval_2x3, vis_2x3)
        _save_inline_eval_winrate(
            available_eval_2x3,
            os.path.join(vis_2x3, "eval_win_rate_inline.png"),
            puzzle_label="2x3b1",
        )
        print(f"Saved eval comparison (2x3): {vis_2x3}")
    else:
        print("Skipped 2x3 eval comparison: no eval JSON files found.")

    if available_train_3x3:
        plot_learning_curves(available_train_3x3, vis_3x3, window=100)
        _save_inline_learning_curves(
            available_train_3x3,
            os.path.join(vis_3x3, "learning_curves_inline.png"),
            puzzle_label="3x3b1",
            window=100,
        )
        print(f"Saved learning curves (3x3): {vis_3x3}")
    else:
        print("Skipped 3x3 learning curves: no training CSV files found.")

    if available_eval_3x3:
        plot_eval_comparison(available_eval_3x3, vis_3x3)
        _save_inline_eval_winrate(
            available_eval_3x3,
            os.path.join(vis_3x3, "eval_win_rate_inline.png"),
            puzzle_label="3x3b1",
        )
        print(f"Saved eval comparison (3x3): {vis_3x3}")
    else:
        print("Skipped 3x3 eval comparison: no eval JSON files found.")


if __name__ == "__main__":
    main()
