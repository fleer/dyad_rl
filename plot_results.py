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

from evaluation.analyze import (
    plot_eval_comparison,
    plot_learning_curves,
)


def _existing(paths: Dict[str, str]) -> Dict[str, str]:
    """Filter Existing Paths.

    Filters an experiment-to-path mapping to only existing files.

    Args:
        paths (Dict[str, str]): Mapping from label to filesystem path.

    Returns:
        Dict[str, str]: Mapping containing only existing file paths.
    """
    return {name: path for name, path in paths.items() if os.path.exists(path)}


def main() -> None:
    """Generate Result Plots.

    Generates and saves training and evaluation comparison plots for 2x3 and 3x3
    experiments when the required result files are available.

    Args:
        None: This function reads predefined result paths.

    Returns:
        None: Plots are saved to the visualizations directories.
    """
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
        print(f"Saved learning curves (2x3): {vis_2x3}")
    else:
        print("Skipped 2x3 learning curves: no training CSV files found.")

    if available_eval_2x3:
        plot_eval_comparison(available_eval_2x3, vis_2x3)
        print(f"Saved eval comparison (2x3): {vis_2x3}")
    else:
        print("Skipped 2x3 eval comparison: no eval JSON files found.")

    if available_train_3x3:
        plot_learning_curves(available_train_3x3, vis_3x3, window=100)
        print(f"Saved learning curves (3x3): {vis_3x3}")
    else:
        print("Skipped 3x3 learning curves: no training CSV files found.")

    if available_eval_3x3:
        plot_eval_comparison(available_eval_3x3, vis_3x3)
        print(f"Saved eval comparison (3x3): {vis_3x3}")
    else:
        print("Skipped 3x3 eval comparison: no eval JSON files found.")


if __name__ == "__main__":
    main()
