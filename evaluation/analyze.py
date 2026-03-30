import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_training_csv(path: str) -> dict[str, list]:
    """Load a training CSV into a dict of column lists."""
    data: dict[str, list] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val) if key != "success" else val == "True")
    return data


def load_eval_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def smooth(values: list[float], window: int = 100) -> np.ndarray:
    """Simple moving average."""
    arr = np.array(values)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def plot_learning_curves(
    experiments: dict[str, str],
    output_dir: str,
    window: int = 100,
) -> None:
    """Plot training return and win rate for multiple experiments.

    Args:
        experiments: Mapping of experiment_name -> path to training CSV.
        output_dir: Directory to save plots.
        window: Smoothing window size.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, csv_path in experiments.items():
        data = load_training_csv(csv_path)
        episodes = data["episode"]

        # Return curve
        returns_smooth = smooth(data["total_return"], window)
        axes[0].plot(
            episodes[window - 1:], returns_smooth, label=name, alpha=0.8
        )

        # Win rate curve
        wins = [1.0 if s else 0.0 for s in data["success"]]
        wr_smooth = smooth(wins, window)
        axes[1].plot(
            episodes[window - 1:], wr_smooth, label=name, alpha=0.8
        )

    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Average Return")
    axes[0].set_title("Training Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Win Rate")
    axes[1].set_title("Training Win Rate")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "learning_curves.png"), dpi=150)
    plt.close()


def plot_eval_comparison(
    experiments: dict[str, str],
    output_dir: str,
) -> None:
    """Plot evaluation win rate over training for multiple experiments.

    Args:
        experiments: Mapping of experiment_name -> path to eval JSON.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

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

    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    ax.set_title("Evaluation Win Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eval_comparison.png"), dpi=150)
    plt.close()
