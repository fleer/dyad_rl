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
    """Plot training return and win rate as separate figures.

    Args:
        experiments: Mapping of experiment_name -> path to training CSV.
        output_dir: Directory to save plots.
        window: Smoothing window size.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig_return, ax_return = plt.subplots(figsize=(8, 5))
    fig_win, ax_win = plt.subplots(figsize=(8, 5))

    for name, csv_path in experiments.items():
        data = load_training_csv(csv_path)
        episodes = data["episode"]

        returns_smooth = smooth(data["total_return"], window)
        ax_return.plot(
            episodes[window - 1:], returns_smooth, label=name, alpha=0.8
        )

        wins = [1.0 if s else 0.0 for s in data["success"]]
        wr_smooth = smooth(wins, window)
        ax_win.plot(
            episodes[window - 1:], wr_smooth, label=name, alpha=0.8
        )

    ax_return.set_xlabel("Episode")
    ax_return.set_ylabel("Average Return")
    ax_return.set_title("Training Return")
    ax_return.legend()
    ax_return.grid(True, alpha=0.3)
    fig_return.tight_layout()
    fig_return.savefig(os.path.join(output_dir, "training_return.png"), dpi=150)
    plt.close(fig_return)

    ax_win.set_xlabel("Episode")
    ax_win.set_ylabel("Win Rate")
    ax_win.set_title("Training Win Rate")
    ax_win.legend()
    ax_win.grid(True, alpha=0.3)
    fig_win.tight_layout()
    fig_win.savefig(os.path.join(output_dir, "training_win_rate.png"), dpi=150)
    plt.close(fig_win)


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
