import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def load_training_csv(path: str) -> dict[str, list]:
    """Load Training CSV.

    Loads training metrics CSV data into a dictionary of column lists.

    Args:
        path (str): Path to a training CSV file.

    Returns:
        dict[str, list]: Parsed CSV columns as lists.
    """
    data: dict[str, list] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, val in row.items():
                data.setdefault(key, []).append(float(val) if key != "success" else val == "True")
    return data


def load_eval_json(path: str) -> list[dict]:
    """Load Evaluation JSON.

    Loads evaluation records from a JSON file.

    Args:
        path (str): Path to an evaluation JSON file.

    Returns:
        list[dict]: Evaluation record dictionaries.
    """
    with open(path) as f:
        return json.load(f)


def smooth(values: list[float], window: int = 100) -> np.ndarray:
    """Smooth Series.

    Applies a simple moving average to a sequence.

    Args:
        values (list[float]): Input values.
        window (int): Smoothing window size.

    Returns:
        np.ndarray: Smoothed values.
    """
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
    """Plot Learning Curves.

    Plots and saves smoothed training return and win-rate figures.

    Args:
        experiments (dict[str, str]): Mapping of experiment names to training
            CSV paths.
        output_dir (str): Directory to save output plots.
        window (int): Moving-average window size.

    Returns:
        None: Plot image files are written to disk.
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


def plot_training_loss_curves(
    experiments: dict[str, str],
    output_dir: str,
    window: int = 100,
    filename: str = "training_loss.png",
) -> None:
    """Plot Training Loss Curves.

    Plots and saves smoothed training loss trajectories.

    Args:
        experiments (dict[str, str]): Mapping of experiment names to training
            CSV paths.
        output_dir (str): Directory to save output plots.
        window (int): Moving-average window size.
        filename (str): Output image filename.

    Returns:
        None: Plot image file is written to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0

    for name, csv_path in experiments.items():
        data = load_training_csv(csv_path)
        episodes = np.asarray(data.get("episode", []), dtype=float)
        losses = np.asarray(data.get("loss", []), dtype=float)

        if len(episodes) == 0 or len(losses) == 0:
            continue

        if len(losses) >= window:
            loss_smooth = smooth(losses.tolist(), window)
            ax.plot(episodes[window - 1 :], loss_smooth, label=name, alpha=0.85)
        else:
            ax.plot(episodes, losses, label=f"{name} (unsmoothed)", alpha=0.65)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Training Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_episode_length_curves(
    experiments: dict[str, str],
    output_dir: str,
    window: int = 100,
    filename: str = "training_episode_length.png",
) -> None:
    """Plot Episode Length Curves.

    Plots and saves per-episode length trajectories from training CSV files.

    Args:
        experiments (dict[str, str]): Mapping of experiment names to training
            CSV paths.
        output_dir (str): Directory to save output plots.
        window (int): Moving-average window size.
        filename (str): Output image filename.

    Returns:
        None: Plot image file is written to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = 0

    for name, csv_path in experiments.items():
        data = load_training_csv(csv_path)
        episodes = np.asarray(data.get("episode", []), dtype=float)
        lengths = np.asarray(data.get("length", []), dtype=float)

        if len(episodes) == 0 or len(lengths) == 0:
            continue

        if len(lengths) >= window:
            lengths_smooth = smooth(lengths.tolist(), window)
            ax.plot(episodes[window - 1 :], lengths_smooth, label=name, alpha=0.85)
        else:
            ax.plot(episodes, lengths, label=f"{name} (unsmoothed)", alpha=0.65)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Length")
    ax.set_title("Training Episode Length")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def plot_eval_comparison(
    experiments: dict[str, str],
    output_dir: str,
) -> None:
    """Plot Evaluation Comparison.

    Plots and saves evaluation win-rate trajectories for multiple experiments.

    Args:
        experiments (dict[str, str]): Mapping of experiment names to evaluation
            JSON paths.
        output_dir (str): Directory to save output plot.

    Returns:
        None: Plot image file is written to disk.
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


def plot_eval_length_comparison(
    experiments: dict[str, str],
    output_dir: str,
    filename: str = "eval_length_comparison.png",
) -> None:
    """Plot Evaluation Length Comparison.

    Plots and saves evaluation average episode-length trajectories for multiple
    experiments.

    Args:
        experiments (dict[str, str]): Mapping of experiment names to evaluation
            JSON paths.
        output_dir (str): Directory to save output plot.
        filename (str): Output image filename.

    Returns:
        None: Plot image file is written to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, json_path in experiments.items():
        evals = load_eval_json(json_path)
        episodes = [e["episode"] for e in evals]
        avg_lengths = [float(e["avg_length"]) for e in evals]
        sem_lengths = [float(e.get("sem_length", 0.0)) for e in evals]
        line = ax.plot(episodes, avg_lengths, marker="o", label=name, alpha=0.85)[0]
        color = line.get_color()
        lower = np.clip(np.array(avg_lengths) - np.array(sem_lengths), 0.0, None)
        upper = np.array(avg_lengths) + np.array(sem_lengths)
        ax.fill_between(episodes, lower, upper, color=color, alpha=0.18)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Average Episode Length")
    ax.set_title("Evaluation Episode Length")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_dyad_sharing_acceptance_rates(
    json_path: str,
    output_dir: str,
    filename: str = "dyad_sharing_acceptance_rates.png",
    window: int = 1000,
) -> None:
    """Plot dyad sharing acceptance rates against episode.

    Computes and plots two per-episode acceptance-rate curves from a dyad
    sharing statistics JSON file:
    - Agent A acceptance rate: accepted_for_a / traj_b_len
    - Agent B acceptance rate: accepted_for_b / traj_a_len

    Args:
        json_path (str): Path to a dyad sharing statistics JSON file.
        output_dir (str): Directory to save output plot.
        filename (str): Output image filename.
        window (int): Moving-average window size for smoothed overlays.

    Returns:
        None: Plot image file is written to disk.
    """
    os.makedirs(output_dir, exist_ok=True)

    sharing_stats = load_eval_json(json_path)
    if not sharing_stats:
        return

    episodes: list[float] = []
    acceptance_rate_a: list[float] = []
    acceptance_rate_b: list[float] = []

    for record in sharing_stats:
        traj_b_len = float(record.get("traj_b_len", 0.0))
        traj_a_len = float(record.get("traj_a_len", 0.0))
        accepted_for_a = float(record.get("accepted_for_a", 0.0))
        accepted_for_b = float(record.get("accepted_for_b", 0.0))

        if traj_a_len <= 0 or traj_b_len <= 0:
            continue

        episodes.append(float(record["episode"]))
        acceptance_rate_a.append(accepted_for_a / traj_b_len)
        acceptance_rate_b.append(accepted_for_b / traj_a_len)

    if not episodes:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(episodes, acceptance_rate_a, label="Agent A", alpha=0.2)
    ax.plot(episodes, acceptance_rate_b, label="Agent B", alpha=0.2)

    if len(episodes) >= window:
        smoothed_episodes = episodes[window - 1 :]
        smoothed_acceptance_rate_a = smooth(acceptance_rate_a, window)
        smoothed_acceptance_rate_b = smooth(acceptance_rate_b, window)
        ax.plot(
            smoothed_episodes,
            smoothed_acceptance_rate_a,
            label=f"Agent A (Moving Average over {window} episodes)",
            alpha=0.95,
            linewidth=2.0,
        )
        ax.plot(
            smoothed_episodes,
            smoothed_acceptance_rate_b,
            label=f"Agent B (Moving Average over {window} episodes)",
            alpha=0.95,
            linewidth=2.0,
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Acceptance Rate")
    ax.set_title("Dyad Sharing Acceptance Rate")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)
