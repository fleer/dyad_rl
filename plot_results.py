"""Generate and save all result plots produced in experiment.ipynb.

This script reproduces the analysis plotting cells from the notebook and saves:
1) Helper-generated comparison plots from evaluation.analyze
2) Inline notebook plots as explicit image files

Usage:
    python plot_results.py
    python plot_results.py --experiments exp1_mlp_samegame_2x3c3s2 exp2_mlp_samegame_2x3c3s2
    python plot_results.py --output-dir custom_plots
    python plot_results.py --list-experiments
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List

from evaluation.analyze import (
    plot_eval_comparison,
    plot_eval_length_comparison,
    plot_episode_length_curves,
    plot_learning_curves,
    plot_training_loss_curves,
)


@dataclass(frozen=True)
class ExperimentSeries:
    """Represents one comparable training/evaluation series in an experiment."""

    experiment_name: str
    series_name: str
    board_size: str
    training_csv_path: str
    eval_json_path: str

    @property
    def plot_label(self) -> str:
        """Return a readable legend label for plotting."""
        if self.series_name == "default":
            return self.experiment_name
        return f"{self.experiment_name} / {self.series_name}"

RESULTS_ROOT = "results"
VISUALIZATIONS_DIR = "visualizations"


def _discover_experiment_directories(results_root: str) -> Dict[str, str]:
    """Return experiment directory paths indexed by experiment name."""
    experiment_directories: Dict[str, str] = {}
    if not os.path.isdir(results_root):
        return experiment_directories

    for entry_name in sorted(os.listdir(results_root)):
        entry_path = os.path.join(results_root, entry_name)
        if not os.path.isdir(entry_path):
            continue
        if entry_name == VISUALIZATIONS_DIR:
            continue
        experiment_directories[entry_name] = entry_path

    return experiment_directories


def _strip_known_suffix(file_name: str, suffix: str) -> str:
    """Remove a known suffix from a filename and return the stem."""
    if not file_name.endswith(suffix):
        return file_name
    return file_name[: -len(suffix)]


def _infer_board_size(name_candidates: Iterable[str]) -> str:
    """Infer board size token (for example 2x3) from candidate strings."""
    for candidate in name_candidates:
        match = re.search(r"(\d+x\d+)", candidate)
        if match:
            return match.group(1)
    return "unknown"


def _derive_series_name(experiment_name: str, series_key: str) -> str:
    """Create a readable series name from a training/eval stem key."""
    if series_key == experiment_name:
        return "default"

    experiment_prefix = f"{experiment_name}_"
    if series_key.startswith(experiment_prefix):
        return series_key[len(experiment_prefix) :]

    return series_key


def _discover_series_for_experiment(
    experiment_name: str,
    experiment_directory: str,
) -> List[ExperimentSeries]:
    """Discover comparable training/eval series in one experiment directory."""
    training_file_by_key: Dict[str, str] = {}
    eval_file_by_key: Dict[str, str] = {}

    for file_name in os.listdir(experiment_directory):
        full_path = os.path.join(experiment_directory, file_name)
        if not os.path.isfile(full_path):
            continue
        if file_name.endswith("_training.csv"):
            series_key = _strip_known_suffix(file_name, "_training.csv")
            training_file_by_key[series_key] = full_path
        elif file_name.endswith("_eval.json"):
            series_key = _strip_known_suffix(file_name, "_eval.json")
            eval_file_by_key[series_key] = full_path

    matching_series_keys = sorted(set(training_file_by_key) & set(eval_file_by_key))
    discovered_series: List[ExperimentSeries] = []

    for series_key in matching_series_keys:
        training_csv_path = training_file_by_key[series_key]
        eval_json_path = eval_file_by_key[series_key]
        series_name = _derive_series_name(experiment_name, series_key)
        board_size = _infer_board_size(
            [experiment_name, series_key, os.path.basename(training_csv_path)]
        )
        discovered_series.append(
            ExperimentSeries(
                experiment_name=experiment_name,
                series_name=series_name,
                board_size=board_size,
                training_csv_path=training_csv_path,
                eval_json_path=eval_json_path,
            )
        )

    return discovered_series


def _parse_args(available_experiment_names: List[str]) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate result plots by comparing any selected result experiments."
        )
    )
    parser.add_argument(
        "-e",
        "--experiments",
        nargs="+",
        choices=available_experiment_names,
        default=available_experiment_names,
        help=(
            "Experiment names to include in comparisons. "
            "Defaults to all available experiment names."
        ),
    )
    parser.add_argument(
        "--list-experiments",
        action="store_true",
        help="Print discovered experiment names and exit.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=os.path.join(RESULTS_ROOT, VISUALIZATIONS_DIR),
        help=(
            "Root directory where generated plots are written. "
            "Board-size subdirectories are created underneath this path."
        ),
    )
    return parser.parse_args()


def _assign_unique_label(label_to_path: Dict[str, str], desired_label: str, path: str) -> str:
    """Insert a unique label key into a mapping and return the final label."""
    candidate_label = desired_label
    suffix_index = 2
    while candidate_label in label_to_path:
        candidate_label = f"{desired_label} ({suffix_index})"
        suffix_index += 1
    label_to_path[candidate_label] = path
    return candidate_label


def _build_plot_input_mappings(
    board_group_series: List[ExperimentSeries],
) -> tuple[Dict[str, str], Dict[str, str]]:
    """Convert series records into aligned label->training/eval mappings."""
    training_csv_by_label: Dict[str, str] = {}
    eval_json_by_label: Dict[str, str] = {}

    for series in board_group_series:
        unique_label = _assign_unique_label(
            training_csv_by_label,
            series.plot_label,
            series.training_csv_path,
        )
        eval_json_by_label[unique_label] = series.eval_json_path

    return training_csv_by_label, eval_json_by_label


def main() -> None:
    """Generate Result Plots.

    Generates and saves training and evaluation comparison plots for 2x3 and 3x3
    experiments when the required result files are available.

    Args:
        None: This function reads predefined result paths.

    Returns:
        None: Plots are saved to the visualizations directories.
    """
    experiment_directory_by_name = _discover_experiment_directories(RESULTS_ROOT)
    available_experiment_names = sorted(experiment_directory_by_name.keys())

    if not available_experiment_names:
        print(f"No experiment directories found under: {RESULTS_ROOT}")
        return

    args = _parse_args(available_experiment_names)

    if args.list_experiments:
        print("Discovered experiments:")
        for experiment_name in available_experiment_names:
            print(f"- {experiment_name}")
        return

    selected_experiment_names: List[str] = args.experiments
    series_grouped_by_board_size: Dict[str, List[ExperimentSeries]] = defaultdict(list)

    for experiment_name in selected_experiment_names:
        experiment_directory = experiment_directory_by_name[experiment_name]
        discovered_series = _discover_series_for_experiment(
            experiment_name=experiment_name,
            experiment_directory=experiment_directory,
        )
        for series in discovered_series:
            series_grouped_by_board_size[series.board_size].append(series)

    if not series_grouped_by_board_size:
        print("No matching *_training.csv and *_eval.json pairs were found.")
        return

    visualization_root_directory = args.output_dir
    os.makedirs(visualization_root_directory, exist_ok=True)

    for board_size, board_group_series in sorted(series_grouped_by_board_size.items()):
        board_output_directory = os.path.join(visualization_root_directory, board_size)
        os.makedirs(board_output_directory, exist_ok=True)

        training_csv_by_label, eval_json_by_label = _build_plot_input_mappings(
            board_group_series
        )

        if not training_csv_by_label or not eval_json_by_label:
            print(f"Skipped {board_size}: no valid training/eval pairs found.")
            continue

        plot_learning_curves(training_csv_by_label, board_output_directory, window=100)
        plot_episode_length_curves(
            training_csv_by_label,
            board_output_directory,
            window=100,
            filename="training_episode_length_comparison.png",
        )
        plot_training_loss_curves(
            training_csv_by_label,
            board_output_directory,
            window=100,
            filename="training_loss_comparison.png",
        )
        plot_eval_comparison(eval_json_by_label, board_output_directory)
        plot_eval_length_comparison(eval_json_by_label, board_output_directory)

        print(f"Saved plots ({board_size}) to: {board_output_directory}")
        print(f"Compared {len(training_csv_by_label)} series")


if __name__ == "__main__":
    main()
