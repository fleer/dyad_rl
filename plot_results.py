"""Generate and save all result plots produced in experiment.ipynb.

This script reproduces the analysis plotting cells from the notebook and saves:
1) Helper-generated comparison plots from evaluation.analyze
2) Inline notebook plots as explicit image files

Usage:
    python plot_results.py
    python plot_results.py --experiments results/exp1/foo_eval.json ./results/exp2/bar_eval.json
    python plot_results.py --output-dir custom_plots
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List
from pathlib import Path

from evaluation.analyze import (
    plot_eval_comparison,
    plot_eval_length_comparison,
)


RESULTS_ROOT = "results"
VISUALIZATIONS_DIR = "visualizations"


def _strip_known_suffix(file_name: str, suffix: str) -> str:
    """Remove a known suffix from a filename and return the stem."""
    if not file_name.endswith(suffix):
        return file_name
    return file_name[: -len(suffix)]



def _derive_series_name(experiment_name: str, series_key: str) -> str:
    """Create a readable series name from a training/eval stem key."""
    if series_key == experiment_name:
        return "default"

    experiment_prefix = f"{experiment_name}_"
    if series_key.startswith(experiment_prefix):
        return series_key[len(experiment_prefix) :]

    return series_key


def _build_series_from_eval_json(eval_json_path: str) -> ExperimentSeries | None:
    """Build one series record from a single *_eval.json file path."""
    eval_json_name = os.path.basename(eval_json_path)
    if not eval_json_name.endswith("_eval.json"):
        return None

    experiment_name = os.path.basename(os.path.dirname(eval_json_path))


    return ExperimentSeries(
        experiment_name=experiment_name,
        eval_json_path=eval_json_path,
    )


def _parse_args() -> argparse.Namespace:
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
        required=True,
        help=(
            "Paths to *_eval.json files to include in comparisons."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=os.path.join(RESULTS_ROOT, VISUALIZATIONS_DIR),
        help=(
            "Root directory where generated plots are written. "
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
) -> Dict[str, str]:
    """Convert series records into a label->eval JSON mapping."""
    eval_json_by_label: Dict[str, str] = {}

    for series in board_group_series:
        _assign_unique_label(
            eval_json_by_label,
            series.plot_label,
            series.eval_json_path,
        )

    return eval_json_by_label


def main() -> None:
    """Generate Result Plots.

    Generates and saves evaluation comparison plots grouped by board size.

    Args:
        None: This function reads predefined result paths.

    Returns:
        None: Plots are saved to the visualizations directories.
    """
    args = _parse_args()

    selected_eval_json_paths: List[str] = args.experiments

    visualization_root_directory = args.output_dir
    os.makedirs(visualization_root_directory, exist_ok=True)
    eval_json_dict = {}
    for eval_json_path in selected_eval_json_paths:
        if not eval_json_path.endswith("_eval.json"):
            print(f"Skipped {eval_json_path}: expected a *_eval.json file")
            continue
        if not os.path.isfile(eval_json_path):
            print(f"Skipped {eval_json_path}: file not found")
            continue
        eval_json_dict[Path(eval_json_path).stem] = eval_json_path


    plot_eval_comparison(eval_json_dict, visualization_root_directory)
    plot_eval_length_comparison(eval_json_dict, visualization_root_directory)

    print(f"Saved plots to: {visualization_root_directory}")
    print(f"Compared {len(eval_json_dict)} series")


if __name__ == "__main__":
    main()
