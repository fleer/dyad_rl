"""Plot dyad sharing acceptance rates from a sharing statistics JSON file.

Usage:
    python plot_dyad_sharing.py results/exp_dyad_x/sharing_stats.json
    python plot_dyad_sharing.py results/exp_dyad_x/sharing_stats.json --output-dir custom_plots
"""

from __future__ import annotations

import argparse
import os

from evaluation.analyze import plot_dyad_sharing_acceptance_rates


DEFAULT_OUTPUT_DIR = os.path.join("results", "visualizations")


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot dyad sharing acceptance rates from a sharing_stats.json file."
    )
    parser.add_argument(
        "json_file",
        help="Path to a dyad sharing statistics JSON file.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where the generated plot is written.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate a dyad sharing acceptance-rate plot for one JSON file."""
    args = _parse_args()

    json_file = args.json_file
    if not json_file.endswith(".json"):
        raise SystemExit(f"Expected a JSON file, got: {json_file}")
    if not os.path.isfile(json_file):
        raise SystemExit(f"File not found: {json_file}")

    plot_dyad_sharing_acceptance_rates(json_file, args.output_dir)
    print(f"Saved plot to: {args.output_dir}")


if __name__ == "__main__":
    main()