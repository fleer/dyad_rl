#!/usr/bin/env python3
"""
average_runs.py: Average JSON experiment results across enumerated run folders.

Usage: python average_runs.py <results_folder>

Discovers groups of experiment folders (e.g. exp_mlp_game_1, exp_mlp_game_2, ...)
and averages the JSON files within each group, writing results to a new
`<base_name>_averaged/` folder.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def get_base_name(folder_name: str) -> str:
    """Strip the trailing _N run index from an experiment folder name."""
    return re.sub(r"_\d+$", "", folder_name)



def average_records(records_per_run: list[list[dict]]) -> list[dict]:
    """
    Average a collection of JSON arrays (one per run), aligning entries by the
    value of the first key in each record (e.g. `episode`).

    Only numeric fields are averaged; the index field is preserved as-is.
    Missing keys in some runs are silently skipped.
    """
    non_empty = [r for r in records_per_run if r]
    if not non_empty:
        return []

    index_key = list(non_empty[0][0].keys())[0]

    # Map index_value -> list of records across runs
    aligned: dict = defaultdict(list)
    for run_records in non_empty:
        for record in run_records:
            aligned[record[index_key]].append(record)

    result = []
    for idx_val in sorted(aligned.keys()):
        group = aligned[idx_val]
        averaged: dict = {index_key: idx_val}
        # Collect all numeric keys (from the first record in the group)
        numeric_keys = [
            k for k in group[0].keys()
            if k != index_key and isinstance(group[0][k], (int, float))
        ]
        for key in numeric_keys:
            values = [d[key] for d in group if key in d and isinstance(d[key], (int, float))]
            if values:
                averaged[key] = sum(values) / len(values)
        result.append(averaged)

    return result


def process_group(base_name: str, folders: list[Path], out_root: Path) -> None:
    """Average all JSON files found in `folders` and write to `out_root/<base_name>_averaged/`."""
    print(f"Group: {base_name!r}  ({len(folders)} run(s))")

    # Collect JSON file paths grouped by their canonical name
    json_groups: dict[str, list[Path]] = defaultdict(list)
    for folder in sorted(folders):
        for json_file in sorted(folder.glob("*.json")):
            json_groups[json_file.name].append(json_file)

    if not json_groups:
        print("  No JSON files found, skipping.")
        return

    out_folder = out_root / f"{base_name}_averaged"
    out_folder.mkdir(parents=True, exist_ok=True)

    for canonical_name, json_files in sorted(json_groups.items()):
        records_per_run = []
        for jf in json_files:
            with open(jf) as f:
                data = json.load(f)
            if isinstance(data, list):
                records_per_run.append(data)
            else:
                print(f"  Warning: {jf.name} is not a JSON array — skipped.")

        if not records_per_run:
            continue

        averaged = average_records(records_per_run)
        out_path = out_folder / canonical_name
        with open(out_path, "w") as f:
            json.dump(averaged, f, indent=2)
        print(f"  [{len(json_files)} run(s)] -> {out_path.relative_to(out_root)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Average JSON results across enumerated experiment run folders."
    )
    parser.add_argument("folder", help="Root folder containing experiment run directories.")
    args = parser.parse_args()

    root = Path(args.folder).resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")

    # Discover all experiment folders that end with _<digits>
    groups: dict[str, list[Path]] = defaultdict(list)
    for item in sorted(root.iterdir()):
        if item.is_dir() and re.fullmatch(r".+_\d+", item.name):
            base = get_base_name(item.name)
            groups[base].append(item)

    if not groups:
        print("No enumerated experiment folders found.")
        return

    for base_name, folders in sorted(groups.items()):
        process_group(base_name, folders, root)


if __name__ == "__main__":
    main()
