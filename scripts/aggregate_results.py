#!/usr/bin/env python3
"""Aggregate test accuracy and AUC from Hydra/TensorBoard log files."""

import argparse
import re
from pathlib import Path

import numpy as np

TEST_METRICS = re.compile(r"Test accuracy:\s*([0-9.]+),\s*Test AUC:\s*([0-9.]+)")


def analyze_log_files(output_folder: Path) -> tuple[list[float], list[float]]:
    """Return the test accuracies and AUCs found in every ``*.log`` file."""
    accuracies: list[float] = []
    aucs: list[float] = []

    for log_file in output_folder.rglob("*.log"):
        try:
            lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as exc:
            print(f"Could not read {log_file}: {exc}")
            continue

        for line in lines:
            match = TEST_METRICS.search(line)
            if match:
                accuracies.append(float(match.group(1)))
                aucs.append(float(match.group(2)))

    return accuracies, aucs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_folder",
        type=Path,
        help="Hydra output folder containing one or more .log files.",
    )
    args = parser.parse_args()

    if not args.output_folder.is_dir():
        parser.error(f"{args.output_folder} is not a directory")

    accuracies, aucs = analyze_log_files(args.output_folder)
    if not accuracies:
        print("No valid log files found.")
        return

    print(f"Runs: {len(accuracies)}")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Mean AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")


if __name__ == "__main__":
    main()
