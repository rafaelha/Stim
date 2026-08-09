#!/usr/bin/env python3
"""Merge the batch-1024 compiled simulator slices and plot per-sample time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loss_simulator_benchmark import DISTANCES, _plot_results


FIELDS = ["distance", "variant", "simulator", "min_seconds", "run_seconds", "error"]
SHOTS = 1024


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _missing_row(distance: int, simulator: str, error: str) -> dict[str, str]:
    return {
        "distance": str(distance),
        "variant": "no_loss",
        "simulator": simulator,
        "min_seconds": "",
        "run_seconds": "[]",
        "error": error,
    }


def merge(stim_csv: Path, clifft_csv: Path, output_dir: Path) -> None:
    rows = _read(stim_csv) + _read(clifft_csv)
    keys = [(row["distance"], row["variant"], row["simulator"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("input slices contain duplicate result keys")

    existing = set(keys)
    for distance in DISTANCES:
        key = (str(distance), "no_loss", "Stim compiled")
        if key not in existing:
            raise ValueError(f"missing Stim compiled result for d={distance}")
        key = (str(distance), "no_loss", "Clifft compiled")
        if key not in existing:
            rows.append(
                _missing_row(
                    distance,
                    "Clifft compiled",
                    "not measured: clifft.compile was killed by the machine at d=50 (exit 137)",
                )
            )

    expected = {
        (str(distance), "no_loss", simulator)
        for distance in DISTANCES
        for simulator in ("Stim compiled", "Clifft compiled")
    }
    actual = {(row["distance"], row["variant"], row["simulator"]) for row in rows}
    if actual != expected:
        raise ValueError(f"result key mismatch: missing={expected - actual}, extra={actual - expected}")

    rows.sort(key=lambda row: (int(row["distance"]), row["simulator"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "benchmark_results_batch1024.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    no_loss_path, _ = _plot_results(
        rows,
        output_dir,
        shots=SHOTS,
        per_sample=True,
        filename_suffix="_batch1024",
    )
    print(f"wrote {output_csv}")
    print(f"wrote {no_loss_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stim-csv", type=Path, required=True)
    parser.add_argument("--clifft-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merge(args.stim_csv, args.clifft_csv, args.output_dir)


if __name__ == "__main__":
    main()
