#!/usr/bin/env python3
"""Merge independently collected benchmark slices and regenerate figures.

This is useful when one backend cannot complete the largest circuit without
stopping the process. Successful rows remain exact five-run minima; unavailable
points are retained as rows with an explanatory ``error`` field and are omitted
from the plotted curve.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from loss_simulator_benchmark import DISTANCES, _plot_results


FIELDS = ["distance", "variant", "simulator", "min_seconds", "run_seconds", "error"]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _missing_row(distance: int, variant: str, error: str) -> dict[str, str]:
    return {
        "distance": str(distance),
        "variant": variant,
        "simulator": "Clifft",
        "min_seconds": "",
        "run_seconds": "[]",
        "error": error,
    }


def merge(
    native_csv: Path,
    clifft_no_loss_csv: Path,
    clifft_loss_csv: Path,
    output_dir: Path,
    clifft_loss_distance_15_runs: list[float],
) -> None:
    rows = _read(native_csv) + _read(clifft_no_loss_csv) + _read(clifft_loss_csv)
    rows.append(
        {
            "distance": "15",
            "variant": "loss",
            "simulator": "Clifft",
            "min_seconds": f"{min(clifft_loss_distance_15_runs):.12g}",
            "run_seconds": json.dumps(clifft_loss_distance_15_runs, separators=(",", ":")),
            "error": "",
        }
    )

    existing = {(row["distance"], row["variant"], row["simulator"]) for row in rows}
    for distance in DISTANCES:
        key = (str(distance), "no_loss", "Clifft")
        if key not in existing:
            rows.append(
                _missing_row(
                    distance,
                    "no_loss",
                    "not measured: Clifft was killed by the machine while preparing d=40",
                )
            )
        key = (str(distance), "loss", "Clifft")
        if key not in existing:
            rows.append(
                _missing_row(
                    distance,
                    "loss",
                    "not measured: Clifft LOSS exceeded the available runtime after d=15",
                )
            )

    expected = {
        (str(distance), variant, simulator)
        for distance in DISTANCES
        for variant in ("no_loss", "loss")
        for simulator in ("Stim", "ppvm", "Clifft")
    }
    actual = {(row["distance"], row["variant"], row["simulator"]) for row in rows}
    if actual != expected:
        raise ValueError(f"result key mismatch: missing={expected - actual}, extra={actual - expected}")

    rows.sort(key=lambda row: (int(row["distance"]), row["variant"], row["simulator"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "benchmark_results.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    no_loss_path, loss_path = _plot_results(rows, output_dir)
    print(f"wrote {output_csv}")
    print(f"wrote {no_loss_path}")
    print(f"wrote {loss_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-csv", type=Path, required=True)
    parser.add_argument("--clifft-no-loss-csv", type=Path, required=True)
    parser.add_argument("--clifft-loss-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--clifft-loss-distance-15-runs",
        type=float,
        nargs=5,
        required=True,
        metavar="SECONDS",
    )
    args = parser.parse_args()
    merge(
        args.native_csv,
        args.clifft_no_loss_csv,
        args.clifft_loss_csv,
        args.output_dir,
        args.clifft_loss_distance_15_runs,
    )


if __name__ == "__main__":
    main()
