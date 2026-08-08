#!/usr/bin/env python3
"""Merge compiled-simulator slices into the recorded benchmark results.

Stim and Clifft compiled runs are collected separately so that Clifft's
distance-50 compiler failure cannot terminate the other measurements.  This
utility keeps successful rows exact, adds explicit unavailable rows, and
regenerates both log-log figures.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from loss_simulator_benchmark import DISTANCES, _plot_results


FIELDS = ["distance", "variant", "simulator", "min_seconds", "run_seconds", "error"]
COMPILED_SIMULATORS = ("Stim compiled", "Clifft compiled")


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


def merge(
    base_csv: Path,
    stim_compiled_csv: Path,
    clifft_compiled_csv: Path,
    output_dir: Path,
) -> None:
    # Allow rerunning the merge against an already merged output: compiled
    # rows from the base are replaced by the freshly collected slices.
    rows = [
        row
        for row in _read(base_csv)
        if row["simulator"] not in COMPILED_SIMULATORS
    ]
    rows += _read(stim_compiled_csv) + _read(clifft_compiled_csv)
    keys = [(row["distance"], row["variant"], row["simulator"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("input slices contain duplicate result keys")

    existing = set(keys)
    for distance in DISTANCES:
        stim_key = (str(distance), "no_loss", "Stim compiled")
        if stim_key not in existing:
            rows.append(
                _missing_row(
                    distance,
                    "Stim compiled",
                    "not measured: Stim compiled slice did not contain this distance",
                )
            )
        clifft_key = (str(distance), "no_loss", "Clifft compiled")
        if clifft_key not in existing:
            rows.append(
                _missing_row(
                    distance,
                    "Clifft compiled",
                    "not measured: clifft.compile was killed by the machine at d=50 (exit 137)",
                )
            )

    expected = {
        (str(distance), variant, simulator)
        for distance in DISTANCES
        for variant in ("no_loss", "loss")
        for simulator in ("Stim", "ppvm", "Clifft")
    }
    expected.update(
        (str(distance), "no_loss", simulator)
        for distance in DISTANCES
        for simulator in COMPILED_SIMULATORS
    )
    actual = {(row["distance"], row["variant"], row["simulator"]) for row in rows}
    if actual != expected:
        raise ValueError(f"result key mismatch: missing={expected - actual}, extra={actual - expected}")

    rows.sort(key=lambda row: (int(row["distance"]), row["variant"], row["simulator"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "benchmark_results.csv"
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    no_loss_path, loss_path = _plot_results(rows, output_dir)
    print(f"wrote {output_csv}")
    print(f"wrote {no_loss_path}")
    print(f"wrote {loss_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--stim-compiled-csv", type=Path, required=True)
    parser.add_argument("--clifft-compiled-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    merge(
        args.base_csv,
        args.stim_compiled_csv,
        args.clifft_compiled_csv,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
