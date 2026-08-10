#!/usr/bin/env python3
"""Run the four-simulator LOSS benchmark with a wall-clock limit per point.

Each simulator/distance pair runs in its own subprocess. This makes the
timeout robust even when a native extension is executing and preserves all
completed points if a later point is too slow. Successful rows are assembled
into the filenames consumed by ``final_benchmark_report.py``.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "benchmarks" / "loss_simulator_benchmark.py"
REPORT = ROOT / "benchmarks" / "final_benchmark_report.py"
DEFAULT_RAW_DIR = ROOT / "benchmarks" / "results" / "final_raw"
DISTANCES = (5, 7, 9, 11, 15, 20, 25, 30, 40, 50)
SIMULATORS = {
    "Stim": "loss_rerun_stim.csv",
    "ppvm": "loss_rerun_ppvm.csv",
    "Clifft": "loss_rerun_clifft.csv",
    "QDK-EC": "loss_rerun_qdk_ec.csv",
}
FIELDS = ("distance", "variant", "simulator", "min_seconds", "run_seconds", "error")


def unavailable_row(distance: int, simulator: str, error: str) -> dict[str, str]:
    return {
        "distance": str(distance),
        "variant": "loss",
        "simulator": simulator,
        "min_seconds": "",
        "run_seconds": "[]",
        "error": error,
    }


def read_result(path: Path, distance: int, simulator: str) -> dict[str, str]:
    if not path.exists():
        return unavailable_row(distance, simulator, "benchmark exited without a result row")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        return unavailable_row(
            distance,
            simulator,
            f"benchmark produced {len(rows)} rows instead of one",
        )
    return {field: rows[0].get(field, "") for field in FIELDS}


def run_point(
    *,
    simulator: str,
    distance: int,
    repeats: int,
    timeout: float,
    work_dir: Path,
    environment: dict[str, str],
) -> dict[str, str]:
    point_dir = work_dir / simulator.lower().replace("-", "_").replace(" ", "_") / f"d{distance}"
    point_dir.mkdir(parents=True, exist_ok=True)
    result_path = point_dir / "result.csv"
    command = [
        sys.executable,
        str(HARNESS),
        "--simulators",
        simulator,
        "--variants",
        "loss",
        "--distances",
        str(distance),
        "--repeats",
        str(repeats),
        "--shots",
        "1",
        "--per-sample",
        "--output-dir",
        str(point_dir),
        "--csv-name",
        result_path.name,
        "--plot-suffix",
        "_local_point",
    ]
    print(f"{simulator:7s} d={distance:<2d} ... ", end="", flush=True)
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT ({timeout:g}s)")
        return unavailable_row(
            distance,
            simulator,
            f"timeout: datapoint exceeded {timeout:g} seconds wall clock",
        )

    row = read_result(result_path, distance, simulator)
    if completed.returncode and not row["error"]:
        tail = completed.stdout.strip().splitlines()[-1:] or ["no diagnostic output"]
        row["error"] = f"exit {completed.returncode}: {tail[0]}"
        row["min_seconds"] = ""
        row["run_seconds"] = "[]"
    if row["min_seconds"]:
        print(f"{float(row['min_seconds']):.6g}s")
    else:
        print(f"SKIP ({row['error']})")
    return row


def write_backend_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--distances", type=int, nargs="+", default=list(DISTANCES))
    parser.add_argument("--raw-output-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--work-dir", type=Path, default=ROOT / ".benchmark_local_points")
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="collect raw CSVs without regenerating the final figure",
    )
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.repeats < 1:
        parser.error("--repeats must be positive")

    environment = os.environ.copy()
    environment.setdefault("QDK_PYTHON_TELEMETRY", "none")
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".benchmark_mplconfig"))
    environment.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 1))

    args.raw_output_dir.mkdir(parents=True, exist_ok=True)
    for simulator, filename in SIMULATORS.items():
        rows = [
            run_point(
                simulator=simulator,
                distance=distance,
                repeats=args.repeats,
                timeout=args.timeout,
                work_dir=args.work_dir,
                environment=environment,
            )
            for distance in args.distances
        ]
        write_backend_csv(args.raw_output_dir / filename, rows)

    if args.no_report:
        print(f"wrote raw results to {args.raw_output_dir}")
        return
    if args.raw_output_dir.resolve() != DEFAULT_RAW_DIR.resolve():
        parser.error("report generation requires the default --raw-output-dir")
    subprocess.run([sys.executable, str(REPORT)], cwd=ROOT, env=environment, check=True)
    print(ROOT / "benchmarks" / "results" / "final_benchmark_loss.png")


if __name__ == "__main__":
    main()
