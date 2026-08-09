#!/usr/bin/env python3
"""Measure cold parsing, warm calls, and batch amortization.

The regular benchmark deliberately constructs circuits and simulator models
before entering its timed region. This audit makes that boundary explicit and
measures the pieces separately for Stim and Clifft, both without LOSS and with
LOSS after every gate. ``warm_batch`` is one API call with ``shots`` samples;
``repeat_one_shot`` is the same number of separate one-shot API calls. The
comparison distinguishes per-call setup from per-sample simulation work.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

from loss_simulator_benchmark import (
    CLIFFT_LOSS_CLASSIFIER,
    DISTANCES,
    LOSS_PROBABILITY,
    NOISE_PROBABILITY,
    _imports,
    _run_clifft,
    _run_stim,
    add_loss_after_gates,
)


DEFAULT_BATCHES = (1, 4, 16, 64, 256, 1024)
REPEATS = 5
FIELDS = [
    "distance",
    "variant",
    "backend",
    "mode",
    "shots",
    "min_seconds",
    "per_sample_seconds",
    "run_seconds",
    "error",
]


def _timed(function: Callable[[], None], repeats: int) -> tuple[float, list[float]]:
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        timings.append(time.perf_counter() - started)
    return min(timings), timings


def _row(
    *,
    distance: int,
    variant: str,
    backend: str,
    mode: str,
    shots: int,
    minimum: float | None,
    timings: list[float],
    error: str = "",
) -> dict[str, str]:
    return {
        "distance": str(distance),
        "variant": variant,
        "backend": backend,
        "mode": mode,
        "shots": str(shots),
        "min_seconds": "" if minimum is None else f"{minimum:.12g}",
        "per_sample_seconds": ""
        if minimum is None
        else f"{minimum / shots:.12g}",
        "run_seconds": json.dumps(timings, separators=(",", ":")),
        "error": error,
    }


def _measure(
    *,
    distance: int,
    variant: str,
    backend: str,
    mode: str,
    shots: int,
    function: Callable[[], None],
    repeats: int,
) -> dict[str, str]:
    try:
        minimum, timings = _timed(function, repeats)
        print(
            f"  {variant:7s} {backend:6s} {mode:18s} shots={shots:4d} "
            f"min={minimum:.6g}s"
        )
        return _row(
            distance=distance,
            variant=variant,
            backend=backend,
            mode=mode,
            shots=shots,
            minimum=minimum,
            timings=timings,
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"  {variant:7s} {backend:6s} {mode:18s} ERROR {message}")
        return _row(
            distance=distance,
            variant=variant,
            backend=backend,
            mode=mode,
            shots=shots,
            minimum=None,
            timings=[],
            error=message,
        )


def run_audit(
    *,
    distances: tuple[int, ...],
    batches: tuple[int, ...],
    repeats: int,
    output_csv: Path,
) -> list[dict[str, str]]:
    stim, _, clifft, noncomp, _ = _imports(require_ppvm=False)
    no_loss_model = noncomp.Model()
    loss_model = noncomp.Model(
        classifier=noncomp.Classifier(CLIFFT_LOSS_CLASSIFIER)
    )
    rows: list[dict[str, str]] = []

    for distance in distances:
        circuit = stim.Circuit.generated(
            "surface_code:rotated_memory_x",
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=NOISE_PROBABILITY,
            before_round_data_depolarization=NOISE_PROBABILITY,
            after_reset_flip_probability=NOISE_PROBABILITY,
            before_measure_flip_probability=NOISE_PROBABILITY,
        )
        loss_circuit = add_loss_after_gates(stim, circuit, LOSS_PROBABILITY)
        stim_circuits = {"no_loss": circuit, "loss": loss_circuit}
        clifft_circuits = {
            variant: clifft.parse(str(source))
            for variant, source in stim_circuits.items()
        }
        clifft_models = {"no_loss": no_loss_model, "loss": loss_model}

        for variant in ("no_loss", "loss"):
            stim_circuit = stim_circuits[variant]
            clifft_circuit = clifft_circuits[variant]
            model = clifft_models[variant]
            stim_text = str(stim_circuit)
            clifft_text = stim_text

            # Parsing is not in the regular benchmark's warm timing. These
            # rows quantify the cost when a caller starts from circuit text.
            rows.append(
                _measure(
                    distance=distance,
                    variant=variant,
                    backend="Stim",
                    mode="parse_only",
                    shots=1,
                    function=lambda text=stim_text: stim.Circuit(text),
                    repeats=repeats,
                )
            )
            rows.append(
                _measure(
                    distance=distance,
                    variant=variant,
                    backend="Clifft",
                    mode="parse_only",
                    shots=1,
                    function=lambda text=clifft_text: clifft.parse(text),
                    repeats=repeats,
                )
            )
            rows.append(
                _measure(
                    distance=distance,
                    variant=variant,
                    backend="Stim",
                    mode="parse_plus_sample",
                    shots=1,
                    function=lambda c=stim_circuit, text=stim_text: (
                        stim.TableauSimulator(seed=11).do(stim.Circuit(text))
                    ),
                    repeats=repeats,
                )
            )
            rows.append(
                _measure(
                    distance=distance,
                    variant=variant,
                    backend="Clifft",
                    mode="parse_plus_sample",
                    shots=1,
                    function=lambda text=clifft_text, m=model: _run_clifft(
                        noncomp, clifft.parse(text), m, 11, shots=1
                    ),
                    repeats=repeats,
                )
            )

            for shots in batches:
                rows.append(
                    _measure(
                        distance=distance,
                        variant=variant,
                        backend="Stim",
                        mode="warm_batch",
                        shots=shots,
                        function=lambda c=stim_circuit, n=shots: _run_stim(
                            stim, c, 101, shots=n
                        ),
                        repeats=repeats,
                    )
                )
                rows.append(
                    _measure(
                        distance=distance,
                        variant=variant,
                        backend="Clifft",
                        mode="warm_batch",
                        shots=shots,
                        function=lambda c=clifft_circuit, m=model, n=shots: _run_clifft(
                            noncomp, c, m, 101, shots=n
                        ),
                        repeats=repeats,
                    )
                )
                rows.append(
                    _measure(
                        distance=distance,
                        variant=variant,
                        backend="Stim",
                        mode="repeat_one_shot",
                        shots=shots,
                        function=lambda c=stim_circuit, n=shots: [
                            _run_stim(stim, c, 202 + index, shots=1)
                            for index in range(n)
                        ],
                        repeats=repeats,
                    )
                )
                rows.append(
                    _measure(
                        distance=distance,
                        variant=variant,
                        backend="Clifft",
                        mode="repeat_one_shot",
                        shots=shots,
                        function=lambda c=clifft_circuit, m=model, n=shots: [
                            _run_clifft(noncomp, c, m, 202 + index, shots=1)
                            for index in range(n)
                        ],
                        repeats=repeats,
                    )
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output_csv}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances", type=int, nargs="+", default=list(DISTANCES[:2]))
    parser.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.batches):
        parser.error("repeats and batches must be positive")
    run_audit(
        distances=tuple(args.distances),
        batches=tuple(args.batches),
        repeats=args.repeats,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
