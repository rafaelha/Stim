#!/usr/bin/env python3
"""Benchmark Stim, ppvm, and Clifft on surface-code circuits with loss.

The no-loss control can use either the non-computational APIs or the compiled
samplers. The compiled curves precompile before timing: Stim uses
``Circuit.compile_sampler`` and Clifft uses ``clifft.compile``. LOSS runs use
the non-computational APIs because those are the implementations that support
the LOSS instruction.

The current ppvm parser on ``main`` predates the ``LOSS`` instruction. Its loss
benchmark therefore runs operation groups through a ``GeneralizedTableau`` and
applies ``loss_channel`` immediately afterwards. Disjoint operations can be
grouped without changing circuit semantics.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


DEFAULT_OUTPUT_DIR = Path(__file__).with_name("results")
DISTANCES = (5, 7, 9, 11, 15, 20, 25, 30, 40, 50)
NOISE_PROBABILITY = 0.001
LOSS_PROBABILITY = 0.001
REPEATS = 5
SHOTS = 1

ANNOTATIONS = {
    "DETECTOR",
    "OBSERVABLE_INCLUDE",
    "QUBIT_COORDS",
    "SHIFT_COORDS",
    "TICK",
}

# P(symbol | level), where levels are G, E, LEAK_G, LEAK_E, LOST. Lost sites
# produce the third (herald) symbol, as described by Clifft's noncomp API.
CLIFFT_LOSS_CLASSIFIER = (
    (1.0, 0.0, 1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 1.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 1.0),
)


@dataclasses.dataclass
class CircuitCase:
    distance: int
    num_qubits: int
    stim_circuit: Any
    stim_loss_circuit: Any | None
    stim_compiled_sampler: Any | None
    ppvm_program: Any | None
    ppvm_loss_groups: list[tuple[Any, tuple[int, ...]]]
    clifft_circuit: Any | None
    clifft_loss_circuit: Any | None
    clifft_compiled_program: Any | None


def _imports(require_ppvm: bool = True) -> tuple[Any, Any, Any, Any, Any]:
    """Import benchmark dependencies with a useful error message."""

    try:
        import clifft
        import stim
        from clifft import noncomp
        ppvm = None
        if require_ppvm:
            import ppvm
    except ImportError as exc:  # pragma: no cover - exercised by CLI users.
        raise SystemExit(
            "This benchmark needs the local Stim build plus clifft"
            + (" and ppvm" if require_ppvm else "")
            + ". "
            "See benchmarks/README.md for installation instructions."
        ) from exc
    return stim, ppvm, clifft, noncomp, clifft.parse


def _qubit_targets(operation: Any) -> list[int]:
    return [int(target.value) for target in operation.targets_copy() if target.is_qubit_target]


def add_loss_after_gates(stim_module: Any, circuit: Any, probability: float) -> Any:
    """Return ``circuit`` with one LOSS after each qubit-targeting gate.

    Repeat blocks are preserved. Annotation instructions and TICK are not
    physical gates, so they do not receive losses. A single LOSS instruction
    contains all targets of the preceding instruction; Stim samples those
    targets independently.
    """

    output = stim_module.Circuit()
    for operation in circuit:
        if isinstance(operation, stim_module.CircuitRepeatBlock):
            body = add_loss_after_gates(stim_module, operation.body_copy(), probability)
            output.append(stim_module.CircuitRepeatBlock(operation.repeat_count, body))
            continue

        output.append(operation.name, operation.targets_copy(), operation.gate_args_copy())
        if operation.name in ANNOTATIONS:
            continue
        qubits = _qubit_targets(operation)
        if qubits:
            output.append("LOSS", qubits, probability)
    return output


def _parse_ppvm_program(ppvm_module: Any, circuit: Any) -> Any:
    """Parse a Stim circuit, falling back to flattened text if necessary."""

    try:
        return ppvm_module.StimProgram.parse(str(circuit))
    except ValueError:
        return ppvm_module.StimProgram.parse(str(circuit.flattened()))


def build_ppvm_loss_groups(
    stim_module: Any, ppvm_module: Any, circuit: Any
) -> list[tuple[Any, tuple[int, ...]]]:
    """Build ppvm programs grouped only across disjoint physical operations.

    This is an adapter for ppvm versions whose Stim parser does not yet accept
    LOSS. Operation order is preserved and every touched qubit receives one
    loss trial immediately after its operation (or its disjoint group).
    """

    groups: list[tuple[Any, tuple[int, ...]]] = []
    group = stim_module.Circuit()
    touched: set[int] = set()

    def flush() -> None:
        nonlocal group, touched
        if len(group):
            groups.append(
                (ppvm_module.StimProgram.parse(str(group)), tuple(sorted(touched)))
            )
        group = stim_module.Circuit()
        touched = set()

    # Flattening is only needed for the adapter; Stim and Clifft retain repeats.
    for operation in circuit.flattened():
        if operation.name in ANNOTATIONS:
            continue
        qubits = set(_qubit_targets(operation))
        if not qubits:
            continue
        if touched.intersection(qubits):
            flush()
        group.append(operation.name, operation.targets_copy(), operation.gate_args_copy())
        touched.update(qubits)
    flush()
    return groups


def build_case(
    stim_module: Any,
    ppvm_module: Any,
    clifft_module: Any,
    distance: int,
    noise_probability: float,
    loss_probability: float,
    include_loss: bool = True,
    simulators: Sequence[str] | None = None,
) -> CircuitCase:
    selected_simulators = set(
        simulators
        if simulators is not None
        else ("Stim", "Stim compiled", "ppvm", "Clifft", "Clifft compiled")
    )
    need_stim_loss = include_loss and "Stim" in selected_simulators
    need_clifft = bool(selected_simulators.intersection(("Clifft", "Clifft compiled")))
    need_clifft_loss = include_loss and "Clifft" in selected_simulators
    need_ppvm = "ppvm" in selected_simulators
    need_stim_compiled = "Stim compiled" in selected_simulators

    circuit = stim_module.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=distance,
        rounds=distance,
        after_clifford_depolarization=noise_probability,
        before_round_data_depolarization=noise_probability,
        after_reset_flip_probability=noise_probability,
        before_measure_flip_probability=noise_probability,
    )
    loss_circuit = (
        add_loss_after_gates(stim_module, circuit, loss_probability)
        if need_stim_loss or need_clifft_loss
        else None
    )
    return CircuitCase(
        distance=distance,
        num_qubits=circuit.num_qubits,
        stim_circuit=circuit,
        stim_loss_circuit=loss_circuit,
        stim_compiled_sampler=(circuit.compile_sampler() if need_stim_compiled else None),
        ppvm_program=(
            None if not need_ppvm else _parse_ppvm_program(ppvm_module, circuit)
        ),
        ppvm_loss_groups=(
            []
            if not need_ppvm or not include_loss
            else build_ppvm_loss_groups(stim_module, ppvm_module, circuit)
        ),
        clifft_circuit=(clifft_module.parse(str(circuit)) if need_clifft else None),
        clifft_loss_circuit=(
            None
            if not need_clifft_loss
            else clifft_module.parse(str(loss_circuit))
        ),
        clifft_compiled_program=(
            clifft_module.compile(str(circuit))
            if "Clifft compiled" in selected_simulators
            else None
        ),
    )


def _timed_minimum(
    function: Callable[[int], None], repeats: int, seed_base: int
) -> tuple[float, list[float]]:
    timings: list[float] = []
    for index in range(repeats):
        started = time.perf_counter()
        function(seed_base + index)
        timings.append(time.perf_counter() - started)
    return min(timings), timings


def _run_stim(stim_module: Any, circuit: Any, seed: int, shots: int = SHOTS) -> None:
    for shot in range(shots):
        simulator = stim_module.TableauSimulator(seed=seed + shot)
        simulator.do(circuit)


def _run_stim_compiled(sampler: Any, seed: int, shots: int = SHOTS) -> None:
    # Stim's compiled sampler does not expose a per-call seed in the current
    # Python API.  The benchmark measures execution time, so use its default
    # random stream; the seed argument is retained for the common runner API.
    del seed
    sampler.sample(shots=shots)


def _run_ppvm_no_loss(
    ppvm_module: Any,
    program: Any,
    num_qubits: int,
    seed: int,
    shots: int = SHOTS,
) -> None:
    ppvm_module.sample_stim(program, n_qubits=num_qubits, num_shots=shots, seed=seed)


def _run_ppvm_loss(
    ppvm_module: Any,
    groups: Iterable[tuple[Any, tuple[int, ...]]],
    num_qubits: int,
    probability: float,
    seed: int,
    shots: int = SHOTS,
) -> None:
    for shot in range(shots):
        tableau = ppvm_module.GeneralizedTableau(num_qubits, seed=seed + shot)
        for program, qubits in groups:
            tableau.do(program)
            for qubit in qubits:
                tableau.loss_channel(qubit, probability)


def _run_clifft(
    noncomp_module: Any,
    circuit: Any,
    model: Any,
    seed: int,
    shots: int = SHOTS,
) -> None:
    noncomp_module.sample(circuit, model, shots=shots, seed=seed)


def _run_clifft_compiled(
    clifft_module: Any,
    program: Any,
    seed: int,
    shots: int = SHOTS,
) -> None:
    clifft_module.sample(program, shots=shots, seed=seed)


def _result_row(
    *,
    distance: int,
    variant: str,
    simulator: str,
    minimum: float | None,
    timings: list[float],
    error: str = "",
) -> dict[str, str]:
    return {
        "distance": str(distance),
        "variant": variant,
        "simulator": simulator,
        "min_seconds": "" if minimum is None else f"{minimum:.12g}",
        "run_seconds": json.dumps(timings, separators=(",", ":")),
        "error": error,
    }


def _time_one(
    function: Callable[[int], None],
    *,
    distance: int,
    variant: str,
    simulator: str,
    repeats: int,
    seed_base: int,
) -> dict[str, str]:
    try:
        minimum, timings = _timed_minimum(function, repeats, seed_base)
        print(
            f"  {variant:7s} {simulator:7s}: min={minimum:.6g}s "
            f"runs={[round(value, 6) for value in timings]}"
        )
        return _result_row(
            distance=distance,
            variant=variant,
            simulator=simulator,
            minimum=minimum,
            timings=timings,
        )
    except Exception as exc:  # Keep other curves running if one package fails.
        message = f"{type(exc).__name__}: {exc}"
        print(f"  {variant:7s} {simulator:7s}: ERROR {message}")
        return _result_row(
            distance=distance,
            variant=variant,
            simulator=simulator,
            minimum=None,
            timings=[],
            error=message,
        )


def _plot_results(
    rows: list[dict[str, str]],
    output_dir: Path,
    shots: int = SHOTS,
    per_sample: bool = False,
    filename_suffix: str = "",
) -> tuple[Path, Path]:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "Stim": "#2563eb",
        "Stim compiled": "#7c3aed",
        "ppvm": "#ea580c",
        "Clifft": "#16a34a",
        "Clifft compiled": "#0891b2",
    }
    markers = {
        "Stim": "o",
        "Stim compiled": "D",
        "ppvm": "s",
        "Clifft": "^",
        "Clifft compiled": "P",
    }
    paths: list[Path] = []
    for variant, title, filename in (
        ("no_loss", "Surface-code simulation without LOSS", "benchmark_no_loss.png"),
        ("loss", "Surface-code simulation with LOSS after every gate", "benchmark_loss.png"),
    ):
        figure, axis = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
        for simulator in colors:
            points = sorted(
                (
                    int(row["distance"]),
                    float(row["min_seconds"]) / shots if per_sample else float(row["min_seconds"]),
                )
                for row in rows
                if row["variant"] == variant
                and row["simulator"] == simulator
                and row["min_seconds"]
            )
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color=colors[simulator],
                    marker=markers[simulator],
                    linewidth=1.8,
                    markersize=5.5,
                    label=simulator,
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Surface-code distance d")
        if per_sample:
            axis.set_ylabel(f"Time per sample (seconds, batch size {shots}, min of 5 runs)")
            axis.set_title(f"{title} (batch size {shots})")
        else:
            axis.set_ylabel("Minimum runtime (seconds, 5 runs)")
            axis.set_title(title)
        axis.grid(True, which="both", color="#cbd5e1", alpha=0.45, linewidth=0.7)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(frameon=False)
        path = output_dir / filename.replace(".png", f"{filename_suffix}.png")
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(path)
    return paths[0], paths[1]


def run_benchmark(
    *,
    distances: Sequence[int] = DISTANCES,
    repeats: int = REPEATS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    noise_probability: float = NOISE_PROBABILITY,
    loss_probability: float = LOSS_PROBABILITY,
    simulators: Sequence[str] = ("Stim", "ppvm", "Clifft"),
    variants: Sequence[str] = ("no_loss", "loss"),
    shots: int = SHOTS,
    per_sample: bool = False,
    plot_suffix: str = "",
    csv_name: str = "benchmark_results.csv",
) -> list[dict[str, str]]:
    if shots < 1:
        raise ValueError("shots must be positive")
    selected_simulators = set(simulators)
    selected_variants = set(variants)
    stim_module, ppvm_module, clifft_module, noncomp_module, _ = _imports(
        require_ppvm="ppvm" in selected_simulators
    )
    include_loss = "loss" in selected_variants
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_name
    rows: list[dict[str, str]] = []
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["distance", "variant", "simulator", "min_seconds", "run_seconds", "error"],
            lineterminator="\n",
        )
        writer.writeheader()
        loss_model = noncomp_module.Model(
            classifier=noncomp_module.Classifier(CLIFFT_LOSS_CLASSIFIER)
        )
        no_loss_model = noncomp_module.Model()

        for distance in distances:
            print(f"distance={distance}")
            case = build_case(
                stim_module,
                ppvm_module,
                clifft_module,
                distance,
                noise_probability,
                loss_probability,
                include_loss=include_loss,
                simulators=selected_simulators,
            )
            seed_base = 100_000 + distance * 10_000
            case_rows: list[dict[str, str]] = []
            if "Stim" in selected_simulators and "no_loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_stim(
                            stim_module, c.stim_circuit, seed, shots=shots
                        ),
                        distance=distance,
                        variant="no_loss",
                        simulator="Stim",
                        repeats=repeats,
                        seed_base=seed_base,
                    )
                )
            if "Stim compiled" in selected_simulators and "no_loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_stim_compiled(
                            c.stim_compiled_sampler, seed, shots=shots
                        ),
                        distance=distance,
                        variant="no_loss",
                        simulator="Stim compiled",
                        repeats=repeats,
                        seed_base=seed_base + 50,
                    )
                )
            if "ppvm" in selected_simulators and "no_loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_ppvm_no_loss(
                            ppvm_module, c.ppvm_program, c.num_qubits, seed, shots=shots
                        ),
                        distance=distance,
                        variant="no_loss",
                        simulator="ppvm",
                        repeats=repeats,
                        seed_base=seed_base + 100,
                    )
                )
            if "Clifft" in selected_simulators and "no_loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_clifft(
                            noncomp_module, c.clifft_circuit, no_loss_model, seed, shots=shots
                        ),
                        distance=distance,
                        variant="no_loss",
                        simulator="Clifft",
                        repeats=repeats,
                        seed_base=seed_base + 200,
                    )
                )
            if "Clifft compiled" in selected_simulators and "no_loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_clifft_compiled(
                            clifft_module, c.clifft_compiled_program, seed, shots=shots
                        ),
                        distance=distance,
                        variant="no_loss",
                        simulator="Clifft compiled",
                        repeats=repeats,
                        seed_base=seed_base + 250,
                    )
                )
            if "Stim" in selected_simulators and "loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_stim(
                            stim_module, c.stim_loss_circuit, seed, shots=shots
                        ),
                        distance=distance,
                        variant="loss",
                        simulator="Stim",
                        repeats=repeats,
                        seed_base=seed_base + 300,
                    )
                )
            if "ppvm" in selected_simulators and "loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_ppvm_loss(
                            ppvm_module,
                            c.ppvm_loss_groups,
                            c.num_qubits,
                            loss_probability,
                            seed,
                            shots=shots,
                        ),
                        distance=distance,
                        variant="loss",
                        simulator="ppvm",
                        repeats=repeats,
                        seed_base=seed_base + 400,
                    )
                )
            if "Clifft" in selected_simulators and "loss" in selected_variants:
                case_rows.append(
                    _time_one(
                        lambda seed, c=case: _run_clifft(
                            noncomp_module,
                            c.clifft_loss_circuit,
                            loss_model,
                            seed,
                            shots=shots,
                        ),
                        distance=distance,
                        variant="loss",
                        simulator="Clifft",
                        repeats=repeats,
                        seed_base=seed_base + 500,
                    )
                )
            rows.extend(case_rows)
            writer.writerows(case_rows)
            handle.flush()

    no_loss_path, loss_path = _plot_results(
        rows,
        output_dir,
        shots=shots,
        per_sample=per_sample,
        filename_suffix=plot_suffix,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {no_loss_path}")
    print(f"wrote {loss_path}")
    return rows


def self_test() -> None:
    """Exercise the loss expansion and both external APIs at distance five."""

    stim_module, ppvm_module, clifft_module, noncomp_module, _ = _imports()
    body = stim_module.Circuit()
    body.append("H", [0])
    body.append("TICK")
    body.append("X", [0])
    circuit = stim_module.Circuit()
    circuit.append(stim_module.CircuitRepeatBlock(2, body))
    loss_circuit = add_loss_after_gates(stim_module, circuit, LOSS_PROBABILITY)
    assert sum(name == "LOSS" for name, _, _ in loss_circuit.flattened_operations()) == 4
    assert "REPEAT 2" in str(loss_circuit)

    case = build_case(
        stim_module,
        ppvm_module,
        clifft_module,
        5,
        NOISE_PROBABILITY,
        LOSS_PROBABILITY,
    )
    _run_stim(stim_module, case.stim_loss_circuit, 1)
    _run_ppvm_no_loss(ppvm_module, case.ppvm_program, case.num_qubits, 2)
    _run_ppvm_loss(
        ppvm_module,
        case.ppvm_loss_groups,
        case.num_qubits,
        LOSS_PROBABILITY,
        3,
    )
    no_loss_model = noncomp_module.Model()
    loss_model = noncomp_module.Model(
        classifier=noncomp_module.Classifier(CLIFFT_LOSS_CLASSIFIER)
    )
    _run_clifft(noncomp_module, case.clifft_circuit, no_loss_model, 4)
    _run_clifft(noncomp_module, case.clifft_loss_circuit, loss_model, 5)
    print("self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--shots", type=int, default=SHOTS)
    parser.add_argument("--per-sample", action="store_true")
    parser.add_argument("--plot-suffix", default="")
    parser.add_argument("--csv-name", default="benchmark_results.csv")
    parser.add_argument("--distances", type=int, nargs="+", default=list(DISTANCES))
    parser.add_argument(
        "--simulators",
        choices=("Stim", "Stim compiled", "ppvm", "Clifft", "Clifft compiled"),
        nargs="+",
        default=["Stim", "ppvm", "Clifft"],
    )
    parser.add_argument(
        "--variants",
        choices=("no_loss", "loss"),
        nargs="+",
        default=["no_loss", "loss"],
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    run_benchmark(
        distances=args.distances,
        repeats=args.repeats,
        output_dir=args.output_dir,
        simulators=args.simulators,
        variants=args.variants,
        shots=args.shots,
        per_sample=args.per_sample,
        plot_suffix=args.plot_suffix,
        csv_name=args.csv_name,
    )


if __name__ == "__main__":
    main()
