#!/usr/bin/env python3
"""Assemble the final benchmark slices and render the final figures.

The benchmark runs themselves are collected by ``loss_simulator_benchmark.py``.
This report keeps every timed slice (including its batch size and source CSV),
normalizes times to seconds per sample, and plots the favorable batch settings
alongside the one-shot latency controls.
"""

from __future__ import annotations

import base64
import csv
import io
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"
RAW = RESULTS / "final_raw"
DISTANCES = (5, 7, 9, 11, 15, 20, 25, 30, 40, 50)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_rows(
    out: list[dict[str, str]],
    source: Path,
    *,
    mode: str,
    shots: int,
    variant: str | None = None,
    simulator: str | None = None,
) -> None:
    for row in read_rows(source):
        if variant is not None and row["variant"] != variant:
            continue
        if simulator is not None and row["simulator"] != simulator:
            continue
        minimum = row["min_seconds"]
        out.append(
            {
                "distance": row["distance"],
                "variant": row["variant"],
                "simulator": row["simulator"],
                "mode": mode,
                "shots": str(shots),
                "min_seconds": minimum,
                "seconds_per_sample": "" if not minimum else f"{float(minimum) / shots:.12g}",
                "source": str(source.relative_to(ROOT)),
                "error": row.get("error", ""),
            }
        )


def assemble() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    base = RESULTS / "benchmark_results.csv"
    # The earlier five-run one-shot run supplies the full latency curves and
    # the distances where batch sampling was impractical.
    append_rows(rows, base, mode="one-shot", shots=1)

    # Current final no-loss batch slices. Compilation happened in build_case,
    # before the timed region, for both compiled backends.
    for name, simulator in (
        ("stim_compiled_batch1024.csv", "Stim compiled"),
        ("clifft_compiled_batch1024.csv", "Clifft compiled"),
        ("clifft_noncomp_batch1024_d5d7.csv", "Clifft"),
        ("clifft_noncomp_batch1024_rest.csv", "Clifft"),
        ("ppvm_batch1024.csv", "ppvm"),
        ("ppvm_batch1024_d20d25.csv", "ppvm"),
    ):
        append_rows(
            rows,
            RAW / name,
            mode="batch1024",
            shots=1024,
            variant="no_loss",
            simulator=simulator,
        )

    # LOSS has no compiled sampler in any of the three APIs. The batch slices
    # below are therefore throughput checks, not compiled curves.
    for name in ("loss_batch1024_d5.csv", "loss_batch1024_d7_stim_ppvm.csv"):
        append_rows(rows, RAW / name, mode="batch1024", shots=1024, variant="loss")
    return rows


def write_csv(rows: Iterable[dict[str, str]]) -> Path:
    path = RESULTS / "final_benchmark_results.csv"
    fields = [
        "distance",
        "variant",
        "simulator",
        "mode",
        "shots",
        "min_seconds",
        "seconds_per_sample",
        "source",
        "error",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            sorted(
                rows,
                key=lambda row: (
                    row["variant"],
                    int(row["distance"]),
                    row["simulator"],
                    row["mode"],
                ),
            )
        )
    return path


def plot(rows: list[dict[str, str]], path: Path, variant: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"Stim": "#2563eb", "ppvm": "#ea580c", "Clifft": "#16a34a"}
    series = (
        ("Stim", "Stim Tableau (one-shot)", "one-shot", ":", "o"),
        ("Stim compiled", "Stim compiled (batch 1024)", "batch1024", "--", "D"),
        ("ppvm", "ppvm (one-shot)", "one-shot", ":", "s"),
        ("ppvm", "ppvm (batch 1024)", "batch1024", "-", "s"),
        ("Clifft", "Clifft non-compiled (one-shot)", "one-shot", ":", "^"),
        ("Clifft", "Clifft non-compiled (batch 1024)", "batch1024", "-", "^"),
        ("Clifft compiled", "Clifft compiled (batch 1024)", "batch1024", "--", "P"),
    )

    figure, axis = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    for simulator, label, mode, linestyle, marker in series:
        points = sorted(
            (
                int(row["distance"]),
                float(row["seconds_per_sample"]),
            )
            for row in rows
            if row["variant"] == variant
            and row["simulator"] == simulator
            and row["mode"] == mode
            and row["seconds_per_sample"]
        )
        # A mode/backend may have no points in this variant.
        if not points:
            continue
        axis.plot(
            [x for x, _ in points],
            [y for _, y in points],
            color=colors[simulator.removesuffix(" compiled")],
            linestyle=linestyle,
            marker=marker,
            linewidth=1.8,
            markersize=5.5,
            label=label,
        )

    # LOSS batch points are useful as an explicit throughput check, but there
    # are not enough affordable points to connect them into a misleading curve.
    if variant == "loss":
        for simulator in ("Stim", "ppvm", "Clifft"):
            points = sorted(
                (
                    int(row["distance"]),
                    float(row["seconds_per_sample"]),
                )
                for row in rows
                if row["variant"] == "loss"
                and row["simulator"] == simulator
                and row["mode"] == "batch1024"
                and row["seconds_per_sample"]
            )
            if points:
                axis.plot(
                    [x for x, _ in points],
                    [y for _, y in points],
                    color=colors[simulator],
                    linestyle="None",
                    marker="X",
                    markersize=7,
                    label=f"{simulator} LOSS (batch 1024 check)",
                )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(DISTANCES)
    axis.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axis.set_xlabel("Surface-code distance d")
    axis.set_ylabel("Time per sample (seconds; minimum of 5 runs)")
    if variant == "no_loss":
        axis.set_title("Surface-code simulation without LOSS")
    else:
        axis.set_title("Surface-code simulation with LOSS(0.001) after every gate")
    axis.grid(True, which="both", color="#cbd5e1", alpha=0.45, linewidth=0.7)
    axis.legend(frameon=False, fontsize=8)
    figure.savefig(path, dpi=200)
    plt.close(figure)


def make_html(no_loss: Path, loss: Path) -> Path:
    def data_url(path: Path) -> str:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    html_path = Path("/workspace/final-simulator-benchmark.html")
    fragment = f'''<div id="final-simulator-benchmark">
  <style>
    #final-simulator-benchmark {{ display: grid; gap: 1rem; }}
    #final-simulator-benchmark figure {{ margin: 0; }}
    #final-simulator-benchmark img {{ display: block; width: 100%; height: auto; }}
    #final-simulator-benchmark figcaption {{ font-size: 0.9rem; margin-top: 0.35rem; }}
  </style>
  <figure>
    <img src="{data_url(no_loss)}" alt="Log-log plot of no-loss surface-code time per sample versus distance, with dotted one-shot controls, batch-1024 curves, and dashed compiled curves.">
    <figcaption>No loss: seconds per sample; one-shot controls are dotted and compiled curves are dashed.</figcaption>
  </figure>
  <figure>
    <img src="{data_url(loss)}" alt="Log-log plot of LOSS surface-code time per sample versus distance, showing one-shot curves and batch-1024 throughput checks.">
    <figcaption>LOSS: one-shot curves span the requested distances; X markers are batch-1024 checks where the run was affordable.</figcaption>
  </figure>
</div>
'''
    html_path.write_text(fragment)
    return html_path


def main() -> None:
    rows = assemble()
    write_csv(rows)
    no_loss = RESULTS / "final_benchmark_no_loss.png"
    loss = RESULTS / "final_benchmark_loss.png"
    plot(rows, no_loss, "no_loss")
    plot(rows, loss, "loss")
    # The visualization surface is kept outside the git repository so it can
    # be rendered directly in the conversation without adding a large HTML
    # data URL to the benchmark PR.
    make_html(no_loss, loss)
    print(no_loss)
    print(loss)
    print("/workspace/final-simulator-benchmark.html")


if __name__ == "__main__":
    main()
