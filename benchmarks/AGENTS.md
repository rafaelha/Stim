# Local LOSS benchmark instructions

When asked to reproduce the LOSS benchmark on the user's CPU, work from the
repository root and use `benchmarks/run_local_loss_benchmark.py`. Do not run
datapoints concurrently: concurrent simulation would contaminate timings.

## Environment setup

1. Create and activate a Python 3.12 virtual environment.
2. Build/install this checked-out Stim branch with `python -m pip install -e .`.
3. Install the benchmark dependencies with the pinned commands in the
   "Local four-simulator rerun" section of `benchmarks/README.md`.
4. Run `python benchmarks/loss_simulator_benchmark.py --self-test` before the
   full benchmark. Stop and report the diagnostic if it fails.

## Benchmark run

Run:

```bash
python benchmarks/run_local_loss_benchmark.py --timeout 60
```

The timeout applies separately to each simulator/distance datapoint and covers
all five repetitions. A timeout is expected for sufficiently expensive points;
the runner records it in the raw CSV and continues. Never replace a timed-out
point with an estimate.

Afterward, verify that:

- `benchmarks/results/final_benchmark_loss.png` exists;
- `benchmarks/results/final_benchmark_results.csv` contains local LOSS rows;
- each plotted simulator has at least two successful points; and
- parsing, model setup, and QDK QIR compilation remain outside the harness's
  measured region.

Report the CPU model, logical CPU count, available SIMD flags, package versions,
successful distance range for each simulator, and the figure path. Preserve the
raw CSVs under `benchmarks/results/final_raw/`.
