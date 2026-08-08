# LOSS simulator benchmark

`loss_simulator_benchmark.py` compares the LOSS-enabled Stim branch with QuEra's `ppvm` and Clifft. It generates `surface_code:rotated_memory_x` circuits at distances `5, 7, 9, 11, 15, 20, 25, 30, 40, 50`, with `rounds=distance` and all four Stim noise parameters set to `0.001`:

```text
after_clifford_depolarization
before_round_data_depolarization
after_reset_flip_probability
before_measure_flip_probability
```

Each point is one shot and the reported value is the minimum of five timed runs. Circuit parsing and model construction happen before timing. The `Stim`, `ppvm`, and `Clifft` rows use their non-compiled/tableau-style paths; the `Stim compiled` and `Clifft compiled` rows are no-loss controls that precompile once before timing and sample the resulting program inside each timed run. Stim's compiled row uses `Circuit.compile_sampler().sample`, while Clifft's uses `clifft.compile` followed by `clifft.sample`.

Compiled controls are plotted only in `benchmark_no_loss.png`: the LOSS figure remains the three LOSS-capable, non-compiled backends.

The loss set adds `LOSS(0.001)` after every qubit-targeting gate, including noise channels and measurement/reset operations. Annotation instructions and `TICK` are not gates and are left unchanged. Repeat blocks are preserved in the Stim/Clifft loss circuit.

The current ppvm parser on `main` rejects the LOSS spelling. For that one backend, the benchmark uses `GeneralizedTableau.loss_channel` immediately after each operation. Disjoint operations are grouped into one parsed program to avoid Python call overhead; because their targets are disjoint, their loss trials commute and this is equivalent to inserting a LOSS after each gate.

## Installation

Build the local Stim Python extension first. In a virtual environment install the two external packages:

```bash
python -m pip install clifft
python -m pip install \
  'ppvm @ git+https://github.com/QuEraComputing/ppvm.git#subdirectory=ppvm-python'
```

ppvm's native build requires Rust. On machines where its build reports a missing AES target, use the documented workaround:

```bash
RUSTFLAGS='-C target-feature=+aes,+sse2' python -m pip install \
  'ppvm @ git+https://github.com/QuEraComputing/ppvm.git#subdirectory=ppvm-python'
```

Point `PYTHONPATH` at the local Stim build (before any wheel-installed Stim) and run:

```bash
export PYTHONPATH="$PWD/python_build_stim/lib.linux-x86_64-cpython-312:$PYTHONPATH"
python benchmarks/loss_simulator_benchmark.py
```

Use `--self-test` for a distance-five API smoke test, or `--distances 5 7 --repeats 1` for a short run. The `--simulators` and `--variants` options can collect one backend or one circuit set at a time. The default output directory is `benchmarks/results/` and contains:

- `benchmark_results.csv`, including all five raw timings per point;
- `benchmark_no_loss.png`, a log-log plot for the control circuits;
- `benchmark_loss.png`, a log-log plot for the LOSS circuits.

The `--simulators` choices are `Stim`, `Stim compiled`, `ppvm`, `Clifft`, and
`Clifft compiled`. The default keeps the original three LOSS-capable backends;
collect compiled controls as independent slices when desired:

```bash
python benchmarks/loss_simulator_benchmark.py \
  --simulators "Stim compiled" --variants no_loss
python benchmarks/loss_simulator_benchmark.py \
  --simulators "Clifft compiled" --variants no_loss \
  --distances 5 7 9 11 15 20 25 30 40
```

`merge_compiled_results.py` merges those slices into the recorded CSV and
regenerates both figures. Independent slices keep a resource failure in one
compiled backend from discarding the other backend's timings.

## Recorded cloud run

The checked-in CSV and figures were collected on the nine-vCPU cloud machine
used for this branch (AMD EPYC host, 15 GiB RAM), with five one-shot runs per
successful point. The full distance list is retained in the CSV even where a
backend could not complete:

- ppvm's Python bindings reject circuits above 2,048 qubits, so d=40 and d=50
  are recorded as unsupported by that package interface;
- Clifft's non-compiled no-loss set completed through d=40, but the machine
  killed the d=50 preparation for memory pressure;
- Clifft's compiled no-loss sampler also completed through d=40; compiling the
  d=50 program was killed by the machine (exit 137) before timing could start;
- Clifft's LOSS set completed five runs through d=15 (minimum 99.09 s there).
  Larger LOSS points were not measured after that run time became impractical.

The checked-in CSV has 80 rows: 69 measured points and 11 explicit unavailable
points. Unavailable points have an empty `min_seconds` and an explanatory
`error` in the CSV; the plotting code omits those points instead of inventing a
value. `merge_benchmark_results.py` reproduces the original non-compiled
merge, and `merge_compiled_results.py` adds the compiled slices.

References: [Stim circuit generation](https://github.com/quantumlib/Stim/wiki/Stim-vDev-Guide), [ppvm](https://github.com/QuEraComputing/ppvm), and [Clifft leakage/loss sampling](https://clifft.readthedocs.io/en/latest/leakage.html).
