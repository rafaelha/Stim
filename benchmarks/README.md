# LOSS simulator benchmark

`loss_simulator_benchmark.py` compares the LOSS-enabled Stim branch with QuEra's `ppvm`, Clifft, and Microsoft QDK-EC's QDK simulator. It generates `surface_code:rotated_memory_x` circuits at distances `5, 7, 9, 11, 15, 20, 25, 30, 40, 50`, with `rounds=distance` and all four Stim noise parameters set to `0.001`:

```text
after_clifford_depolarization
before_round_data_depolarization
after_reset_flip_probability
before_measure_flip_probability
```

By default, each point is one shot and the reported value is the minimum of five timed runs. Circuit parsing and model construction happen before timing. The `Stim`, `ppvm`, and `Clifft` rows use their non-compiled/tableau-style paths; the `Stim compiled` and `Clifft compiled` rows are no-loss controls that precompile once before timing and sample the resulting program inside each timed run. Stim's compiled row uses `Circuit.compile_sampler().sample`, while Clifft's uses `clifft.compile` followed by `clifft.sample`.

Use `--shots 1024 --per-sample` to time a batch of 1,024 samples and report seconds per sample. For compiled rows, compilation is performed while constructing the benchmark case, before any timed run; only sampling is included in the five-run minimum. The batch plot and CSV use the `_batch1024` suffix when requested.

Compiled controls are plotted only in `benchmark_no_loss.png`: the LOSS figure contains the four LOSS-capable paths. In both figures, a simulator keeps one color across variants; compiled curves use dashed lines and non-compiled curves use solid lines.

The loss set adds `LOSS(0.001)` after every qubit-targeting gate, including noise channels and measurement/reset operations. Annotation instructions and `TICK` are not gates and are left unchanged. Repeat blocks are preserved in the Stim/Clifft loss circuit.

The current ppvm parser on `main` rejects the LOSS spelling. For that one backend, the benchmark uses `GeneralizedTableau.loss_channel` immediately after each operation. Disjoint operations are grouped into one parsed program to avoid Python call overhead; because their targets are disjoint, their loss trials commute and this is equivalent to inserting a LOSS after each gate.

QDK's experimental Stim frontend spells the same independent loss channel as
`LOSS_ERROR(p)`. The harness translates only the instruction name, calls
`qdk.stim.compile(source, None)` once while constructing the case, and times
only `qdk.simulation.run_qir(..., type="clifford")`. Set
`QDK_PYTHON_TELEMETRY=none` so the benchmark process contains no telemetry
worker.

## Installation

Build the local Stim Python extension first. In a virtual environment install the external packages:

```bash
python -m pip install clifft qdk
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

The `--simulators` choices are `Stim`, `Stim compiled`, `ppvm`, `Clifft`,
`Clifft compiled`, and `QDK-EC`. The default keeps all four LOSS-capable backends;
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

For the batch compiled comparison, collect the two no-loss slices independently:

```bash
python benchmarks/loss_simulator_benchmark.py \
  --simulators "Stim compiled" --variants no_loss \
  --shots 1024 --per-sample --plot-suffix _batch1024 \
  --csv-name benchmark_results_batch1024.csv
python benchmarks/loss_simulator_benchmark.py \
  --simulators "Clifft compiled" --variants no_loss \
  --distances 5 7 9 11 15 20 25 30 40 \
  --shots 1024 --per-sample --plot-suffix _batch1024 \
  --csv-name benchmark_results_batch1024.csv
python benchmarks/merge_batch_compiled_results.py \
  --stim-csv /path/to/stim/benchmark_results_batch1024.csv \
  --clifft-csv /path/to/clifft/benchmark_results_batch1024.csv \
  --output-dir benchmarks/results
```

`merge_batch_compiled_results.py` records the unavailable Clifft d=50 point
when its compilation is killed by the machine, and writes
`benchmark_no_loss_batch1024.png` with log-scaled axes and seconds per sample.

## Timing audit

The regular benchmark's timed region starts after `build_case` has generated
the Stim circuit, parsed the ppvm/Clifft representation, and constructed any
compiled sampler. It therefore does not charge text parsing to a sample. To
check whether one-shot calls are nevertheless a misleading measure of
non-compiled throughput, `benchmark_timing_audit.py` measures four modes:

- `parse_only`: parse an already-created circuit string;
- `parse_plus_sample`: parse that string and simulate one sample;
- `warm_batch`: one pre-parsed API call with `shots=N`;
- `repeat_one_shot`: `N` separate pre-parsed one-shot API calls.

The checked-in `results/timing_audit.csv` is a five-run-minimum audit at d=5
and d=7 for both no-loss and LOSS circuits. It uses the branch-native Stim
polyfill and Clifft 0.7.0, and batches of 1, 4, 16, and 64. The main findings
are:

- Parsing is not the dominant cost of the measured runs. At d=5, parsing is
  36 microseconds for Stim and 194 microseconds for Clifft without LOSS,
  compared with 58 microseconds and 1.91 milliseconds for their warm
  one-shot simulations. With LOSS, the corresponding Clifft values are 434
  microseconds and 28.4 milliseconds.
- One shot is representative of Stim's tableau path: its per-sample time is
  nearly flat as the batch grows because `_run_stim` intentionally creates a
  fresh `TableauSimulator` for every sample.
- One-shot Clifft no-loss calls are not representative of steady-state
  throughput. At d=5, Clifft falls from 1.91 milliseconds/sample at one shot
  to 33 microseconds/sample in one 64-shot call; 64 separate one-shot calls
  remain at about 1.97 milliseconds/sample. At d=7 the corresponding batch
  values are 6.09 milliseconds/sample and 104 microseconds/sample.
- LOSS makes Clifft's per-sample state evolution much more expensive, but
  batching still helps: at d=5 the Clifft LOSS path drops from 28.4 to 16.3
  milliseconds/sample between one and 64 shots; at d=7 it drops from 303 to
  193 milliseconds/sample. Separate one-shot calls do not get that
  amortization.

Consequently, one-shot timings are appropriate for reporting per-shot latency
and for Stim's non-compiled path, but they overstate Clifft non-compiled
steady-state throughput. The final no-loss comparison retains the one-shot
latency curves for the original apples-to-apples benchmark and adds the
compiled batch-1024 controls; the audit documents the batch sensitivity rather
than silently replacing the requested one-shot measurement.

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

The batch-1024 verification has 20 requested compiled points: 10 Stim points
and 9 measured Clifft points, plus an explicit unavailable Clifft d=50 point.
Compilation is excluded from every timed value. On this run Stim compiled was
faster at every measured distance (about 0.58 microseconds/sample versus 4.76
microseconds/sample for Clifft at d=5, and about 264 versus 5,087
microseconds/sample at d=40). The batch Stim slice used the installed Stim
1.16.0 wheel's compiled sampler API because a complete branch-native extension
build was not available in the cloud image; Clifft was 0.7.0.

References: [Stim circuit generation](https://github.com/quantumlib/Stim/wiki/Stim-vDev-Guide), [ppvm](https://github.com/QuEraComputing/ppvm), [Clifft leakage/loss sampling](https://clifft.readthedocs.io/en/latest/leakage.html), and [QDK-EC's loss tutorial](https://github.com/microsoft/qdk-ec/blob/main/deq/documents/tutorial/chapters/qdk-loss-simulation.md).

## Final favorable-settings run

`final_benchmark_report.py` assembles the final runs and renders
`results/final_benchmark_no_loss.png` and `results/final_benchmark_loss.png`.
The no-loss figure includes Stim and Clifft compiled samplers, with compilation
performed before timing and dashed lines for those compiled curves. It also
includes the best affordable batch-1024 settings for ppvm and Clifft's
non-compiled API, plus the one-shot latency controls. The corrected LOSS figure
is a consistent one-shot latency rerun for Stim, ppvm, Clifft, and QDK-EC. It
deliberately omits the old isolated batch-check markers: every LOSS legend item
now corresponds to a connected measured curve.

Every plotted value is the minimum of five timed runs and is normalized to
seconds per sample. Surface-code circuits use `d` rounds and all four Stim
noise parameters equal to `0.001`; LOSS runs insert `LOSS(0.001)` after every
gate (`LOSS_ERROR(0.001)` for QDK's parser). Stim's `TableauSimulator` and the
ppvm LOSS adapter do not expose a native multi-shot sampler; the corrected LOSS
plot therefore reports one-shot latency uniformly instead of mixing isolated
batch checks into the same figure. Runs that were not completed within the
machine's resource budget are left absent and documented in
`results/final_benchmark_results.csv`.
