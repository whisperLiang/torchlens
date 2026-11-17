# TorchLens Benchmarks

This directory contains reproducible benchmark scripts and regression baselines.

The nightly same-runner gate always measures both parent/current (10%) and
fixed-anchor/current (25%) comparisons through `perf_ab`. The anchor is the
latest reachable `v*` release tag. For a fork without a release tag,
`perf_anchor.json` records the reviewed initial full commit SHA and its reason;
it stays fixed across subsequent commits. `scripts/prepare_perf_baselines.py`
checks all three commits and their archive contents before creating either
baseline worktree. CI uploads `perf-baselines.json` with the measured
`perf-ab.json`, recording the selected SHA and why it was selected. Missing
anchor content fails preparation; it never removes the cumulative comparison.
Generated reports stay local and are ignored by Git.

## Performance benchmark suite

`perf_suite.py` launches `perf_runner.py` in a fresh subprocess for each
operation/model/device/pass cell. Its default outputs are `perf_results.json`
and `perf_results.md`; use `--out-json` and `--out-md` to choose other paths.
`perf_baselines/linux-cpu.json` is the canonical comparison baseline consumed by
the performance documentation and `perf_gate.py`.

Typical commands:

```bash
python benchmarks/perf_suite.py --smoke
python benchmarks/perf_suite.py --rerun
```

Supporting files:

- `perf_models.py` builds the benchmark model/input fixtures without importing
  TorchLens in pure raw-forward subprocesses.
- `perf_peers.py` contains peer-tool hook/capture implementations and structured
  import skips.
- `perf_runner.py` executes one timing or memory pass cell and writes JSON.
- `host_label.py` names the machine in every record: the `hostname` field holds
  `TORCHLENS_BENCH_HOST_LABEL` (default `benchmark-host`), never the real hostname, and
  captured output spells the checkout as `<repo>`.

## Capture benchmarks

`capture/` holds the matched-contract suites (A: every arm produces the same output;
B: what each TorchLens operation keeps), the operation fixtures, the real-model panel,
the mech-interp workloads and the "what hooks miss" census, all writing one JSON result
format. See `capture/README.md`.

## Intervention overhead

`intervention_overhead.py` measures TorchLens intervention primitives. Run
`python benchmarks/intervention_overhead.py` to generate a local
`intervention_overhead_results.md`; no historical result is shipped.

## Pre-hook provenance overhead

`prehook_provenance_overhead.py` measures exhaustive capture on a 181-module
ReLU chain with no user pre-hooks and with one process-global observational
pre-hook. It prints JSON suitable for comparing revisions:

```bash
PYTHONPATH=$PWD python benchmarks/prehook_provenance_overhead.py
```

## Live-index re-seen labels

`live_index_append_counters.py` counts what a re-seen raw label costs the
`LiveIndex` (a multi-pass `Recorder` fold, a failed-partial concatenation, an
amendment replace): full edge rebuilds, child-list operations against the old
plus new parent count, and total Python calls per forward, with a digest of the
index state after every pass. Counters gate; wall time is context only:

```bash
python benchmarks/live_index_append_counters.py --out live_index_counters.json
```

## Save-path counters

`save_path_counters.py` counts the work one `trace.save` of the `tinynet` capture
does on the default portable path (saved format 3): record-table write passes and
encoder calls, verifying re-reads of the record tables, `fsync` calls, Python-level
opens (direct opens, and opens routed through `pathlib` counted apart, so the
split is the same on every Python), and the files and bytes written (split into manifest, metadata and blobs).
Byte counts hold each wall-clock timing float (`TIMING_FLOAT_PATHS`) at a fixed
width, since format 3 writes floats by exact repr; every other byte counts as
written, and the raw size is information only. The pinned values
live in `tests/perf_baselines/save_path_counters.json`, independently of host timing
reports, and `tests/test_save_path_counters.py` holds them.
Counters gate; wall time and the profiled call count are context only:

```bash
python benchmarks/save_path_counters.py --out save_counters.json
```
