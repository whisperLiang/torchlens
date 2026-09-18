# TorchLens Benchmarks

This directory contains standalone benchmark scripts and generated benchmark
artifacts.

## Split memory

`split_memory.py` measures RF-DETR-N preparation and representative split replays
in a fresh process, with numerical checks against native outputs. It requires
the already-cached official checkpoint and never downloads weights.

```bash
CUDA_VISIBLE_DEVICES='' python benchmarks/split_memory.py --device cpu
python benchmarks/split_memory.py --device cuda
```

Use `--source-root /path/to/another/checkout` for a same-environment comparison
between revisions supporting the `SplitRuntime.retains_trace` API.
Use `--capture-grad` to measure caller-enabled autograd instead of inference
capture. Default Torch inference is compact; use `--retain-trace` to measure the
complete diagnostic capture on revisions supporting that option. Compare modes
in separate processes with identical inputs and runtime versions; process RSS
includes libraries and allocator caches, not just retained tensor storage.

## Performance benchmark suite

`perf_suite.py` drives the 2026-05-14 performance benchmark matrix described in
`.research/perf-benchmarks_PLAN.md`. It launches `perf_runner.py` in a fresh
subprocess for each operation/model/device/pass cell, writes
`perf_results_2026-05-14.json`, and renders `perf_results_2026-05-14.md`.

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

## Intervention overhead

`intervention_overhead.py` is the earlier focused benchmark for TorchLens
intervention primitives. Its committed output lives in
`intervention_overhead_results.md`.

## Pre-hook provenance overhead

`prehook_provenance_overhead.py` measures exhaustive capture on a 181-module
ReLU chain with no user pre-hooks and with one process-global observational
pre-hook. It prints JSON suitable for comparing revisions:

```bash
PYTHONPATH=$PWD python benchmarks/prehook_provenance_overhead.py
```
