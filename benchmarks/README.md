# TorchLens Benchmarks

This directory contains standalone benchmark scripts. Generate result artifacts
locally when measuring a change; historical run outputs are not committed.

## Native split acceptance and repartition

Run a fresh process for each model, mode and placement. Strict acceptance writes
JSON before exiting and returns zero only for `passed`:

```bash
python benchmarks/split_graph_partition.py --model resnet18 --training --strict --output resnet-eval.json
python benchmarks/split_graph_partition.py --model transformer --model-mode train --strict --output distilbert-train.json
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --strict --output yolo-train.json
python benchmarks/split_graph_partition.py --model rfdetr --weights /trusted/rf-detr-nano.pth --training --strict --output rfdetr-eval.json
python benchmarks/split_graph_partition.py --model resnet18 --training --strict --prefix-device cpu --suffix-device cuda:0 --output resnet-heterogeneous.json
```

The report separates sequential/noncontiguous partitions, forward, backward,
optimizer, dynamic batch, repartition, cross-device and actual train-mode checks.
Unsupported or unexecuted checks never count as passing. Eval-mode AD uses a
synthetic objective; actual detection training uses the official criterion and
synthetic annotations. Float32 remains the default. An additional `--dtype float64`
experiment is labelled separately and retains the same tolerances.

Collect synchronized raw repartition samples and separate memory evidence:

```bash
python benchmarks/split_repartition.py --device cuda --repeats 50 --output repartition-cuda.json
python benchmarks/split_repartition.py --model transformer --prefix-device cpu --suffix-device cuda:0 --repeats 50 --output transformer-heterogeneous.json
python benchmarks/split_repartition.py --device cuda --memory-policy retain --memory-iterations 100 --output retain.json
python benchmarks/split_repartition.py --device cuda --memory-policy clear --memory-iterations 100 --output clear.json
```

The default memory policy keeps only the latest runtime. `retain` deliberately
holds every measured runtime until the final GC checkpoint; `clear` invalidates
analysis/profile caches on each iteration. All policies assert that only one
measured runtime survives release and GC. Cache counts, RSS, tracemalloc and
CUDA allocated/reserved/peak memory are separate metrics. Run policies and
Before/After revisions in separate processes using identical environments and
arguments; `--source-root` selects the implementation under measurement.
The `device_transfer` status is based on observed state-copy samples; different
placement strings alone do not establish that a recut copied state.

The facet-indirection benchmark reads a locally generated capture-spine measurement.
Run these in order to create the ignored JSON files under `benchmarks/perf/`:

```bash
python benchmarks/s2_capture_spine_overhead.py
python benchmarks/f1_facet_indirection.py
```

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

`perf_suite.py` drives the performance benchmark matrix. It launches `perf_runner.py` in a fresh
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
intervention primitives. Running it writes `intervention_overhead_results.md` locally.

## Pre-hook provenance overhead

`prehook_provenance_overhead.py` measures exhaustive capture on a 181-module
ReLU chain with no user pre-hooks and with one process-global observational
pre-hook. It prints JSON suitable for comparing revisions:

```bash
PYTHONPATH=$PWD python benchmarks/prehook_provenance_overhead.py
```
