# Native split runtime repairs — 2026-11-02

The dynamic split operand, transported mutable-buffer state and device-specific
random mapping defects described in the [causal analysis](split_gpu_error_analysis.md)
have separate repairs. Existing numerical tolerances and refusal boundaries remain.
The final-source full CPU broad suite and GPU Split suite pass. Float32 numerical
equivalence remains configuration-dependent; double-precision success does not
change a failed float32 verdict. Acceptance remains partial for the numerical,
native-kernel and latency limits recorded below.

## Design

### Exact dynamic operands

Output shape inference alone cannot repair a captured `split(batch_size)` call.
The shape compiler now records an exact positional or keyword operand address
with its symbolic extent expression. Native `split` and `split_with_sizes` require
a symbolic receiver axis, complete partitioning into retained call outputs, and
independent witness agreement. The compiler handles scalar equal chunks and
complete positive section vectors, including unequal sections. It preserves axis numbers,
fixed query sizes and unrelated scalar arguments. Scalar remainder cases without
this proof do not receive the new lowering.

The existing B=2 structural/numerical probe remains mandatory. It authorizes
sampled extrapolation, never universal correctness for every batch or hidden
Python branch. A negative regression deliberately changes a scalar branch and
continues to fail the probe. B=1/2/4/8 regressions check values and gradients in
compact and retained-trace graphs, across recuts.

### Persistent state after transport

A native suffix can mutate the destination copy of a boundary buffer. Keeping
only the source storage does not publish that write. The runtime now identifies
the operands actually written by planner-certified native calls, stages owned
snapshots keyed by their existing source identities, and publishes those entries
to all effective segment bindings after suffix execution. Sequential suffix
microbatches use the same publication path.

Publication rebinds storage instead of overwriting an old tensor saved for prefix
autograd. Reused BatchNorm state on both sides therefore retains the correct
running statistics without invalidating backward version counters. Recuts inherit
the effective state. Parameters and optimizer bindings are unaffected by buffer
publication. Direct non-gradient buffer writes across the cut require actual
buffer provenance, direct operand identity and unchanged effect order; custom
dispatch, unproven aliases and unsupported effects still refuse.

### Native generator-device semantics

Equal CPU/CUDA seeds do not imply equal Dropout masks. Recognized native
stochastic calls now execute on the captured generator device with their saved
per-call RNG/autocast context. Inputs and outputs move with ordinary native
copies, preserving autograd edges. Other calls retain their segment placement.
Compact graphs retain only scalar capture-device metadata for this purpose.

This is an execution-placement policy with transfer and compute costs. It does
not invent masks, disable Dropout or substitute an eval reference. Native callable
identity controls the policy; custom functions do not inherit it by name.
Cross-device explicit generators and random `out=` calls refuse without an
independent state/alias contract. Missing capture-device RNG state also refuses.
Coverage is for recognized calls, not arbitrary custom random implementations.

### Numerical controls

The benchmark's `--deterministic` option explicitly requests deterministic native
algorithms and records cuDNN/cuBLAS settings in JSON. Unsupported deterministic
kernels fail. `--dtype float64` supplies a separate precision control; detection
annotations and RF-DETR's official criterion follow the actual logits dtype,
while categorical labels remain integral.

Neither option guarantees bitwise CPU/CUDA equivalence. The original synthetic
AD objective, optimizer epsilon, comparison tolerances (`atol=1e-4`, `rtol=1e-3`)
and strict exit rules remain. Small gradients can make Adam updates sensitive to
rounding even after output and gradient comparisons pass. Large differences after
discrete selection require their own evidence rather than a blanket precision
explanation.

Independent final-source diagnostics retain the native operators and original
strict assertions. In RF-DETR heterogeneous eval at B=2, top-k input scores differ
by at most `5.5313e-5`; two selected indices change, while the smallest adjacent
native score gap is `9.5367e-7`. Applying native CPU/CUDA top-k to identical
scores gives identical indices in this case. The perturbed scores therefore
change a discrete selection; the subsequent box difference reaches `0.2613`.
This reproduces how small floating-point differences can produce large output
differences without active Dropout. The diagnostic uses its own inputs and does
not replace the original acceptance row.

The DistilBERT diagnostic identifies
`core.transformer.layer.0.attention.out_lin.bias`: native gradient maximum
`7.5585e-9`, split maximum `2.3149e-9`, maximum difference `6.6160e-9`, all
32 elements below Adam's unchanged `eps=1e-8`. The parameter-update difference
reaches `3.4435e-4`. These gradients pass the existing absolute comparison but
are sensitive to Adam's normalization at that scale. The diagnostic preserves
the failing strict exit; it does not alter the loss or optimizer.

## Validation provenance

The starting revision is `9bf9d64f8e75c12d2323fcc86621d0dda3665c84`; the earlier
acceptance implementation is `0be60ff1b52f5babba759f5e6ce348fb549fc1c7` and the
performance baseline is `f5f559f2689631af7c1df35fcad421d9383c906f`.
The repaired implementation and benchmark controls are committed through
`2d40405ffb5d4b989346dd89397009a8753d065a` (dynamic operands `ce1005967`,
buffer state `74e9ef9e1`, generator-device semantics `f4e7fbc28`).
Remote verification matches all 608 baseline TorchLens Python files to the exact
baseline archive, and 714 repaired runtime, Split-test and measurement-driver
Python files to the implementation commit, using normalized SHA-256 manifests.
Measurements use frozen source copies and normalized source manifests. Archived
copies have no Git metadata; raw `git_commit: null` fields are retained rather
than rewritten to a later commit. A final extraction moves two pure witness-axis
helpers unchanged by AST comparison, keeping the shape module under the existing
2,000-line ceiling.

Local checks use Windows x86-64, Python 3.10.18 and PyTorch 2.13.0+cpu.
Real GPU checks use Ubuntu 22.04.5, Linux 5.15.185-tegra aarch64, Python 3.10.12,
PyTorch 2.11.0, CUDA 12.6, driver 540.5.0 and one NVIDIA Orin. Model packages are
torchvision 0.26.0, transformers 5.6.2, ultralytics 8.4.152 and rfdetr 1.8.3.
RF-DETR uses the existing trusted official Nano checkpoint. Its loader discloses
one missing `_kp_active_mask` entry and uses the configured default; no substitute
model or replacement checkpoint is used.

The initial repaired snapshot passed 46 focused tests on real CPU/CUDA hardware
with zero skips. Local complete Split passed 502 tests with 227 hardware,
optional-dependency and opt-in skips and zero failures. The file-size gate then exposed module
growth; extracting the existing helpers preserved the ceiling and passed 42
file-size/shape/probe tests. This preliminary complete Split result precedes that
extraction and is not labelled final-source GPU acceptance.

## Real-model evidence

Each passing training partition kind executes B=1/2/4/8, SGD/momentum/Adam and A/B/C/A
recuts: 12 updates with independent native models/optimizers. Comparisons include
outputs, loss, input/parameter/frontier gradients, parameters, optimizer state,
effective buffers and zero recaptures. Sequential results never stand in for DAG
results.

Final-source controls below run after all internal refactors.
The table separates them from the earlier numerical controls below.

| Final-source configuration | Sequential | Noncontiguous DAG |
| --- | --- | --- |
| RF-DETR train float64, CUDA/CUDA | passed, 12 updates | passed, 12 updates |
| RF-DETR train float64, CPU→CUDA | passed, 12 updates | passed, 12 updates |
| RF-DETR train float64, CUDA→CPU | passed, 12 updates | passed, 12 updates |
| DistilBERT train float64, CPU→CUDA | passed, 12 updates | passed, 12 updates |
| DistilBERT train float64, CUDA→CPU | passed, 12 updates | passed, 12 updates |
| RF-DETR eval float64, CPU→CUDA | passed, forward B=1/2/4/8 | passed, forward B=1/2/4/8 |
| RF-DETR eval float64, CUDA→CPU | passed, forward B=1/2/4/8 | passed, forward B=1/2/4/8 |
| DistilBERT train float32 deterministic, CUDA/CUDA | passed, 12 updates | passed, 12 updates |
| ResNet train float32 deterministic, CUDA/CUDA | passed, 12 updates | unsupported, effect order |
| YOLO train float32 deterministic, CUDA/CUDA | passed, 12 updates | unsupported, effect order |
| RF-DETR train float32 deterministic, CUDA/CUDA | failed, native deterministic backward unavailable | failed, native deterministic backward unavailable |

RF-DETR's deterministic control fails in the independent native reference:
PyTorch 2.11.0 has no deterministic `grid_sampler_2d_backward_cuda` implementation.
The run retains strict deterministic mode and zero completed updates. This is an
unavailable native kernel, not a numerical pass or evidence that deterministic
mode repairs RF-DETR's earlier float32 differences.

The eval rows above request forward verification only: backward, optimizer and
repartition checks remain `not_run`. Their double-precision maximum forward
errors are below `8e-13`. They corroborate the ranking-sensitivity diagnosis
without changing the earlier float32 DAG failures.

| Earlier numerical-control snapshot | Sequential | Noncontiguous DAG |
| --- | --- | --- |
| RF-DETR eval float32, CUDA/CUDA | passed | passed |
| RF-DETR eval float32, either heterogeneous direction | passed | failed, box values |
| RF-DETR train float32, CUDA/CUDA | failed, parameter gradients | failed, box values after an update |
| ResNet train float32 deterministic, CPU→CUDA | failed, parameter gradients | unsupported, effect order |
| ResNet train float32 deterministic, CUDA→CPU | failed, input gradients | unsupported, effect order |
| YOLO train float32 deterministic, CPU→CUDA | failed, output values | unsupported, effect order |
| DistilBERT train float32 deterministic, either heterogeneous direction | failed at Adam after 8 updates | failed at Adam after 8 updates |

The earlier ResNet float64 heterogeneous controls passed sequential training in
both directions, and YOLO's CPU→CUDA float64 control passed sequential training.
Their train-mode DAG candidates still refused effect ordering. Final-source
single-GPU deterministic ResNet/YOLO controls corroborate sequential training,
with overall strict status `not_validated` because DAG remains unsupported.
These earlier controls are separately labelled source snapshots, not substitutes
for final-source or float32 heterogeneous passes.

## Final-source quality checks

| Check | Result |
| --- | --- |
| Full Ruff formatting / lint | passed; 1,820 Python files formatted |
| Full mypy | passed; 612 source files |
| Smoke | 5,745 passed, 178 skipped, one expected failure, zero failures; 620.79 s |
| Broad (`not rare and not slow`) | 15,527 passed, 561 skipped, three expected failures, 357 deselected, zero failures; 3,401.11 s, exit 0 |
| Final GPU focused transport/shape/RNG tests | 46 passed, zero skipped |
| Final full GPU Split | 572 passed, 157 skipped, zero failures; 1,334.34 s |

The broad run uses the official Windows compiler/environment setup, cached
official Hugging Face models in offline mode, and seed 20260920 passed only to
top-level pytest. It emits a native process-group teardown warning for two
unwaited collectives; the test process exits zero. JAX, MLX, TensorFlow, Paddle
and tinygrad are absent on both hosts, so their native execution is `not_run`.

Earlier stopped checks remain in the evidence: module-size growth was fixed by
extraction; new deferred lint debt was removed without changing ceilings; and
the historical analysis snippet gained an import, an exact native `3*x`
assertion and registration in the executable documentation suite. No golden
files, test exclusions, comparison tolerances or duration limits were changed.

## Repartition measurements

All 42 measurement processes complete and pass source, sample-count, exit and
runtime-liveness audits: 32 initial Before/After cases, eight reverse-order
controls and two additional memory-policy controls. Each uses a fresh process,
one CPU thread, batch two, the same final measurement driver and synchronized
CUDA boundaries. The initial matrix covers four models on CPU/CPU, CUDA/CUDA,
CPU→CUDA and CUDA→CPU. Environment settings include
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and `NVIDIA_TF32_OVERRIDE=0`.

The analysis/materialization/recut distributions each contain 50 uninstrumented
samples per process; AD and post-AD recut contain ten. The AD timing uses the
driver's eval-mode synthetic objective, so it does not measure active Dropout
device-routing costs or official train-mode detection losses. Phase timing and
tracemalloc are separate instrumented runs. Capture and normalization happen
before recut timing; every 100-recut stress run rejects any recapture.

Values below are median milliseconds, Before→After, except the explicit rotating
P95 column. The retained distributions CSV and raw JSON include every metric's
P95 and every sample. The first two tables preserve the initial process pair;
later controls do not overwrite it.

| Model / placement | Cold / warm analysis | Cold / warm materialize | Same / rotating / cold recut | Rotating P95 | AD step | Post-AD recut |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 cpu→cpu | 16.201→16.104 / 5.599→5.579 | 13.187→13.379 / 10.976→11.226 | 12.999→13.148 / 13.100→13.260 / 23.987→24.182 | 13.460→13.642 | 190.806→191.650 | 15.659→15.896 |
| ResNet18 cuda→cuda | 16.632→16.646 / 6.034→6.009 | 14.268→14.351 / 12.157→12.316 | 14.086→14.290 / 14.218→14.316 / 25.240→25.402 | 14.893→14.907 | 89.598→77.028 | 16.000→17.982 |
| DistilBERT cpu→cpu | 7.183→7.151 / 1.703→1.678 | 4.243→4.311 / 3.343→3.394 | 4.090→4.153 / 4.124→4.189 / 9.979→10.087 | 4.250→4.368 | 38.122→37.672 | 4.939→5.071 |
| DistilBERT cuda→cuda | 7.313→7.386 / 1.840→1.828 | 4.567→4.666 / 3.651→3.736 | 4.386→4.543 / 4.434→4.570 / 10.301→10.553 | 4.612→4.798 | 60.389→62.841 | 5.718→6.493 |
| YOLOv8n cpu→cpu | 53.633→54.061 / 17.429→17.576 | 41.542→41.988 / 32.615→32.977 | 38.636→38.912 / 38.846→39.149 / 78.236→78.734 | 40.249→41.085 | 468.921→473.918 | 44.818→45.068 |
| YOLOv8n cuda→cuda | 55.493→55.855 / 18.805→19.117 | 44.301→45.330 / 35.389→36.109 | 41.690→42.318 / 41.797→42.637 / 81.388→83.027 | 42.680→43.486 | 279.553→254.588 | 48.365→47.898 |
| RF-DETR cpu→cpu | 98.612→99.755 / 18.547→18.648 | 59.860→59.921 / 42.947→42.955 | 53.849→53.949 / 54.781→54.936 / 139.703→139.836 | 112.453→111.484 | 10102.026→10154.696 | 63.169→62.632 |
| RF-DETR cuda→cuda | 100.866→101.512 / 20.240→20.386 | 63.011→63.129 / 46.268→46.508 | 57.285→57.200 / 58.040→58.207 / 143.066→143.911 | 115.069→114.133 | 830.307→842.887 | 74.363→72.745 |

| Model / placement | Cold / warm analysis | Cold / warm materialize | Same / rotating / cold recut | Rotating P95 | AD step | Post-AD recut |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 cpu→cuda | 13.923→13.998 / 3.587→3.545 | 21.082→21.497 / 18.776→19.302 | 10.057→10.282 / 22.184→22.685 / 33.484→34.492 | 26.502→27.297 | 98.715→97.823 | 23.816→24.331 |
| ResNet18 cuda→cpu | 14.321→14.341 / 3.808→3.750 | 23.051→24.657 / 20.707→20.671 | 10.518→10.480 / 26.410→26.388 / 37.620→37.626 | 47.748→71.481 | 194.038→194.108 | 30.543→31.437 |
| DistilBERT cpu→cuda | 6.859→6.918 / 1.360→1.342 | 3.746→3.853 / 2.846→2.940 | 3.595→3.703 / 3.647→3.746 / 9.514→9.719 | 3.805→3.892 | 51.368→51.860 | 4.507→4.738 |
| DistilBERT cuda→cpu | 6.985→6.954 / 1.426→1.421 | 3.827→3.988 / 2.921→3.062 | 3.687→3.818 / 3.727→3.851 / 9.630→9.822 | 3.903→3.994 | 51.631→47.879 | 4.706→4.943 |
| YOLOv8n cpu→cuda | 51.240→51.755 / 14.799→14.835 | 38.429→38.240 / 29.631→29.474 | 34.794→34.700 / 37.156→37.082 / 77.194→77.178 | 40.616→38.234 | 391.428→391.942 | 43.615→42.967 |
| YOLOv8n cuda→cpu | 52.968→52.708 / 16.047→15.933 | 40.358→40.006 / 31.519→31.216 | 36.914→36.434 / 38.855→38.331 / 79.024→78.471 | 44.293→43.642 | 365.351→385.539 | 46.288→45.776 |
| RF-DETR cpu→cuda | 97.804→98.489 / 16.720→16.672 | 56.936→57.346 / 40.217→40.367 | 51.104→51.038 / 52.055→52.211 / 137.880→138.462 | 110.911→109.537 | 9565.009→9586.344 | 76.900→78.531 |
| RF-DETR cuda→cpu | 99.262→100.443 / 18.218→18.390 | 59.092→60.311 / 42.356→43.576 | 53.342→54.237 / 54.211→55.086 / 139.595→141.244 | 110.947→113.686 | 1661.441→1673.457 | 62.487→62.356 |

| Model / placement | State inspection | State binding | State copy | Segment materialization |
| --- | --- | --- | --- | --- |
| ResNet18 cpu→cuda | 0.064→0.065 (n=50/50) | 1.762→1.766 (n=200/200) | 0.124→0.129 (n=1709/1709) | 2.241→2.418 (n=50/50) |
| ResNet18 cuda→cpu | 0.065→0.066 (n=50/50) | 1.959→1.909 (n=200/200) | 0.132→0.131 (n=1709/1709) | 2.199→2.326 (n=50/50) |
| DistilBERT cpu→cuda | 0.054→0.053 (n=50/50) | 0.579→0.569 (n=200/200) | not_run: no copies | 1.077→1.231 (n=50/50) |
| DistilBERT cuda→cpu | 0.052→0.052 (n=50/50) | 0.563→0.565 (n=200/200) | not_run: no copies | 1.091→1.214 (n=50/50) |
| YOLOv8n cpu→cuda | 0.072→0.071 (n=50/50) | 7.117→7.102 (n=200/200) | 0.163→0.150 (n=166/166) | 6.015→6.000 (n=50/50) |
| YOLOv8n cuda→cpu | 0.071→0.068 (n=50/50) | 7.500→7.412 (n=200/200) | 0.151→0.142 (n=166/166) | 6.070→5.990 (n=50/50) |
| RF-DETR cpu→cuda | 0.072→0.072 (n=50/50) | 7.795→6.423 (n=200/200) | not_run: no copies | 23.038→22.692 (n=50/50) |
| RF-DETR cuda→cpu | 0.071→0.072 (n=50/50) | 9.012→6.796 (n=200/200) | not_run: no copies | 22.799→23.123 (n=50/50) |

| Model / placement / metric | Baseline 1 | Final 1 | Final 2 | Baseline 2 |
| --- | --- | --- | --- | --- |
| ResNet18 cuda→cuda, rotating_repartition | 14.218 | 14.316 | 14.548 | 14.264 |
| ResNet18 cuda→cuda, training_step | 89.598 | 77.028 | 84.821 | 77.214 |
| ResNet18 cuda→cuda, post_training_repartition | 16.000 | 17.982 | 18.013 | 17.511 |
| DistilBERT cuda→cuda, rotating_repartition | 4.434 | 4.570 | 4.560 | 4.437 |
| DistilBERT cuda→cuda, training_step | 60.389 | 62.841 | 51.883 | 50.705 |
| DistilBERT cuda→cuda, post_training_repartition | 5.718 | 6.493 | 5.656 | 6.038 |
| YOLOv8n cuda→cpu, rotating_repartition | 38.855 | 38.331 | 38.474 | 38.647 |
| YOLOv8n cuda→cpu, training_step | 365.351 | 385.539 | 387.303 | 385.547 |
| YOLOv8n cuda→cpu, post_training_repartition | 46.288 | 45.776 | 45.892 | 45.968 |

Initial rotating-recut medians change by −1.3% to +3.3%. This is descriptive
evidence of modest recut differences on this Orin, not a uniform speedup claim.
Process variation is visible in the reverse-order controls above. Aggregating
the two processes per revision gives post-AD recut medians of
16.449→17.982 ms for ResNet CUDA/CUDA (+9.3%) and 5.972→6.379 ms for DistilBERT
CUDA/CUDA (+6.8%). Their P95 values are 37.704→32.082 and 13.325→10.941 ms,
respectively. Two process replicates do not exclude a median regression in
those submetrics; latency acceptance remains partial there.

The initial ResNet CUDA→CPU rotating P95 is 47.748→71.481 ms. A fresh reverse-order
pair gives 30.451→32.595 ms; combining the 100 samples per revision gives
37.615→37.675 ms, with medians 26.486→26.455 ms. The initial tail increase is not
stable across these processes. Both initial and repeated distributions are kept.

Phase medians are per instrumented call and must not be summed into the
uninstrumented recut total. One recut can bind or copy many state entries.
Different placement strings do not establish a state transfer: the selected
DistilBERT and RF-DETR cuts produce zero observed state-copy samples during
recuts. Their execution-time boundary transfers are a separate operation.

## Memory and ownership

All 16 initial final-source configurations end their 100-recut latest-only
stress run with one measured runtime, three analysis entries, six blueprint
entries and one profile entry. CUDA controls below distinguish deliberately
retained runtimes from allocator residency. Metadata, RSS and CUDA values are
MiB. The before-release checkpoint already includes GC; release then runs GC
again. CUDA allocated/reserved remain 155.31 / 338.00 MiB in every checkpoint,
with a process peak allocated of 293.59 MiB.

| Policy | Runtime count before→after release | Metadata before→after release | Metadata peak | RSS start→before release→after release | Final analysis / blueprint / profile entries |
| --- | --- | --- | --- | --- | --- |
| latest | 1→1 | 1.020→1.002 | 1.501 | 1655.29→1656.07→1656.07 | 3 / 6 / 1 |
| retain | 100→1 | 13.368→1.002 | 13.461 | 1661.02→1684.17→1682.95 | 3 / 6 / 1 |
| clear | 1→1 | 0.873→0.856 | 1.899 | 1656.93→1657.71→1657.71 | 0 / 6 / 0 |

The held objects release; RSS only partially follows their release. Latest-only
RSS changes over the 16 final-source stress runs range from −41.03 to +2.07 MiB,
with bounded caches and one surviving measured runtime at the final checkpoint. This does not
reproduce sustained per-recut RSS growth. It supports normal object release and
allocator retention over the measured 100 iterations, not an infinite-run leak
proof. The stress loop never calls `empty_cache()`.

## Limits and reproduction

Concrete limits remain: float32 heterogeneous numerical failures and discrete
ranking/Adam sensitivity; native RF-DETR deterministic CUDA backward unavailable;
ResNet/YOLO train-mode DAG candidates refused by effect order; the post-AD latency
submetrics above; two-GPU coverage; and absent optional native backends.
Dedicated crapper, dryer and mutator tools are unavailable, so their runs are
`not_run`. Ruff complexity checks, typing and numerical regressions are not
presented as those tools' output. No scheduler, partition API, secondary FLOPs
engine, schema bump, golden refresh or tolerance relaxation was introduced.

Run each model, mode, placement, dtype and revision in a fresh process, keeping
the JSON and exit code. Representative commands are:

```bash
python benchmarks/split_graph_partition.py --model rfdetr --weights /trusted/rf-detr-nano.pth --model-mode train --device cpu --prefix-device cpu --suffix-device cuda --dtype float64 --strict --output rf-train-cpu-cuda.json
python benchmarks/split_graph_partition.py --model transformer --model-mode train --device cuda --prefix-device cuda --suffix-device cuda --deterministic --strict --output distil-train-cuda.json
python benchmarks/split_repartition.py --source-root /path/to/baseline --model resnet18 --device cuda --repeats 50 --memory-iterations 100 --output before.json
python benchmarks/split_repartition.py --source-root /path/to/repaired --model resnet18 --device cuda --repeats 50 --memory-iterations 100 --output after.json
python benchmarks/split_repartition.py --model resnet18 --device cuda --memory-policy retain --repeats 50 --memory-iterations 100 --output retain.json
python benchmarks/split_repartition.py --model resnet18 --device cuda --memory-policy clear --repeats 50 --memory-iterations 100 --output clear.json
```

Swap device directions for the reverse placement. Use `--memory-policy latest`
for the latest-only control. Preserve dtype-specific verdicts and the original
strict failure behavior. The retained evidence includes manifests, commands,
red/green regressions, initial/repeated timing samples, diagnostics and all test
logs; partial and failed runs remain separate from completed passes.

See [runtime semantics and reproduction commands](split_native_runtime.md), and
the [2026-10-31 acceptance report](split_runtime_acceptance.md) for explicitly
historical CPU measurements. No duration-budget enforcement was reintroduced.
