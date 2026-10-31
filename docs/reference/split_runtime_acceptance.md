# Native split runtime acceptance — 2026-10-31

This change repairs mutable-buffer training, makes benchmark exits reflect the
evidence, and replaces the remaining test-only point strings with canonical
`GraphPartition` selections. Acceptance is partial: the supported CPU paths below
pass, some train-mode DAG candidates remain unsupported, and final GPU validation
was deferred at the user's request. An unsupported or unexecuted case is never a
numerical pass. Some CPU latency submetrics remain inconclusive; the measurements
do not establish a uniform speedup or a uniform performance pass.

## Source and environments

The implementation commit is `0be60ff1b52f5babba759f5e6ce348fb549fc1c7`. The follow-up report
commit changes documentation only. Executable sources match the normalized
manifest retained with the measurements.

The baseline is `f5f559f2689631af7c1df35fcad421d9383c906f`. Measurements used
fresh processes against that baseline and the working implementation. Raw JSON
retains the commit actually visible during execution; uncommitted worktree runs
therefore name the baseline, and archived remote source copies have no Git
metadata. Source manifests accompany the retained evidence rather than rewriting
those fields to imply that a later commit was executed.

| Component | Local correctness and CPU measurements | Earlier remote measurements |
| --- | --- | --- |
| OS | Windows build 10.0.26300, x86-64 | Ubuntu 22.04.5 LTS, Linux 5.15.185-tegra, aarch64 |
| CPU | Intel Core Ultra 9 185H, 16 cores / 22 logical CPUs | aarch64; CPU model not collected |
| Python | 3.10.18 | 3.10.12 |
| PyTorch | 2.13.0+cpu | 2.11.0 |
| torchvision | 0.28.0+cpu | 0.26.0 |
| CUDA / driver | Unavailable | CUDA 12.6 / driver 540.5.0 |
| GPU | None | One NVIDIA Orin |
| transformers | 5.17.0 | 5.6.2 |
| ultralytics | 8.4.24 | 8.4.152 |
| rfdetr | Not installed; no local trusted checkpoint | 1.8.3, cached official Nano checkpoint |
| Lint/type tools | Ruff 0.15.4, mypy 2.1.0 | Not used as a substitute for local gates |

The remote measurements were taken during this task, before the final native
observer and write-operand refinements. They describe that working snapshot,
not the final commit. Subsequent remote runs could not be collected after SSH
connectivity failed; further GPU work was then explicitly deferred. No result
from the older 2026-10-28/29 validation runs is counted as a new pass.

## Implementation and invariants

| Surface | Result |
| --- | --- |
| Partition API | `GraphPartition` remains the sole production partition abstraction. |
| Analysis/planning | One planner; no scheduler or alternative compatibility layer added. |
| Profiling | `GraphProfile` continues to use the capture FLOPs engine. |
| Compact training | Covered alongside retained-trace training, frozen buffers and microbatches. |
| Repartition | Reuses the captured graph; stress runs reject any recapture attempt. |
| State ownership | Effective execution tensors are checked through `parameter_bindings()`. |
| Optimizer migration | Caller explicitly copies/reassociates history after recuts; no claim that `plan_state_migration()` performs it. |
| Caches | Existing bounded analysis, profile and immutable segment-blueprint mechanisms retained. |
| Test durations | Removed per-test, family, collection/import and session-exit duration enforcement, including the separate import-duration check. Marker consistency, state isolation, memory guards and the coverage floor remain. |

The reproduced runtime defect was train-mode BatchNorm in the suffix: cloning a
frozen buffer frontier meant native updates changed only a temporary tensor.
`num_batches_tracked` stayed zero while an independent native model reached one.
Training now keeps the live storage for buffer operands actually written by
planner-certified suffix calls. Ordinary activation/index roots remain isolated.
A detached mutable-state snapshot refuses with the existing typed connected-boundary
error, instead of silently losing updates. Sequential microbatches apply the same
rule and match sequential native chunk execution, including every counter update.

Write operands come from the captured argument recipe. A read-only buffer used by
a local activation mutation does not become writable state merely because it is
a parent. A regression covers that distinction. Mutation/alias checks, numerical
tolerances, schema versions and producer goldens were not relaxed or refreshed.

Test fixtures now select canonical node sets directly. The small pure selectors
keep native calls atomic and support an unambiguous node/module anchor. The
original noncontiguous DAG cases and numerical assertions remain. An AST count
audit found no split test file with fewer assertions than the baseline.

## Strict validation method

`--strict` writes complete JSON first and exits zero only for `passed`.
`failed`, `unsupported`, `skipped` and `not_validated` all exit nonzero. Diagnostic
mode permits incomplete evidence but still fails on execution/numerical errors.
Subprocess tests cover all five statuses in both modes and missing trusted weights.
Partial rows retain candidate refusals, completed batches and completed updates.

Each supported partition kind executes B=1/2/4/8 with independent native and split
models/optimizers, SGD, momentum and Adam, and three distinct cuts in A/B/C/A order.
Checks include outputs, loss, input/parameter/connected-frontier gradients, all
optimizer history, effective parameter identities, buffer state and graph reuse.
Reference hooks observe the native forward by semantic call family and captured
output path; they never execute split recipes. Counter writes, augmented assignment
and both `Tensor.split` overloads have direct observer regressions.

`--model-mode train` runs actual training behavior. DistilBERT uses nonzero
Dropout; detection models use their official criteria with valid synthetic box
annotations. Eval-mode autograd instead uses a disclosed synthetic graph objective.
Per-call captured RNG/autocast contexts are restored for the native reference.
Different CPU/CUDA generators can still yield different dropout masks; such
comparisons fail rather than being normalized into a pass. Float32 tolerances stay
`atol=1e-4`, `rtol=1e-3`; `--dtype float64` is a separate experiment.

## Latest CPU real-model results

All executed rows below use float32 and the final observer/runtime implementation.
`passed` in a partition column includes all 12 optimizer/batch steps, gradients, current buffer
state and continuing after recuts. Cross-device execution is `not_run` in these
CPU controls. An absent Dropout operation is not counted as tested RNG behavior.

| Model | Mode | Sequential | Noncontiguous DAG | Forward/backward/optimizer/batches/recuts | Strict result |
| --- | --- | --- | --- | --- | --- |
| ResNet18 | eval + AD | passed | passed | passed for both kinds | passed, exit 0 |
| ResNet18 | train | passed | unsupported | passed for sequential | not_validated, exit 1 |
| DistilBERT | eval + AD | passed | passed | passed for both kinds | passed, exit 0 |
| DistilBERT | train + Dropout | passed | passed | passed for both kinds, including RNG | passed, exit 0 |
| YOLOv8n | eval + AD | passed | passed | passed for both kinds | passed, exit 0 |
| YOLOv8n | train + official loss | passed | unsupported | passed for sequential | not_validated, exit 1 |
| RF-DETR Nano | eval/train | skipped locally | skipped locally | not_run on final local source | skipped without trusted weights, exit 1 |

These are official implementations with explicit benchmark configurations:
ResNet18 has random weights, 10 classes and 64x64 images; DistilBERT uses the
official two-layer, dim-32 architecture with differentiable 12-token embedding
inputs; YOLOv8n uses the official YAML architecture, random weights and 160x160
images. They are not pretrained accuracy or convergence evaluations.

The ResNet18 train DAG search rejected 88 candidates; YOLOv8n rejected 276.
The refusal evidence includes `partition_effect_order_unproven`. This establishes
that this bounded candidate search found no legal cut, not that every possible
DAG partition is impossible. Sequential success is not substituted for DAG success.

The task also executed the real official RF-DETR Nano with its cached checkpoint,
384x384 images and a proper `NestedTensor` spatial mask on the remote CPU. An
intermediate observer completed sequential and genuine noncontiguous eval AD,
including all optimizers, batches and recuts. Its train path exposed the overloaded
split observer defect. The final observer regression passes locally; full RF-DETR
train revalidation remains outstanding. The earlier eval observation is not
promoted to final-source acceptance.
The package verified the cached checkpoint's MD5 and reported one absent
`_kp_active_mask` field left at initialization; checkpoint/model accuracy was not
evaluated by these runtime-equivalence probes.

## Device and backend coverage

The expanded three-branch test checks real noncontiguous A/B/C/A execution,
boundary/gradient transport, B=1/2/4/8 and SGD/momentum/Adam history, in compact and
retained modes. Placement cases are CPU/CPU, CPU/CUDA, CUDA/CPU, one-GPU control
and two distinct GPUs. The local CPU controls run; hardware-dependent cells skip.
The Orin host has only one GPU, so two-GPU migration was never validated.

An intermediate full Orin split run recorded 538 passed, 157 skipped and one
GPU-only legacy-fixture failure. The fixture passed a selector directly to
`plan_state_migration()` and assumed a direct buffer source was suffix-owned.
It now passes a `GraphPartition` and checks the actual prefix ownership while
preserving parameter-copy checks; all nine ownership tests passed on Orin.
A complete final-source GPU rerun is not claimed.

Earlier real-model FP32 CUDA/heterogeneous probes also recorded numerical failures,
including small gradient/update mismatches and cross-device Dropout differences.
Observer failures were separately reproduced and repaired. Neither category is
silently counted as a runtime pass, and final GPU revalidation is `not_run` after
the user's deferral.

| Native runtime | Local | Remote during this task |
| --- | --- | --- |
| PyTorch CPU | Executed | Executed in intermediate measurements |
| PyTorch CUDA | not_run: CPU build | Intermediate execution only; final work deferred |
| JAX | not_run: missing runtime | not_run: missing runtime |
| TensorFlow | not_run: missing runtime | not_run: missing runtime |
| MLX | not_run: missing runtime/platform | not_run: missing runtime/platform |
| Paddle | not_run: missing runtime | not_run: missing runtime |
| tinygrad | not_run: missing runtime | not_run: missing runtime |

Backend-neutral IR checks are not native-backend execution evidence.

## Performance and memory

Use separate processes with one CPU thread, fixed seed/batch and the identical
extended `split_repartition.py` driver for both source roots. CUDA measurements
synchronize at timing boundaries. Cold/warm analysis and materialization,
same/rotating/cold recuts have 50 raw samples each; training and post-training
recuts retain the script's 10-sample cap and are labelled accordingly. Acceptance
driver training time includes the independent oracle; repartition-driver training
time measures split execution.

The repartition ResNet18 configuration remains the original 1,000-class 32x32
model. It must not be compared directly to the 10-class 64x64 acceptance case.
Phase accounting runs separately with exclusive nesting, and memory runs
separately from latency sampling. No test pass/fail deadline is imposed.

Local acceptance telemetry below is the first B=1 sequential SGD step in each
float32 run, not a latency distribution. Prepare time includes capture,
normalization and the shape witness; forward includes runtime transport,
and loss+backward excludes the independent native oracle. RSS includes both
models. GraphProfile was timed only in the eval probes; no train-profile
measurement or final local RF-DETR timing is implied.

| Model/mode | Prepare (ms) | Profile (ms) | Forward (ms) | Loss+backward (ms) | Forward frontier bytes | RSS (MiB) |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet18 eval | 8383.133 | 3.293 | 55.949 | 38.872 | 79,872 | 649.48 |
| ResNet18 train | 10547.827 | not_run | 70.174 | 39.030 | 79,952 | 649.58 |
| DistilBERT eval | 2750.965 | 1.378 | 19.864 | 1.788 | 6,144 | 395.13 |
| DistilBERT train | 4412.977 | not_run | 31.636 | 4.301 | 6,144 | 394.55 |
| YOLOv8n eval | 25746.954 | 11.935 | 151.211 | 47.950 | 221,964 | 554.36 |
| YOLOv8n train | 29836.962 | not_run | 97.309 | 35.863 | 225,760 | 538.14 |
| RF-DETR final local | not_run | not_run | not_run | not_run | not_run | not_run |

The first unpinned local set used fresh processes and n=50 for analysis/
materialization/recuts, n=10 for training/post-training. The table preserves
P50 / P95 milliseconds; these noisy observations are not a speedup claim.

| Model | Metric | Baseline P50/P95 | Final P50/P95 | n per revision |
| --- | --- | --- | --- | --- |
| ResNet18 | Cold analysis | 6.265 / 10.280 | 6.166 / 9.369 | 50 |
| ResNet18 | Warm analysis | 2.382 / 3.308 | 2.811 / 5.658 | 50 |
| ResNet18 | Cold materialize | 5.645 / 10.424 | 6.586 / 11.989 | 50 |
| ResNet18 | Warm materialize | 4.708 / 8.544 | 5.012 / 10.125 | 50 |
| ResNet18 | Same recut | 5.392 / 8.549 | 5.475 / 11.805 | 50 |
| ResNet18 | Rotating recut | 6.045 / 11.968 | 5.461 / 10.353 | 50 |
| ResNet18 | Cold recut | 9.420 / 15.674 | 13.343 / 18.676 | 50 |
| ResNet18 | Training step | 55.664 / 149.212 | 90.405 / 100.889 | 10 |
| ResNet18 | Post-train recut | 6.078 / 8.683 | 10.084 / 13.882 | 10 |
| DistilBERT | Cold analysis | 2.765 / 4.650 | 2.921 / 4.581 | 50 |
| DistilBERT | Warm analysis | 0.722 / 1.394 | 0.670 / 1.481 | 50 |
| DistilBERT | Cold materialize | 2.555 / 3.459 | 2.031 / 3.571 | 50 |
| DistilBERT | Warm materialize | 1.259 / 1.358 | 2.529 / 3.046 | 50 |
| DistilBERT | Same recut | 1.578 / 2.783 | 1.630 / 2.977 | 50 |
| DistilBERT | Rotating recut | 1.551 / 2.280 | 1.632 / 3.123 | 50 |
| DistilBERT | Cold recut | 4.028 / 6.734 | 5.280 / 7.175 | 50 |
| DistilBERT | Training step | 16.224 / 30.060 | 17.573 / 29.813 | 10 |
| DistilBERT | Post-train recut | 3.199 / 4.445 | 2.969 / 6.054 | 10 |
| YOLOv8n | Cold analysis | 22.589 / 34.334 | 20.908 / 30.961 | 50 |
| YOLOv8n | Warm analysis | 6.860 / 9.609 | 11.117 / 15.668 | 50 |
| YOLOv8n | Cold materialize | 24.114 / 34.351 | 19.742 / 33.651 | 50 |
| YOLOv8n | Warm materialize | 19.834 / 29.970 | 13.595 / 20.640 | 50 |
| YOLOv8n | Same recut | 19.795 / 30.660 | 21.269 / 32.489 | 50 |
| YOLOv8n | Rotating recut | 19.894 / 31.787 | 18.637 / 30.407 | 50 |
| YOLOv8n | Cold recut | 39.832 / 57.256 | 34.587 / 50.270 | 50 |
| YOLOv8n | Training step | 171.092 / 241.569 | 180.153 / 222.667 | 10 |
| YOLOv8n | Post-train recut | 22.504 / 167.531 | 29.948 / 138.148 | 10 |

Several unpinned medians changed substantially in both directions, including
ResNet training (+62.4%) and DistilBERT warm materialization (+100.9%).
The raw results are retained. A fresh ABBA control fixes process affinity to
logical CPU 2 on the Intel Core Ultra 9 185H, with one Torch/OMP/MKL thread:
baseline, final, final, baseline for each model. It runs after the full
regression releases the host, with the same driver, shapes, batches and seeds.
Each process has 50 samples per ordinary metric and 10 per training metric.
The following distributions concatenate the two independent process samples
per revision (n=100 or n=20); P95 uses the nearest-rank definition.

| Model | Metric | Baseline P50/P95 | Final P50/P95 | n per revision |
| --- | --- | --- | --- | --- |
| ResNet18 | Cold analysis | 11.722 / 26.335 | 14.137 / 28.502 | 100 |
| ResNet18 | Warm analysis | 6.033 / 15.911 | 5.209 / 13.330 | 100 |
| ResNet18 | Cold materialize | 15.134 / 33.263 | 13.650 / 31.365 | 100 |
| ResNet18 | Warm materialize | 12.075 / 23.277 | 14.196 / 26.048 | 100 |
| ResNet18 | Same recut | 11.639 / 26.322 | 14.885 / 28.379 | 100 |
| ResNet18 | Rotating recut | 14.263 / 26.834 | 13.929 / 26.684 | 100 |
| ResNet18 | Cold recut | 21.364 / 37.649 | 21.806 / 45.887 | 100 |
| ResNet18 | Training step | 137.541 / 195.834 | 137.380 / 200.933 | 20 |
| ResNet18 | Post-train recut | 14.017 / 26.046 | 13.462 / 38.150 | 20 |
| DistilBERT | Cold analysis | 5.009 / 12.135 | 4.946 / 12.481 | 100 |
| DistilBERT | Warm analysis | 1.293 / 3.909 | 1.378 / 3.907 | 100 |
| DistilBERT | Cold materialize | 3.632 / 11.756 | 3.186 / 6.208 | 100 |
| DistilBERT | Warm materialize | 2.903 / 8.695 | 2.339 / 6.093 | 100 |
| DistilBERT | Same recut | 3.294 / 9.183 | 2.900 / 8.195 | 100 |
| DistilBERT | Rotating recut | 3.574 / 9.877 | 3.649 / 9.797 | 100 |
| DistilBERT | Cold recut | 8.008 / 17.089 | 7.772 / 20.244 | 100 |
| DistilBERT | Training step | 22.531 / 41.120 | 23.979 / 42.901 | 20 |
| DistilBERT | Post-train recut | 4.132 / 11.745 | 4.825 / 7.704 | 20 |
| YOLOv8n | Cold analysis | 54.378 / 78.020 | 50.849 / 72.418 | 100 |
| YOLOv8n | Warm analysis | 18.421 / 30.829 | 17.559 / 32.138 | 100 |
| YOLOv8n | Cold materialize | 48.231 / 68.323 | 50.371 / 78.770 | 100 |
| YOLOv8n | Warm materialize | 40.226 / 69.310 | 37.207 / 65.195 | 100 |
| YOLOv8n | Same recut | 44.109 / 70.294 | 46.156 / 74.496 | 100 |
| YOLOv8n | Rotating recut | 46.971 / 75.857 | 46.642 / 81.074 | 100 |
| YOLOv8n | Cold recut | 78.011 / 120.473 | 77.902 / 147.079 | 100 |
| YOLOv8n | Training step | 311.404 / 395.536 | 314.858 / 360.172 | 20 |
| YOLOv8n | Post-train recut | 45.387 / 85.502 | 44.193 / 69.259 | 20 |

Rotating recut medians changed by -2.3% (ResNet), +2.1% (DistilBERT)
and -0.7% (YOLO). The unpinned ResNet training increase did not recur
in the ABBA aggregate (-0.1%); DistilBERT and YOLO training changed by
+6.4% and +1.1%. Other metrics remain variable: ResNet same-cut recut
+27.9%, cold analysis +20.6%, warm materialization +17.6%, and
DistilBERT post-training recut +16.8%. P95 ranges are wide. Two process
replicates per revision do not establish a uniform speedup or exclude
regression in those submetrics. Latency acceptance remains inconclusive
for them; no specific new bottleneck is inferred from these samples.

The ABBA rotating-recut process medians in execution order (ms) retain
the within-revision variation rather than hiding it in the aggregate:

| Model | Baseline 1 | Final 1 | Final 2 | Baseline 2 |
| --- | --- | --- | --- | --- |
| ResNet18 | 15.183 | 13.798 | 14.758 | 14.030 |
| DistilBERT | 3.735 | 3.648 | 3.649 | 3.025 |
| YOLOv8n | 45.958 | 49.438 | 44.989 | 48.726 |

Separate instrumented CPU phase medians below concatenate both ABBA
processes per revision. Values are Before→After ms per call, with n per
revision; one recut can make multiple calls. These do not sum to the
uninstrumented recut total. No state-copy samples were observed on CPU/CPU.

| Model | State inspection | State binding | State copy | Segment materialization |
| --- | --- | --- | --- | --- |
| ResNet18 | 0.023→0.025 (n=100) | 2.148→2.160 (n=400) | not_run: zero copies | 1.500→1.673 (n=100) |
| DistilBERT | 0.019→0.018 (n=100) | 0.497→0.489 (n=400) | not_run: zero copies | 0.793→0.749 (n=100) |
| YOLOv8n | 0.041→0.043 (n=100) | 7.827→7.720 (n=400) | not_run: zero copies | 5.780→5.622 (n=100) |

Final local CPU memory controls use separate 100-recut runs. RSS and metadata
are MiB; cache counts are analysis / blueprint / profile after release and GC.
Runtime counts cover the measured recuts, not every object in the process.

| Policy | Runtime count before→after release | Metadata before→after release | Metadata peak | RSS start→before release→after release | Final caches |
| --- | --- | --- | --- | --- | --- |
| latest | 1→1 | 0.39→0.38 | 0.86 | 485.37→485.94→485.94 | 3 / 6 / 1 |
| retain | 100→1 | 12.40→0.38 | 12.44 | 486.46→508.22→504.96 | 3 / 6 / 1 |
| clear | 1→1 | 0.25→0.23 | 1.27 | 485.65→487.57→487.57 | 0 / 6 / 0 |

All three policies release obsolete measured runtimes. Keeping all 100 holds
about 12.4 MiB of Python metadata until release; RSS only partially follows
object release. Clearing analysis also clears its associated profile cache in
this final local configuration, while six immutable blueprints remain.
CUDA allocated/reserved/peak are unavailable in this CPU build. No repeated
capture or per-iteration `empty_cache()` is used. These observations establish
release and bounded cache counts over 100 recuts, not an infinite-run guarantee.

Earlier synchronized Orin measurements below compare the baseline with the
intermediate working snapshot only. Values are median milliseconds, Before → After.
The seven analysis/materialization/recut columns use n=50; training/post-training
use n=10. They do not demonstrate a 54% GPU gain or final-source GPU acceptance.

| Model/device | Cold/warm analysis | Cold/warm materialize | Same/rotating/cold recut | Train | Post-train recut |
| --- | --- | --- | --- | --- | --- |
| ResNet18 CPU | 15.914→15.944 / 5.484→5.552 | 12.997→13.146 / 10.817→11.019 | 12.734→12.964 / 12.866→13.067 / 23.627→23.751 | 191.736→191.595 | 15.512→15.720 |
| DistilBERT CPU | 7.162→7.135 / 1.704→1.703 | 4.200→4.207 / 3.284→3.289 | 4.039→4.046 / 4.077→4.100 / 9.937→9.943 | 37.388→37.194 | 4.907→4.912 |
| YOLOv8n CPU | 53.589→53.595 / 17.289→17.127 | 41.477→41.461 / 32.634→32.158 | 38.564→38.499 / 38.745→38.723 / 78.114→77.883 | 470.619→467.325 | 44.493→44.562 |
| RF-DETR CPU | 98.327→98.405 / 18.793→18.970 | 59.907→59.769 / 43.224→43.057 | 53.548→53.508 / 54.624→54.593 / 138.639→137.963 | 10124.876→10137.025 | 61.612→61.625 |
| ResNet18 CUDA | 16.666→16.554 / 5.991→6.055 | 14.258→14.363 / 12.188→12.278 | 14.129→14.197 / 14.215→14.348 / 25.196→25.270 | 85.685→76.484 | 16.497→19.278 |
| DistilBERT CUDA | 7.344→7.291 / 1.828→1.810 | 4.526→4.455 / 3.619→3.556 | 4.365→4.326 / 4.423→4.374 / 10.333→10.338 | 60.234→61.509 | 6.278→6.152 |
| YOLOv8n CUDA | 55.149→55.735 / 18.855→18.913 | 44.949→45.053 / 35.737→36.015 | 42.099→42.286 / 42.438→42.319 / 81.461→82.209 | 251.517→271.706 | 47.745→49.898 |
| RF-DETR CUDA | 100.648→100.439 / 20.268→20.531 | 63.008→63.334 / 46.025→46.760 | 57.045→57.410 / 58.019→58.419 / 142.492→142.174 | 830.659→821.875 | 72.569→74.069 |

Rotating recut medians in that snapshot changed by approximately -1.1% to +1.6%.
Training and post-training show larger variation, particularly the ten-sample
YOLO CUDA training measurement (+8.0%) and ResNet CUDA post-training measurement
(+16.9%); these remain disclosed measurements, not evidence of an established
speedup or a diagnosed new bottleneck. Raw distributions include P95 and outliers.

The intermediate heterogeneous runs separately recorded the following phase
medians (ms). Phase values are per instrumented call and must not be summed into
the uninstrumented total; one recut can contain multiple binding/copy calls.

| Model/placement | Rotating recut P50/P95 | State inspection | State binding | State copy | Segment materialization |
| --- | --- | --- | --- | --- | --- |
| ResNet18 CPU→CUDA | 22.304 / 27.274 | 0.066 | 1.757 | 0.133 | 2.217 |
| ResNet18 CUDA→CPU | 26.527 / 31.003 | 0.063 | 1.964 | 0.130 | 2.194 |
| DistilBERT CPU→CUDA | 3.602 / 3.752 | 0.053 | 0.560 | not_run: zero observed copies | 1.087 |
| DistilBERT CUDA→CPU | 3.739 / 3.893 | 0.053 | 0.579 | not_run: zero observed copies | 1.100 |
| YOLOv8n CPU→CUDA | 36.828 / 39.550 | 0.070 | 8.505 | 0.160 | 6.004 |
| YOLOv8n CUDA→CPU | 38.546 / 48.900 | 0.069 | 7.591 | 0.138 | 5.998 |
| RF-DETR CPU→CUDA | 51.529 / 109.105 | 0.070 | 7.888 | not_run: zero observed copies | 22.339 |
| RF-DETR CUDA→CPU | 54.141 / 113.689 | 0.072 | 8.938 | not_run: zero observed copies | 23.424 |

Heterogeneous placement alone does not prove state copied during recuts: selected
cuts can retain the same ownership. Activation/boundary transport occurs during
execution and is a separate operation. The old top-level `device_transfer` label
incorrectly inferred measurement from different placement strings, even for
`cpu`/`cpu:0`. A real CPU-alias subprocess regression reproduced that error. The
final driver labels state-copy measurement only when its instrumentation observed
copies; the table uses those counts even in the older raw reports.

The earlier 100-recut Orin memory controls distinguish retained objects from
allocator residency. Retaining every runtime leaves 100 weak references live;
releasing them and running GC leaves exactly one. CPU metadata falls from about
12.4 MiB to 0.38 MiB, while RSS remains about 831 MiB. CUDA metadata falls from
about 13.0 MiB to 1.0 MiB; allocated CUDA memory stays about 155.3 MiB throughout.
The clear-analysis controls end with zero analysis entries, six blueprint entries
and one measured runtime. Latest-only controls also end with one runtime and
bounded cache counts. No `empty_cache()` occurs inside the stress loop.
This supports object release and allocator retention over the measured 100 recuts;
it is not an unbounded-duration leak proof or a reason to add a memory framework.

## Checks and remaining verification

The initial local split baseline was 460 passed, 187 skipped, zero failed.
The complete final local split run passed 486 tests with 213 skips and zero
failures (417.06s). The final reporting/import/marker/docs/coverage governance
selection passed 86 tests with two real-CUDA skips. The assertion inventory
compared 97 existing split files with the baseline and found no assertion losses.
The focused final observer/buffer/producer checks passed 39 tests with 10 hardware
skips. The Windows quality script passed full Ruff formatting/lint (1,814 files)
and mypy (608 source files). Smoke passed 5,745 tests with 178 skips and one
expected failure (656.16s). The complete broad suite then passed **15,510 tests,
547 skips, three expected failures and zero failures**, with 357 deselections
from the requested `not rare and not slow` selection (4,390.76s, exit 0).

Earlier broad attempts are retained separately. One stopped on ELECTRA's external
Hugging Face connection timeout before TorchLens capture. The actual ELECTRA test
subsequently passed with the already-cached official weights and
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` (14.78s). Another reached 13,490 passes
before an invocation error: putting `--randomly-seed` in `PYTEST_ADDOPTS` leaked
the option into a child deliberately disabling that plugin. Removing that
environment option and passing seed 20260920 only to the top-level command fixed
the invocation; the unchanged isolation module passed all four tests on its own
and in the complete broad rerun. The final run keeps the offline cache settings,
MSVC environment and full selection. No downloaded-model or isolation test was
removed, and incomplete/failed logs are not presented as complete passes.

Dedicated `crapper`, `dryer` and `mutator` tools were unavailable in both tool
metadata and executable discovery. Their runs are `not_run`; passing Ruff,
type checks and numerical tests is not presented as those tools' output.

Remaining limits are concrete: inconclusive CPU latency submetrics; unsupported train DAG candidates for ResNet18 and
YOLOv8n; missing local RF-DETR dependencies/checkpoint and outstanding final train
validation; unresolved FP32 cross-device numerical/RNG comparisons; absent native
optional backends; two-GPU coverage; and final GPU testing deferred by the user.

## Reproduction

Run each placement/mode/model in a fresh process and preserve both exit code and JSON:

```bash
python benchmarks/split_graph_partition.py --model transformer --model-mode train --strict --output distilbert-train.json
python benchmarks/split_graph_partition.py --model rfdetr --weights /trusted/rf-detr-nano.pth --training --strict --output rfdetr-eval.json
python benchmarks/split_graph_partition.py --model rfdetr --weights /trusted/rf-detr-nano.pth --model-mode train --prefix-device cpu --suffix-device cuda:0 --strict --output rfdetr-train-cpu-cuda.json
python benchmarks/split_repartition.py --source-root /path/to/baseline --model resnet18 --device cpu --repeats 50 --memory-iterations 100 --output before.json
python benchmarks/split_repartition.py --source-root /path/to/current --model resnet18 --device cpu --repeats 50 --memory-iterations 100 --output after.json
python benchmarks/split_repartition.py --model resnet18 --device cuda --memory-policy retain --memory-iterations 100 --repeats 50 --output retain-cuda.json
python benchmarks/split_repartition.py --model resnet18 --device cuda --memory-policy clear --memory-iterations 100 --repeats 50 --output clear-cuda.json
python -m pytest tests/split -q -rs --tb=short
python -m pytest tests/ -m "not rare and not slow" --randomly-seed=20260920 -x --tb=short
pwsh -File scripts/check_windows.ps1 -Broad
```

Swap the placement directions for CUDA→CPU; use two real device indices for a
two-GPU run. RF-DETR uses the trusted cached checkpoint and never substitutes a
smaller stand-in model. Keep original tolerances and retain every refusal/failure.

Pass the random seed to the top-level pytest command, not through inherited
`PYTEST_ADDOPTS`. On Windows, load the existing MSVC environment for compile
tests; `check_windows.ps1 -Broad` does that for the complete default gate. Cached
Hugging Face models can use the two offline flags recorded above.
