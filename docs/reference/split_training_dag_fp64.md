# Training DAG partitions and FP64 functional acceptance

The subsequent [YOLO CPU/CUDA tolerance revalidation](split_yolo_cross_device_tolerance.md)
uses an explicitly requested, separate comparison profile. This report keeps
its original strict tolerance and historical verdicts.

This report records the training partition repair against
`fabc570dfc5cd6440bb9fe220b652d6b9c858b97`. GraphPartition remains the only partition abstraction;
`prepare_graph`, `analyze`, `materialize`, `repartition` and native replay retain their APIs.
Functional oracles use FP64; runtime and performance workloads keep the caller's dtype.

Strict numerical acceptance remains partial: 29/32 default model rows pass. Three YOLO CUDA-related
Train rows fail the fixed Adam oracle; independent full-native pairs reproduce a comparable
limitation. All measured failures and unreached updates remain visible below.

The full-model matrix uses frozen numerical runtime `046f65452`, built on the two ordered repairs
`faec92205` and `79ef1ba3a`. The normalized 166-file Python manifest used by the isolated hardware
queue has SHA256 `68ffa9ffbf061cfc1a319e5954060057b6aca44810a0a02b8796ebcf0d7c3ab4`. The later
`c6e1a53ff` change registers an optional test dependency; it changes no runtime or numerical oracle.
Raw intermediate records retain their own source identity and are not relabelled as final runs.

The final source-check package has normalized 166-file manifest SHA256
`06759b2d78844d990fb4fc0b3603c53383050fe564894446942b3a430d2c3de1`. This is source guard commit
`a2197bcdd`. Only `_effects.py` and its dedicated test module differ in that manifest: the interval
prefilter changes source-proof cost, with no execution or numerical oracle change. Full-model matrix
results below retain their `046f65452` identity; final-source safety, candidate-equivalence and
regression checks are separate.

## A. Implementation

The graph-local effect index records conservative native storage origins and read-after-write,
write-after-read and write-after-write constraints. The existing planner rejects a required
suffix-before-prefix edge. Supported fresh allocations and their local mutations can move across
independent calls; independent BatchNorm counters and running statistics can move across independent
state. Shared state, unknown aliases and unsupported writes retain typed refusals.

The former `_crossed_operation_is_independent` / `_independent_crossings` per-write scans were
removed. Source collection, candidate requirements and effect validation have separate small
functions; the isolated complexity checks and repository debt ratchet remain unchanged. Structural
metadata is computed once per immutable graph. Live source aliases are rechecked for cached analyses
and execution; no runtime tensor replicas, optimizer state or mutable source snapshots enter the
effect cache.

Current source byte footprints also feed a sorted interval prefilter. A cluster that contains a
written source must have only that logical owner before pairwise checks can be skipped. Unknown
geometry or possible cross-source overlaps go to the shared exact byte relation, including disjoint
strided views. No current address or interval is retained across checks. This removes the common
quadratic scan over independent BatchNorm state without omitting the live alias guard.

An actual native running-mean view exposed a separate boundary hazard: isolating the prefix value
loses its storage connection to a later suffix BatchNorm write. The planner now refuses such
activation frontiers in inference and training; dedicated mutable-buffer slots retain their existing
publication contract. Native embedding gathers and multiplication have exact fresh-allocation
identities, so a real copy is not mistaken for a state view. Existing frozen embedding and
torchvision coverage pins those distinctions.

The internal bounded candidate search combines atomic call/data/effect closure, local write
co-location and representative subgraph unions. It submits every candidate to the existing
`analyze()` method. The public topological candidate method retains its contract. There is no
alternate planner, executor or scheduler.

The runtime's existing state ownership, buffer publication, autograd connection, state migration and
bounded analysis/segment/profile caches remain authoritative. No forward, inplace flag, dropout
probability, optimizer formula or runtime dtype is changed to obtain acceptance.

| Files | Responsibility |
| --- | --- |
| `torchlens/split/_effects.py`, `graph.py` | Cached structural resource origins, native effects and private diagnostics |
| `planner.py` | Existing planner consumes effect order, source disjointness and boundary-state proof |
| `context.py`, `runtime.py`, `errors.py` | Recheck current source assumptions and retain private refusal evidence |
| `candidates.py` | Bounded call/data/effect closure and subgraph-union discovery |
| `benchmarks/split_graph_partition.py` | Strict FP64 model oracles, scoped loss setup, native frontier observation and census |
| `tests/split/conftest.py`, partition/training tests | Scoped functional precision and positive/negative/state/dtype coverage |
| Producer/skip inventories and documentation | Reviewed metadata read sites, optional dependency attribution and reproducible policy |

## B. Reproduced cause and candidate coverage

On the untouched baseline, every ResNet18 and YOLOv8n ancestor-search training candidate first
refuses the BatchNorm counter update `add_1_2:1`: `bn1.num_batches_tracked` and
`model.0.bn.num_batches_tracked`, respectively. This is an unchanged-order proof barrier, not
evidence that either whole DAG has no legal training cut. Independent counter storage is not a
global state barrier.

All counts below are measured on the same Orin environment and use FP64 actual training captures and
CPU/CPU analysis placement. Heterogeneous parameter/state ownership can reject additional members;
execution rows find their three cuts separately for each placement. Numerators require a nonempty
real compute frontier as well as successful analysis; denominators are generated candidates,
including those later excluded. A legal analysis is not a numerical run. The searches have different
budgets and do not enumerate all possible DAG partitions.

| Model | Baseline ancestors | Repaired ancestors | Bounded effects | Repaired unique union |
| --- | ---: | ---: | ---: | ---: |
| ResNet18 | 0 / 88 | 43 / 88 | 106 / 106 | 106 / 151 |
| YOLOv8n | 0 / 276 | 162 / 276 | 345 / 345 | 362 / 476 |
| DistilBERT | 12 / 12 | 12 / 12 | 14 / 14 | 14 / 14 |
| RF-DETR Nano | 1,177 / 1,193 | 1,177 / 1,193 | 369 / 370 | 1,308 / 1,325 |

The repaired generic analyzer accepts 63 ResNet and 219 YOLO ancestor sets. Twenty and 57 of these
are state-counter-only frontiers, respectively. They are excluded from the usable counts above and
the execution driver continues its search after a typed `candidate_compute_frontier_absent` refusal.
The two union sets therefore contain 131/419 candidates with real computation at the frontier, of
which 106/362 pass. Neither model's old 0/88 or 0/276 result establishes the absence of legal
training cuts.

Eight final-source censuses (four models, two searches) reproduce every canonical candidate identity
and accepted/refused verdict from the numerical snapshot. The interval prefilter changes checking
cost, not these measured candidate sets.

The following first-refusal distribution keeps data and native-call checks separate.
Alias/buffer/other columns are zero in these measured families; their negative regressions still
exercise those refusals independently.

| Model / search | Generated | Data valid | Atomic | Mutation | Alias | Buffer | Other | No compute frontier | Usable |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet baseline ancestors | 88 | 88 | 88 | 88 | 0 | 0 | 0 | 0 | 0 |
| ResNet repaired ancestors | 88 | 88 | 88 | 25 | 0 | 0 | 0 | 20 | 43 |
| ResNet bounded effects | 106 | 106 | 106 | 0 | 0 | 0 | 0 | 0 | 106 |
| YOLO baseline ancestors | 276 | 276 | 276 | 276 | 0 | 0 | 0 | 0 | 0 |
| YOLO repaired ancestors | 276 | 276 | 276 | 57 | 0 | 0 | 0 | 57 | 162 |
| YOLO bounded effects | 345 | 345 | 345 | 0 | 0 | 0 | 0 | 0 | 345 |

The remaining ancestor refusals include floating producer/write cuts across a training boundary. The
new search closes those local groups without weakening the boundary restriction. Its one RF-DETR
refusal is `silu_3_366:1`: the selected frontier cuts the LayerNorm-to-permute alias chain before
its inplace write. Immediate write/view co-location does not prove the whole allocation chain stays
local. This is an unproven candidate under the two-segment contract, while 1,308 other measured
RF-DETR partitions pass analysis.

Local Torch 2.13/Ultralytics 8.4.24 yields 347 bounded YOLO candidates; remote Torch
2.11/Ultralytics 8.4.152 yields 345. These are different native capture inventories, not
interchangeable measurements.

## C. Correctness and precision

The isolated model driver defaults to float64 and uses finite-value checks, `atol=1e-8`,
`rtol=1e-7`. Integer and boolean leaves compare exactly; detection classes and indices remain
integer tensors. The scoped functional test fixture restores the preceding default dtype. Explicit
float16/bfloat16/float32/float64 passthrough tests remain separate from high precision equivalence
oracles.

The official YOLO criterion on the newer native package allocates implicit floating temporaries in
the ambient default dtype. Its unconfigured FP64 native reference fails before split execution. The
functional oracle therefore scopes criterion construction/calls to the output's floating dtype and
restores the ambient default afterward. This affects loss-test setup only; forward/replay, explicit
native dtype arguments, categorical targets and operator implementations remain unchanged. Recorded
losses and floating model outputs must be FP64.

Boundary value observation snapshots the independent native forward at the post-prefix epoch,
including prefix-local alias writes. Its gradient observer uses the current autograd output slot
after an inplace view rebase. Missing value/gradient coverage fails. Source-state slots use the
independent state oracle; their snapshots do not masquerade as activation comparisons.

Each training row uses an independent full model, three cuts in A/B/C/A order, B=1/2/4/8 and SGD,
momentum SGD and Adam (12 updates per partition kind). It checks output, loss,
input/parameter/connected-boundary gradients, parameter updates, optimizer history, effective
buffers, graph reuse and current state after recuts. ResNet keeps native training BatchNorm/inplace
ReLU; YOLO keeps its official training structure and detection criterion; DistilBERT keeps active
Dropout; RF-DETR uses its official model, mask and detection annotations/criterion.

The local environment is Python 3.10.18, Torch 2.13.0+cpu, torchvision 0.28, Ultralytics 8.4.24 and
Transformers 5.17. The hardware environment is Python 3.10, Torch 2.11/CUDA 12.6, torchvision 0.26,
Ultralytics 8.4.152, Transformers 5.6.2 and RF-DETR 1.8.3 on one Orin GPU. All model processes use
one Torch CPU thread. CUDA checks set the cuBLAS workspace and disable TF32; their recorded
deterministic-algorithm setting remains explicit.

ResNet18 uses 10 output classes and 64x64 images. YOLOv8n uses its official YAML and 160x160 images.
DistilBERT uses the original benchmark's official `DistilBertModel` configuration (two layers, width
32, four heads, sequence 12, Dropout 0.1 in Train). RF-DETR Nano uses a trusted checkpoint, 384x384
images, a real spatial mask and one valid labelled box per image. The small DistilBERT configuration
and synthetic labelled inputs are disclosed; this is functional graph/state validation rather than a
model-quality experiment.

The RF checkpoint SHA256 is `d8d6b9ee57d4d0ed2b1f305163624712a0532cb7bce0c747317984fc5457440d`.
RF-DETR 1.8.3 reports the missing `_kp_active_mask` key on this checkpoint; its native
initialization is retained and copied into the independent reference.

The complete default matrix contains 32 mode/placement rows: 29 passed and 3 failed. Each row
records sequential and noncontiguous DAG independently. Eval rows execute forward at B=1/2/4/8;
Train rows additionally execute the AD/optimizer/state and A/B/C/A checks. There are 370 completed
updates out of 384 requested; unreached updates in failed rows are not passes.

CPU/CPU controls for ResNet, YOLO and DistilBERT run on the local Torch 2.13 CPU environment.
RF-DETR CPU/CPU and every CUDA-related row run on the Orin Torch 2.11 environment. Comparisons are
independent within each row, not cross-version comparisons between table rows.

| Model | Prefix / suffix | Eval | Train | Completed Train updates |
| --- | --- | --- | --- | ---: |
| ResNet18 | cpu / cpu | Passed | Passed | 24 / 24 |
| ResNet18 | cuda / cuda | Passed | Passed | 24 / 24 |
| ResNet18 | cpu / cuda | Passed | Passed | 24 / 24 |
| ResNet18 | cuda / cpu | Passed | Passed | 24 / 24 |
| YOLOv8n | cpu / cpu | Passed | Passed | 24 / 24 |
| YOLOv8n | cuda / cuda | Passed | Failed | 20 / 24 |
| YOLOv8n | cpu / cuda | Passed | Failed | 18 / 24 |
| YOLOv8n | cuda / cpu | Passed | Failed | 20 / 24 |
| DistilBERT | cpu / cpu | Passed | Passed | 24 / 24 |
| DistilBERT | cuda / cuda | Passed | Passed | 24 / 24 |
| DistilBERT | cpu / cuda | Passed | Passed | 24 / 24 |
| DistilBERT | cuda / cpu | Passed | Passed | 24 / 24 |
| RF-DETR Nano | cpu / cpu | Passed | Passed | 24 / 24 |
| RF-DETR Nano | cuda / cuda | Passed | Passed | 24 / 24 |
| RF-DETR Nano | cpu / cuda | Passed | Passed | 24 / 24 |
| RF-DETR Nano | cuda / cpu | Passed | Passed | 24 / 24 |

The following maximum absolute differences cover completed Train updates, including completed
prefixes of failed rows. Failed attempts are described separately below. The actual finite
elementwise oracle is `abs(actual-reference) <= 1e-8 + 1e-7 * abs(reference)`, so an absolute
maximum above 1e-8 can still pass. Native categorical state retains its dtype and exact-value
checks.

| Model | Forward | Loss | Boundary value |
| --- | ---: | ---: | ---: |
| ResNet18 | 3.233e-11 | 1.324e-12 | 1.180e-10 |
| YOLOv8n | 1.089e-08 | 4.903e-10 | 9.056e-09 |
| DistilBERT | 2.421e-11 | 1.155e-14 | 1.296e-11 |
| RF-DETR Nano | 5.226e-08 | 8.674e-11 | 1.569e-08 |

| Model | Input grad | Parameter grad | Boundary grad |
| --- | ---: | ---: | ---: |
| ResNet18 | 6.161e-14 | 1.872e-11 | 3.759e-14 |
| YOLOv8n | 1.386e-10 | 3.511e-09 | 1.690e-10 |
| DistilBERT | 1.516e-16 | 4.043e-12 | 2.054e-14 |
| RF-DETR Nano | 3.515e-09 | 4.988e-08 | 6.097e-10 |

| Model | Parameter update | Optimizer state | Buffer state |
| --- | ---: | ---: | ---: |
| ResNet18 | 4.725e-10 | 1.954e-12 | 2.856e-11 |
| YOLOv8n | 6.764e-09 | 3.511e-10 | 1.021e-10 |
| DistilBERT | 7.082e-12 | 9.403e-13 | 0.000e+00 |
| RF-DETR Nano | 5.044e-10 | 3.430e-10 | 0.000e+00 |

Completed Train records contain FP64 model outputs and losses, actual frontier value comparisons and
connected boundary gradients. Native optimizer scalar bookkeeping retains its backend dtype;
momentum/variance payloads follow the FP64 parameters. No saved checkpoint activation is used as a
runtime numerical oracle.

### YOLO numerical context

The default nondeterministic same-GPU YOLO Train row fails the fixed strict oracle after ten
completed updates per kind: SGD/Momentum and the first two Adam updates pass, then a few forward
elements exceed the tolerance (maximum about 1.88e-8 in the DAG attempt). This is a **Failed**
numerical row, not a claim that its native kernels lack FP64 support.

An independent pair of full native YOLO models, without TorchLens capture, replay or an AD boundary,
likewise fails the default CUDA comparison after ten updates (about 1.23e-8). Its deterministic
native control completes all twelve updates. The same-GPU Split control with the existing
`--deterministic` flag also completes twelve updates per kind. The CPU-to-CUDA deterministic Split
control still fails; determinism is not a universal cross-device rounding proof. All controls keep
the same FP64 data, native criterion, optimizers and fixed tolerances. Failed default rows remain
separate from configured controls.

The independent native CPU/CUDA pairs also fail after nine completed updates in both directions,
with deterministic algorithms either disabled or enabled. Their first failing forward differences
range from about 1.07e-8 to 1.59e-8. This reproduces the strict multi-step numerical limitation
without any split execution. It does not turn the corresponding Split rows into passes.

| YOLO Train context | Default Split steps per kind | Deterministic Split steps per kind | Native default pair | Native deterministic pair |
| --- | --- | --- | --- | --- |
| CUDA / CUDA | Failed after 10 | Passed, 12 | Failed after 10 | Passed, 12 |
| CPU / CUDA | Failed after 9 | Failed after 9 | Failed after 9 | Failed after 9 |
| CUDA / CPU | Failed after 10 | Failed after 10 | Failed after 9 | Failed after 9 |

Each Split count applies independently to sequential and noncontiguous DAG. All eight SGD/Momentum
updates complete before these Adam failures. Later Adam checks that were not reached remain
unexecuted. The runtime continues to honor native algorithm policy; it adds no gradient rounding,
loss scaling, optimizer compensation or dtype conversion to force the oracle through.

## D. Performance and memory

### Planning cost and effect metadata

These are final-source `a2197bcdd` Orin FP64 Train census measurements in milliseconds. Effect time
covers the existing mutation proof and the new storage-order check. In the effect-search column,
generation also includes lazy structural-index creation; the ancestor search creates that index on
its first analysis instead.

| Model | Ancestor generation | Effect generation | Effect-search analysis median | Effect validation median | Index reachable bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| ResNet18 | 8.590 | 18.584 | 13.237 | 3.631 | 191,562 |
| YOLOv8n | 72.369 | 100.793 | 41.145 | 11.044 | 546,378 |
| DistilBERT | 2.983 | 6.459 | 3.527 | 0.875 | 64,104 |
| RF-DETR Nano | 1,285.041 | 317.786 | 71.227 | 13.704 | 1,366,707 |

The old ResNet/YOLO analyzer stops at its first counter refusal (median total 1.652/4.466 ms). Those
failed early exits are not successful-planning performance baselines. These census measurements
describe current proof/search cost; the matched ordered-cut measurements below isolate the
additional live-source checking cost. Current geometry must be rechecked rather than trusting a
capture-time address or a cached runtime copy.

Index sizes deduplicate the Python objects reachable from the new graph-local index. They exclude
the preexisting graph, native tensor storage, model state and allocator peaks. Shared identifier
objects reached by the index count once; this is not an incremental heap-allocation estimate. No
payload or cross-runtime replica is stored there. Analysis, blueprint and profile cache capacities
retain their existing bounds; repartition does not recapture. Full process/CUDA memory and paired
runtime latencies are measured separately below.

### Repartition measurements

#### Final quiet CPU pair

The explicit baseline archive and final source run adjacently after correctness jobs, with identical
seeds, cut identities, B=2, one thread and native Float32 deployment precision. This benchmark uses
Eval capture with AD probes; it does not claim to measure every mutable-state Train configuration.
Its ResNet18 uses 1,000 classes and 32x32 inputs, distinct from the 10-class 64x64 functional case.
Three identical native-call cuts are compared on the two source revisions.

| Phase | Baseline median / P95 ms | Repaired median / P95 ms | Samples |
| --- | ---: | ---: | ---: |
| Cold analysis | 5.997 / 9.951 | 5.434 / 7.759 | 50 |
| Warm analysis | 2.558 / 4.749 | 1.998 / 2.696 | 50 |
| Cold materialization | 4.843 / 7.801 | 4.837 / 6.309 | 50 |
| Warm materialization | 4.127 / 6.848 | 4.126 / 7.430 | 50 |
| Same-cut repartition | 4.795 / 11.085 | 4.668 / 6.820 | 50 |
| Rotating repartition | 4.623 / 7.162 | 4.629 / 7.195 | 50 |
| Cold repartition | 9.019 / 16.643 | 8.295 / 10.245 | 50 |
| Post-training repartition | 5.565 / 8.933 | 5.660 / 8.026 | 10 |
| Native AD/optimizer step | 53.917 / 189.182 | 54.013 / 94.636 | 10 |

The 100-iteration latest-runtime memory pass retains one measured runtime, three analysis entries
(capacity 32), six blueprints (capacity 64), and one profile entry (capacity 8). Both sides report
98 cache hits and three misses, 909,449 bytes peak / 400,224 bytes retained Python metadata in the
measured recut window, and zero recaptures. The initial effect index is outside that window and is
accounted for separately above. Process peak working set is 630,448,128 to 626,937,856 bytes;
retained RSS is 513,609,728 to 509,718,528 bytes. These are observed distributions and process
measurements, not a universal speedup or memory guarantee.

Earlier CPU timings collected under concurrent correctness load are excluded from the final pair:
unchanged native math slowed similarly. Raw observations remain available separately; only the
isolated matched pair supports the table.

#### Mutable-state Train cuts

A separate adjacent CPU pair captures each model in Train mode, with the functional model dimensions
above, native Float32 deployment precision, seed 73, one thread and exactly the same three
topological cut identities before and after. Each phase has 50 samples. These cuts measure the
common mutable-state path; they are not evidence of a noncontiguous DAG or optimizer equivalence.

| Model / phase | Baseline median / P95 ms | Repaired median / P95 ms |
| --- | ---: | ---: |
| ResNet18 cold analysis | 12.266 / 16.920 | 21.911 / 24.009 |
| ResNet18 warm analysis | 4.133 / 5.340 | 6.866 / 7.851 |
| ResNet18 cold repartition | 25.583 / 28.995 | 30.702 / 34.985 |
| ResNet18 warm repartition | 8.468 / 13.759 | 14.607 / 19.216 |
| YOLOv8n cold analysis | 51.109 / 59.135 | 67.062 / 75.374 |
| YOLOv8n warm analysis | 11.769 / 13.229 | 16.525 / 19.684 |
| YOLOv8n cold repartition | 77.104 / 90.480 | 91.468 / 103.388 |
| YOLOv8n warm repartition | 34.274 / 40.062 | 42.916 / 48.586 |

The added structural and live-state proofs have a measurable cost. The baseline already accepts
these ordered cuts, so this is a successful before/after comparison, unlike its early-refusing DAG
searches. These timings use a different workload/run from the Eval pair and should not be compared
across tables as a precision or mode speed ratio.

Profiling identified the pairwise source guard as the main avoidable cost: 302 checks used 19.720
cumulative seconds and about 38 million object-ID reads. The sorted interval prefilter reduces the
same measured guard to 3.153 seconds under profiling. A single-capture original/sieve/sieve/original
comparison also reduces YOLO warm-repartition medians from 76.988–80.470 to 34.154–40.433 ms. Those
are diagnostic comparisons against the repaired quadratic guard; the table above is the final
matched comparison against the original repository. Current handles and footprints are still
evaluated on every check. There is no address cache, weakened alias refusal or additional runtime
executor.

A separate warmed Train-mode memory run cycles the same three cuts 100 times per model and revision,
retaining only the latest measured runtime. Native forward/backward is outside this metadata window.
The initial effect index is also outside it and is accounted for above.

| Model | Python metadata peak bytes, before → after | Retained bytes, before → after |
| --- | ---: | ---: |
| ResNet18 | 730,563 → 797,291 | 201,090 → 201,570 |
| YOLOv8n | 1,390,721 → 1,469,041 | 486,004 → 486,444 |

Both revisions finish with one measured live runtime, three analysis entries (capacity 32), six
blueprints (capacity 64), one profile (capacity 8), and 103 cache hits / three misses. The
25/50/75/100 snapshots and final GC check retain one runtime. A patched capture entry point raises
if any recut recaptures; all four 100-iteration runs complete with zero recaptures. Temporary
interval records increase peak Python metadata without accumulating runtime replicas.

#### Orin CUDA pair

These adjacent baseline/`046f65452` runs use native Float32, Eval models with AD probes, B=2, one
CPU thread, 50 repeats and 100 memory iterations. Every pair matches model, placement, seed,
dimensions and all three canonical cut identities. They retain their numerical-snapshot source
identity; the final Train-mode prefilter measurements and source-delta validation are separate
above/below. Native AD probes use the performance objective, rather than claiming the
official-criterion Train matrix's workload or a DAG performance comparison.

| Model / phase | Baseline median / P95 ms | Repaired median / P95 ms | Samples |
| --- | ---: | ---: | ---: |
| ResNet18 Cold analysis | 16.854 / 17.199 | 16.923 / 17.440 | 50 |
| ResNet18 Warm analysis | 6.138 / 6.424 | 6.132 / 6.411 | 50 |
| ResNet18 Cold materialization | 14.691 / 14.939 | 14.598 / 14.880 | 50 |
| ResNet18 Warm materialization | 12.623 / 12.732 | 12.441 / 12.702 | 50 |
| ResNet18 Same-cut repartition | 14.536 / 14.823 | 14.483 / 14.710 | 50 |
| ResNet18 Rotating repartition | 14.567 / 15.152 | 14.592 / 15.580 | 50 |
| ResNet18 Cold repartition | 25.716 / 26.112 | 25.803 / 26.195 | 50 |
| ResNet18 Post-training repartition | 17.031 / 34.501 | 17.054 / 39.438 | 10 |
| ResNet18 Native AD/optimizer step | 90.377 / 388.349 | 87.606 / 363.222 | 10 |
| YOLOv8n Cold analysis | 54.985 / 56.026 | 56.149 / 58.978 | 50 |
| YOLOv8n Warm analysis | 18.815 / 19.143 | 19.000 / 19.261 | 50 |
| YOLOv8n Cold materialization | 44.948 / 45.376 | 45.099 / 45.534 | 50 |
| YOLOv8n Warm materialization | 35.600 / 35.880 | 36.114 / 36.526 | 50 |
| YOLOv8n Same-cut repartition | 42.075 / 42.629 | 42.541 / 43.501 | 50 |
| YOLOv8n Rotating repartition | 42.436 / 43.451 | 42.711 / 44.102 | 50 |
| YOLOv8n Cold repartition | 82.062 / 86.232 | 83.614 / 85.687 | 50 |
| YOLOv8n Post-training repartition | 48.443 / 70.852 | 50.913 / 70.412 | 10 |
| YOLOv8n Native AD/optimizer step | 285.304 / 587.048 | 280.230 / 568.089 | 10 |
| DistilBERT Cold analysis | 7.368 / 7.816 | 7.426 / 7.759 | 50 |
| DistilBERT Warm analysis | 1.839 / 1.926 | 1.843 / 1.950 | 50 |
| DistilBERT Cold materialization | 4.685 / 4.796 | 4.669 / 4.766 | 50 |
| DistilBERT Warm materialization | 3.782 / 3.829 | 3.751 / 3.803 | 50 |
| DistilBERT Same-cut repartition | 4.523 / 4.589 | 4.517 / 4.581 | 50 |
| DistilBERT Rotating repartition | 4.581 / 4.740 | 4.575 / 4.736 | 50 |
| DistilBERT Cold repartition | 10.522 / 10.589 | 10.582 / 10.740 | 50 |
| DistilBERT Post-training repartition | 5.722 / 11.146 | 6.546 / 13.685 | 10 |
| DistilBERT Native AD/optimizer step | 50.765 / 162.506 | 61.564 / 165.129 | 10 |
| RF-DETR Nano Cold analysis | 100.969 / 106.312 | 102.755 / 106.815 | 50 |
| RF-DETR Nano Warm analysis | 20.399 / 20.839 | 20.246 / 20.686 | 50 |
| RF-DETR Nano Cold materialization | 63.183 / 63.878 | 62.780 / 63.481 | 50 |
| RF-DETR Nano Warm materialization | 46.518 / 46.841 | 46.069 / 46.459 | 50 |
| RF-DETR Nano Same-cut repartition | 57.288 / 58.290 | 56.753 / 58.060 | 50 |
| RF-DETR Nano Rotating repartition | 58.026 / 115.230 | 58.131 / 116.130 | 50 |
| RF-DETR Nano Cold repartition | 143.739 / 148.811 | 145.988 / 152.148 | 50 |
| RF-DETR Nano Post-training repartition | 72.540 / 125.321 | 74.683 / 502.808 | 10 |
| RF-DETR Nano Native AD/optimizer step | 846.882 / 1079.914 | 837.431 / 1095.647 | 10 |

| Model | Python metadata peak bytes, before → after | Retained bytes, before → after | CUDA peak allocated bytes, before → after |
| --- | ---: | ---: | ---: |
| ResNet18 | 1,574,046 → 1,574,289 | 1,122,410 → 1,122,653 | 307,848,704 → 307,848,704 |
| YOLOv8n | 2,756,585 → 2,756,351 | 1,906,450 → 1,906,216 | 112,363,520 → 112,363,520 |
| DistilBERT | 585,135 → 585,135 | 288,402 → 288,402 | 69,100,544 → 69,100,544 |
| RF-DETR Nano | 3,263,672 → 3,263,609 | 2,038,970 → 2,038,907 | 2,031,294,464 → 2,031,294,464 |

Both revisions retain one measured runtime, 3/32 analyses, 6/64 blueprints and 1/8 profiles, with 98
hits / three misses and zero recaptures for each model. Python metadata values cover the recut
window; the initial graph/effect index is outside it. CUDA peaks are allocator counters for the
separate memory run.

The distributions do not establish a universal speedup. In particular, DistilBERT's ten-sample AD
median rises from 50.765 to 61.564 ms, and RF-DETR's ten-sample post-training recut P95 rises from
125.321 to 502.808 ms. Those observed regressions remain in the table; stronger tail-latency
guarantees would require additional profiling and a longer warmed measurement.

### End-to-end FP64 functional memory

These maxima come from completed Train records across the matrix placements. RSS is sampled at the
recorded step seams; CUDA uses the allocator's peak allocated counter. The processes include the
independent native reference, criterion, gradients, optimizer history and captured graph, so these
are not incremental production-runtime costs or before/after performance comparisons.

| Model | Maximum sampled process RSS bytes | CUDA peak allocated bytes |
| --- | ---: | ---: |
| ResNet18 | 2,880,479,232 | 1,159,326,208 |
| YOLOv8n | 2,231,603,200 | 466,735,616 |
| DistilBERT | 1,460,424,704 | 72,753,152 |
| RF-DETR Nano | 26,217,074,688 | 13,290,335,232 |

Orin uses shared physical memory; RSS and CUDA counters are not additive.

## E. Refusals and remaining limits

### Repository quality checks

| Scope | Verified source | Result |
| --- | --- | --- |
| Complete local Split suite | `a2197bcdd` | 531 passed, 233 hardware/optional skips |
| Complete Orin Split suite | `a2197bcdd` | 607 passed, 157 hardware/optional skips |
| Focused Orin CPU/CUDA safety and workflow | `a2197bcdd` | 144 passed, 6 two-GPU skips |
| Final-source candidate equivalence | `a2197bcdd` | All canonical identities/verdicts match in eight searches |
| Alias geometry, live invalidation and debt gates | `a2197bcdd` | 40 passed, 6 CUDA skips |
| Repository format / Ruff | `a2197bcdd` | Passed |
| mypy `torchlens/` | `a2197bcdd` | Passed, 613 source files |
| Windows smoke | `a2197bcdd` | 5,745 passed, 178 skipped, 1 xfail; 10,559 deselected |
| Windows broad `not rare and not slow` | `046f65452` plus optional-dependency inventory | 15,542 passed, 573 skipped, 3 xfails; 357 deselected |

The broad run covers the API/runtime repair and precision changes. The later same-module interval
prefilter has its own complete Split, safety and repository smoke checks above. The official Windows
script uses UTF-8 and the existing MSVC environment. An earlier broad attempt stopped on a
HuggingFace connection timeout; the identical test and complete broad script passed with a
process-local download proxy. No assertion or skip policy changed to repair that network run.

### Remaining limits

- Storage-order inversions remain unsupported under prefix-then-suffix execution.
- Same-object shared mutable state retains RAW/WAR/WAW order. Distinct overlapping
  sources refuse; observed overlap and unknown geometry have separate diagnostics.
- View, reshape, flatten, permutation, detach, split and chunk propagate possible
  source storage. Operations that may return either a view or a copy retain possible
  aliases. Unknown view/allocation writes and indexing writes gain no exemption.
- Floating training-boundary mutations, split native calls, reverse data edges and
  heterogeneous shared trainable state without gradient consolidation still refuse.
- Captured taken paths and the existing B=2 batch probe are not universal proofs of
  unobserved Python paths. Explicit Generator copies do not prove source independence.
- One physical Orin does not cover two-GPU migration. Unsupported FP64/native kernels
  and unexecuted tests are never promoted to passed results.

Dedicated crapper, dryer and mutator tools are unavailable in this environment. The available Ruff
complexity/debt gates and three targeted manual guard mutations are recorded separately; they do not
constitute a formal CRAP/duplication/mutation campaign. Removed effect-order and live-alias checks
are detected by behavioral regressions. An always-accepting interval prefilter fails three of the
eight geometry cases; that expected mutant failure is separate from the green suite.

The pre-existing [Nightly failure on the
baseline](https://github.com/whisperLiang/torchlens/actions/runs/41443712717) contains
rendering/reference-asset, recursion and export failures. Those results remain distinct from this
change's Split validation. New local failures in complexity and temporarily unresolved report links
were corrected directly; neither the checks nor their expectations were weakened.

The newer [Latest Torch Canary on the same
baseline](https://github.com/whisperLiang/torchlens/actions/runs/41510258134) uses Torch 2.15.0+cpu
and reports 70 failed / 5,723 passed. Its failures include new distributed dispatcher entries
(`wait_tensors`, `gather_into_tensor_`), the `flex_gemm.scaled_mm` identity inventory, sparse
callable resolution and the W&B scalar table. None of its listed failures is a Split test. This
change does not update those unrelated compatibility inventories or treat that canary as passed; the
measured numerical environments remain Torch 2.11 and 2.13.

Reproduce model acceptance with:

```bash
python benchmarks/split_graph_partition.py --model resnet18 --model-mode train --candidate-census --strict --output resnet-train.json
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --candidate-census --strict --output yolo-train.json
python benchmarks/split_graph_partition.py --model transformer --model-mode train --candidate-census --strict --output distil-train.json
python benchmarks/split_graph_partition.py --model rfdetr --weights /trusted/rf-detr-nano.pth --model-mode train --dtype float64 --strict --output rfdetr-train.json
python benchmarks/split_repartition.py --model resnet18 --repeats 50 --output resnet-repartition.json
python -m pytest tests/split -q -rs --tb=short
```

For CPU/CUDA add `--device cpu --prefix-device cpu --suffix-device cuda`; for the reverse direction,
use `--device cuda --prefix-device cuda --suffix-device cpu`; for CUDA/CUDA use
`--device cuda --prefix-device cuda --suffix-device cuda`. Each row runs in a fresh process.
Default strict YOLO CUDA-related Train commands retain their nonzero numerical-failure exit;
the separately labelled same-GPU control adds `--deterministic`.
