# Training DAG partitions and SGD FP64 acceptance

GraphPartition remains the sole partition abstraction. The training repair extends
storage-effect proofs inside the existing planner; capture, replay, placement and
repartition keep their existing APIs. Runtime dtype behavior remains caller-owned.

The model acceptance driver now has one optimizer: plain SGD with lr=0.01,
no momentum, no weight decay and no Nesterov acceleration. It uses FP64 by default
and fixed atol=1e-8/rtol=1e-7, with exact integer/boolean comparisons. The independent
full-model reference remains part of each Split check.

The planner repair consists of the ordered source changes `faec92205`, `79ef1ba3a`,
`046f65452` and `a2197bcdd` against baseline `fabc570df`. This cleanup changes the
benchmark and its dependent tests, not production partition or execution code.
The candidate analysis below retains its measured planner-source provenance.

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

## C. SGD correctness and precision

Each Train row runs B=1/2/4/8 across three cuts in A/B/C/A order, independently
for Sequential and noncontiguous DAG partitions. It checks forward output, loss,
input/parameter/connected-boundary gradients, parameter updates, current buffers
and optimizer bindings. Plain SGD has no momentum history; a recut rebinds the
current execution leaves. Missing, failed or unexecuted checks never count as passes.

Eval rows check dynamic batches with the unchanged model. YOLO and RF-DETR Train
use official detection criteria and valid synthetic annotations. ResNet18 and the
small DistilBERT fixture use the shared squared-output objective. BatchNorm,
inplace activations and Train-mode Dropout retain native behavior.

ResNet18 uses 10 classes and 64x64 input. YOLOv8n uses official YAML initialization
and 160x160 input. DistilBERT uses two layers, width 32, four heads and sequence 12.
RF-DETR Nano uses its trusted pretrained checkpoint and 384x384 input with a valid
mask and annotations. Explicit native categorical and bookkeeping dtypes remain
unchanged. Criterion-only floating factories follow output precision and restore
the ambient default afterward.

The fresh SGD-only run on 2026-11-03 passes all 32 model/mode/placement rows.
Every table cell below contains independent Passed Eval and Train rows; each
row validates both Sequential and noncontiguous DAG partitions.

| Model | CPU / CPU | CUDA / CUDA | CPU / CUDA | CUDA / CPU |
| --- | --- | --- | --- | --- |
| ResNet18 | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed |
| YOLOv8n | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed |
| DistilBERT | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed |
| RF-DETR Nano | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed | Eval + Train Passed |

The 16 Train rows contain 128 checked SGD steps across the two partition kinds.
The 16 Eval rows contain 128 checked batch executions. Every kind selects three
distinct partitions and every recut reports zero new captures. YOLO's maximum
parameter-update absolute error across its Train rows is 3.71e-12. Comparisons
use the fixed elementwise absolute-plus-relative budget, never a per-model
exception.

The environment is one Orin GPU, Python 3.10.12, Torch 2.11.0, CUDA 12.6,
torchvision 0.26.0, Ultralytics 8.4.152, Transformers 5.6.2 and RF-DETR 1.8.3.
The seed is 42 and OMP/MKL use one thread. A normalized SHA256 manifest verifies
the benchmark, dependent regressions and every `torchlens/split` source file
against the working tree. The tested benchmark source is committed in
`08ce3fbbf`; the remote snapshot consists of base `c5f853630` plus those four
Python-file replacements, rather than an independently versioned checkout.

## D. Performance scope

The optimizer cleanup does not change the production effect index, candidate
search, cache limits or segment execution. Performance claims require a separate
run with the same source, dtype and inputs; a passing functional matrix is not a
latency or training-convergence measurement.

The effect index is graph-local structural metadata. Live alias checks remain
uncached, execution parameter replicas are not cached by analysis, and recuts use
the established graph without recapture. The bounded analysis and blueprint
caches retain their existing capacities.

## E. Refusals and verification

Unsafe storage-order inversions, unproved alias writes, split native calls,
reverse data edges and unsupported shared mutable state remain refused. Captured
paths and the B=2 probe are finite evidence, not a proof of every future Python path.
One physical GPU does not establish two-GPU support.

| Repository check | Result |
| --- | --- |
| Repository format and Ruff | Passed |
| mypy `torchlens/` | Passed, 613 source files |
| Smoke | 5,745 passed, 178 skipped, 1 xfailed |
| Full local Split suite | 541 passed, 233 skipped |
| Orin benchmark/effect regressions | 65 passed |
| Documentation snippets, UTF-8/MSVC | 64 passed, 2 skipped |
| Skip/size/Ruff governance | 26 passed |

The extra broad regression remains Failed: after 3,380 passes and 165 skips,
`test_validate_forward_pass_dropout_value_drift_does_not_warn_on_retrace` returns
False. The same test passes alone; its complete validation module reproduces
the failure under shuffle seed 835877347 (301 passed, 1 skipped). Neither the
validation implementation nor that test module changes in this cleanup. The
earlier broad attempt stopped on a Hugging Face TLS error; the unchanged Electra
test passes using cached weights. Neither attempt is counted as a passing broad
gate.

Standalone optimizer experiments, duplicate full-model diagnostic sweeps and
their old artifacts are removed. The acceptance workflow retains the independent
full-model reference that attests each Split execution.

Reproduce acceptance with:

```bash
python benchmarks/split_graph_partition.py --model resnet18 --model-mode train --strict --output resnet-sgd.json
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --strict --output yolo-sgd.json
python benchmarks/split_graph_partition.py --model transformer --model-mode train --strict --output distil-sgd.json
python benchmarks/split_graph_partition.py --model rfdetr --weights /trusted/rf-detr-nano.pth --model-mode train --strict --output rfdetr-sgd.json
python -m pytest tests/split -q -rs --tb=short
```
