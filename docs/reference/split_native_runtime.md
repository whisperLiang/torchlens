# Native split runtime

`GraphPartition(prefix_node_ids=frozenset(...))` is the sole split description.
Canonical compute-node IDs come from `graph.trace_graph.compute_nodes`; display
labels and module names are not partition IDs. Capture first, choose a set, analyze
it, then materialize it. There is one planner and one backend replay engine.

```python
import torch
from torch import nn
from torchlens.split import GraphPartition, SplitFeatures, prepare_graph

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
x = torch.randn(2, 4)
graph = prepare_graph(model, x, features=SplitFeatures(training=True, retain_trace=False))
partition = GraphPartition(frozenset(
    node.canonical_id for node in graph.trace_graph.compute_nodes[:2]
))
analysis = graph.analyze(partition)  # no executable segments or state replicas
runtime = graph.materialize(analysis)
torch.testing.assert_close(runtime.replay(x), model(x))
new_partition = GraphPartition(frozenset(
    node.canonical_id for node in graph.trace_graph.compute_nodes[:1]
))
migration = runtime.plan_state_migration(new_partition)
new_runtime = runtime.repartition(new_partition)
torch.testing.assert_close(new_runtime.replay(x), model(x))
```

`prepare(model, inputs, SplitRequest(partition, ...))` combines these same stages
when canonical IDs are already known. `partition_candidates()` returns a bounded
linear sequence of native-call-atomic topological prefixes, including empty/all
compute selections. It neither enumerates nor limits all legal DAG partitions.
An analysis belongs to exactly one captured graph; another capture refuses it.
Inputs belong to the prefix and output reconstruction belongs to the suffix.

Every dependency into the prefix must originate in the prefix. Every output of
a native multi-output call must have the same owner. The frontier contains each
crossing value once, even when several suffix operations consume it; direct
input skips and structured output paths participate in this same boundary ABI.
Graph partitions use the `activation` role when no more specific role applies.

Legality is conditional on replay and state safety, not merely on DAG closure.
Unsupported stateful operations must produce a structured `SplitUnsupportedError`.
Partitioning is over the captured taken path; it does not prove unseen Python
control flow. Dynamic batches retain the existing independent batch probe and
shape guards.

## Compact training

Torch supports explicit `SplitFeatures(training=True, retain_trace=False)`.
The training default (`retain_trace=None`) still retains the diagnostic Trace.
Compaction happens after normalization repairs argument dependencies, before
shape compilation and the independent batch probe. The compact graph retains
call targets, argument recipes, live parameter/buffer handles, source constants,
RNG/autocast context and effect evidence. Ordinary historical activations and
full Op/Param/Trace objects are released.

Parameters are not detached or copied by compaction. Each new forward builds
its own autograd graph and existing recut state inheritance remains authoritative.
Retained source tensors with a historical autograd graph refuse with
`compact_training_autograd_source`; use a diagnostic runtime for inspection
instead of silently dropping that history. See [memory and training contracts](split_memory.md).

## Static graph costs

`runtime.graph_profile(batch_size=8)` evaluates retained ShapeProgram metadata;
it does not execute the model or materialize segments. `profile.nodes` uses the
canonical compute-value IDs accepted by `GraphPartition`. Multi-output calls
also expose their shared call identity and complete output set so external
algorithms can keep them atomic.

Costs are theoretical tensor sizes, not measured peak GPU memory or latency.
Unknown dtype, shape or operation cost remains `None` with diagnostics.
Parameter inventories deduplicate tied parameters and retain their aliases.
Multiple consumers do not multiply a tensor's byte size. Analysis metrics
describe the selected frontier at the capture batch; request a different graph
profile batch to evaluate a different static workload.

## Reusing analysis

Partition analyses share a 32-entry LRU plan cache within one capture
lineage. The key includes graph and shape identity, partition, backend, features,
validation and placement. Recut children share this metadata cache; independent
captures do not. Cached programs contain no current boundary, optimizer state,
or effective parameter replicas. Every hit checks current state compatibility,
and materialization inherits current effective values. A combined `repartition()`
reuses its own state inspection once; it never retains state snapshots in the cache.

Torch segment schedules have a separate 64-entry graph-local LRU. Its immutable
blueprints contain only call order, last-use release schedules and state argument
locations. Every materialization creates new SegmentState bindings, execution
scratch and replica pools. It recomputes frozen-state sharing eligibility from
current handles. Blueprints never contain boundary tensors, gradients, optimizer
state or private mutable replicas. Same-device recuts preserve effective tensor
identity; necessary device copies use the existing SegmentState migration path.

Graph costs have a separate eight-entry batch/rule-revision bound. `clear_analysis_cache()` clears
both caches for the lineage; `analysis_cache_info()` exposes hit/miss counts,
current size and capacity. Caller mutations to returned boundary dictionaries or
capability reports do not poison later cache hits. Replacing the normalized
graph detaches that runtime's caches from its siblings, so costs from the replacement
cannot contaminate the original capture. Parameter values are never hashed on replay.

### Repartition measurements

Run `python benchmarks/split_repartition.py --repeats 25 --output result.json`.
Uninstrumented samples report median, nearest-rank P95 and every sample for cold
and warm analysis/materialization, same-partition and rotating recuts, and recuts
after training. A separate instrumented pass breaks down validation, frontier,
lowering, capabilities, materialization, binding, inheritance and device transfer.
A separate memory pass reports Python metadata peak/retained bytes; these are not
tensor allocator totals. CUDA measurements require a CUDA-enabled installation.

`python benchmarks/split_runtime_overhead.py --repeats 20` measures execution and
candidate-analysis overhead. `benchmarks/split_training_compact.py` measures
retention modes in separate processes. `benchmarks/split_planner.py` isolates
uncached synthetic planning. Each benchmark records its configuration and measurements
when run; generated result files are not part of the source tree.

## FLOPs authority

GraphProfile calls `torchlens.capture.flops.compute_forward_flops()` and
`compute_backward_flops()` directly. ReplayValueRef arguments resolve to pure
shape/dtype metadata; LiteralTensor arguments contribute shape only, and literal
scalars/containers retain their call positions. ShapeProgram evaluates the requested
batch and rewrites scalar shape recipes. No tensors are allocated, captured,
replayed or read for their values. Required unavailable metadata stays unknown.
There is no second formula table or registration API.

`NodeCostFeatures.forward_flops` and `backward_flops` distinguish forward work from
the capture module's heuristic backward multiplier (`backward_flops_estimated=True`).
Numeric capture-time costs are retained as `captured_forward_flops` and
`captured_backward_flops`, including compact graphs. A multi-output native call is
charged once, on its first row. Zero-FLOPs memory operations still expose tensor
bytes. `register_op_rule()` changes invalidate cached profiles/analysis metrics via
the capture engine's rule revision. Rules requiring tensor payloads remain unknown.

## Effect safety

Known native local writes can be certified through their fresh allocation and
storage-preserving views. Independent branches may cross a local write only if
neither allocation nor its dependent aliases cross the other branch. Out= anchor
initialization retains its non-resizing shape proof and strict ordering barrier.
Training BatchNorm is allowed when its state mutations remain in original effect
order; unknown or reordered state writes refuse. Positional functional inplace
arguments, custom targets and dispatch overrides are included in this analysis.

A sequential same-device inference cut can preserve native aliases through detached
boundary leaves. That does not license mutating floating autograd leaves:
`run_training_prefix()` refuses such inference-only plans with
`partition_training_boundary_write`. Training preparation refuses unproven writes
at analysis time. Cross-device shared trainable parameters still require a gradient
merge contract and refuse when the runtime cannot consolidate them.

For connected Torch training, boundary buffers consumed by planner-certified
suffix writes remain bound to current state. BatchNorm running means, variances
and counters therefore update their owners in both full-batch and sequential
microbatch execution. Ordinary activation/index roots still receive isolated
copies. Mutable buffer training requires `run_training_prefix()`; a detached or
cached state snapshot refuses instead of silently losing updates. Repartition
inherits those updated values. BatchNorm microbatches follow native sequential
chunk statistics, which differ from a single full-batch BatchNorm call.

## Real-model validation

Run each command in a fresh process:

```bash
python benchmarks/split_graph_partition.py --model resnet18 --training --strict --output resnet18.json
python benchmarks/split_graph_partition.py --model transformer --model-mode train --strict --output distilbert-train.json
python benchmarks/split_graph_partition.py --model yolov8 --model-mode train --strict --output yolov8n-train.json
python benchmarks/split_graph_partition.py --model rfdetr --training --strict --weights /trusted/checkpoint.pth --output rfdetr.json
```

Sequential and truly noncontiguous DAG results are independent; no fallback turns
a sequential pass into a DAG pass. Candidate generation checks the unchanged
captured order, native-call closure, dependency closure and a real compute frontier.
The script records every refusal and requires three different partitions for its
A/B/C/A recut experiment. Every B=1/2/4/8 executes explicitly.

The native reference is a separate model/optimizer. The script compares finite
outputs, loss, input/parameter/connected-boundary gradients, buffers, parameter
updates and momentum/Adam state. Native observation hooks align semantic call
families in captured order and use captured output paths, including overloaded
multi-output calls; missing gradient coverage fails. Eval mode
permits gradients while controlling dropout/running-state changes. Tied state,
train-mode BatchNorm, microbatches, cache isolation and device migration have
additional tests under `tests/split/`. The split model is captured once plus its
independent B=2 shape witness; replay/recuts add zero captures.

`--model-mode eval` remains the default; `--training` adds AD/optimizer probes to
that eval graph. `--model-mode train` runs the native training path and enables
training checks automatically. DistilBERT then uses nonzero Dropout. Detection
training uses official losses with synthetic labelled boxes; eval AD keeps its
separate synthetic scalar objective. The native reference uses the same captured
per-call RNG contexts. Different CPU/CUDA generators can still produce different
dropout masks; the benchmark discloses the resulting failure.

`--prefix-device cpu --suffix-device cuda:0` uses `PlacementPlan.across()` and
actual boundary/gradient transport. Reverse placement and distinct CUDA device
indices use the same path. Comparisons explicitly move values to a common device.
Optimizers bind `parameter_bindings()` and explicitly reassociate their complete
state after each recut; `plan_state_migration()` alone does not migrate history.

`--strict` exits zero only for `passed`, after writing complete JSON. `failed`,
`unsupported`, `skipped` and `not_validated` remain distinct, with per-partition
attempts and per-capability statuses. Unrequested capabilities show `not_run`.
Diagnostic mode permits incomplete reports but still exits nonzero on failures.
Use `--dtype float64` for a separately labelled precision experiment; it does not
turn a failing float32 experiment into a pass or change tolerances.

RF-DETR is skipped without both the package and a trusted local checkpoint. A failed,
skipped or unsupported row is never reported as a numerical pass.

Repartition measurements also accept `--model`, `--weights`, `--prefix-device`
and `--suffix-device`. CUDA work is synchronized at measurement boundaries.
Run `--memory-policy latest`, `retain`, and `clear` in separate processes, each
with `--memory-iterations 100`, to distinguish live runtime ownership from Python
and CUDA allocator retention. Reports include RSS, tracemalloc, CUDA allocated,
reserved and peak bytes, weak-reference liveness, cache counts and GC checkpoints.
The stress test never calls `empty_cache()` in its loop.
