# Native split runtime

Point splits (`before`, `after`, `percent`) retain their ordered-cut semantics.
`GraphPartition(prefix_node_ids=frozenset(...))` selects a dependency-closed set
of canonical compute-node IDs from `runtime.trace_graph.compute_nodes`. Display
labels are not partition IDs. A partition has its own stable semantic identity;
it does not invent a target node. Inputs execute in the prefix and final output
reconstruction executes in the suffix, including empty/all-compute partitions.

Use `runtime.analyze_partition(partition)` to validate and lower without building
segments, then `runtime.materialize(analysis)`. `runtime.partition(partition)`
combines those calls. All reuse the normalized capture and ShapeProgram, and
rebind current effective segment state. An analysis cannot be materialized on
an unrelated captured graph. The historical `SplitPointAnalysis` type describes
both forms of analysis.

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

Point and partition analysis share a 32-entry LRU plan cache within one capture
lineage. The key includes graph and shape identity, partition, backend, features,
validation and placement. Recut children share this metadata cache; independent
captures do not. Cached programs contain no current boundary, optimizer state,
or effective parameter replicas. Every hit checks current state compatibility,
and materialization again inherits the current effective values.

Graph costs have a separate eight-batch bound. `clear_analysis_cache()` clears
both caches for the lineage; `analysis_cache_info()` exposes hit/miss counts,
current size and capacity. Caller mutations to returned boundary dictionaries or
capability reports do not poison later cache hits. Replacing the normalized
graph detaches that runtime's caches from its siblings, so costs from the replacement
cannot contaminate the original capture. Parameter values are never hashed on replay.

### Planning benchmark

`python benchmarks/split_planner.py --writes 250 --repeats 25 --warmup 3`
measures uncached planning for a synthetic 500-node chain with 250 native writes.
Use the same script with `--source-root` pointing at each checkout. It reports
the cold call separately, median, nearest-rank P95 and every measured sample.

On an ARM64 NVIDIA Orin host with Python 3.10.12 and Torch 2.11.0, comparing
`24e82949f` with this refactor gave:

| Uncached planning | Before (ms) | After (ms) |
| --- | ---: | ---: |
| Median | 30.327 | 7.429 |
| P95 | 30.414 | 7.536 |

Boundary schemas and canonical bindings are built together. Write-order checks
precompute the leading prefix and trailing suffix runs once, replacing one
whole-graph scan per write. The write-order portion is O(V + W), previously
O(V × W), for V compute nodes and W writes. This workload measures planning;
it performs no tensor execution and establishes no replay or training speedup.

## Capability evidence and execution profiling

`capability_report.feature_evidence` separates backend support, model preflight,
device availability and numerical evidence. Each level is nullable: `None`
means unexamined or unknown. Device availability does not establish free memory
or kernel support. Numerical evidence for `dynamic_batch` refers only to the
recorded unplaced B=2 forward probe; it does not validate this partition,
another batch, gradients or heterogeneous execution.

`runtime.profile_split(inputs, warmup=3, repeats=10)` measures prefix/suffix
forward phases. Pass `targets=` and optionally `loss_fn=` to a training runtime
to execute and separately time real suffix/prefix backward phases. Each result
contains measured samples, median and nearest-rank P95 in milliseconds.

Profiling uses the existing segment builder with isolated copies of current
effective state and inputs; it restores RNG and never performs optimizer steps.
Custom loss functions must be pure with respect to external state: their Python
closures are caller-owned. Gradient accumulation is limited to the isolated
trial leaves even when an objective references an external parameter.
Materialization timing includes isolation copies. CUDA phases synchronize the
devices in use. Effects whose alias semantics cannot be isolated refuse before
measurement. Process-wide allocator peak counters are left unchanged, so this
API reports peak allocated memory as unknown; use the isolated-process memory
benchmarks for allocated/reserved peaks. Cold model download is outside profiling.

## Boundary layout and payload ownership

`runtime.boundary_layout(batch_size=8, training=True)` returns immutable rows
joining the boundary ABI dictionary key to its canonical graph value ID. Each
row includes shape, dtype, logical bytes, capture `requires_grad`, container
path and alias group. `alias_group=None` means aliasing is unknown; it does not
promise independent storage. Unique values are counted once across consumers.
The layout evaluates the existing shape program and never executes the model.

Forward byte totals exclude transport framing and serialization overhead.
Backward bytes remain unknown for potentially differentiable values because a
future loss may ignore any boundary leaf. Integer/boolean payloads require no
gradient: both layout and partition metrics report zero backward bytes for an
empty frontier or one containing only recognized integer/boolean dtypes.
Floating, complex and unknown dtypes keep the backward byte total unknown.
The Torch training engine creates floating/complex suffix roots even
when capture inputs did not require gradients, so `requires_grad=False` alone
is not evidence of zero backward payload. Captured frozen buffer/parameter state
keeps its non-differentiable contract when suffix roots are created, including
BatchNorm running statistics in full-batch and microbatch training.

`ReplayBoundary.to()` follows the backend's movement semantics; a same-device
Torch move does not copy tensor storage. `detach()` shares storage while cutting
the autograd connection; `clone()` explicitly isolates storage. Training keeps
the necessary clones, including integer/index payloads that a suffix can mutate.
Microbatch training also isolates every chunk. No asynchronous transport or
network protocol is introduced. The authenticated local pickle cache must not
be used as an untrusted network deserialization endpoint.

## Partition safety limits

Training executes the native backend's backward kernels and inherits their
numerical reproducibility limits. A supported plan is not a numerical validation
certificate for a particular loss, dtype and device. Split execution does not
replace native backward kernels or change their determinism settings.

The reusable model check in
[`split_graph_partition.py`](../../benchmarks/split_graph_partition.py) uses
`atol=1e-4, rtol=1e-3`. Only RF-DETR 1.8.3 CUDA float32 **parameter gradients**
use `atol=5e-4`, calibrated against repeated native backward variation for the
benchmark's loss. Forward, input gradients and SGD updates retain the stricter
threshold. Every comparison rejects NaN/Inf, including matching infinities.
This acceptance policy is specific to the benchmark loss; it does not establish
a tolerance for arbitrary rescaled losses or change runtime arithmetic.

The DAG planner accepts a bounded set of native ReLU, SiLU and residual-add
in-place writes when their target is a fresh allocation in the same segment.
Supported fresh producers include native convolutions, linear/add/LayerNorm calls
and read-only BatchNorm; native permutations of those local allocations retain
their proven storage origin. Every write must preserve capture order relative to all
computation across the cut; this prevents an independent alias consumer from
moving across a write. A native `arange(out=new_full(...))` initialization is
also supported when the direct destination and write remain in the same segment,
their captured and symbolic extents/dtypes agree, and both are non-differentiable.
This preserves the initialization used by recent YOLO anchor builders.
Other unproven views, transported values, resizing or generic `out=` writes,
training BatchNorm running-stat writes and unknown custom callables refuse with
structured reasons. The exact internal TorchLens identity placeholder is supported.

Plans using this allocation proof require ordinary Tensor/Parameter operands and
no custom dispatch modes at execution. The runtime checks every native call,
including fresh producers, and refuses overrides with
`partition_mutation_dispatch_unsupported`. Model in-place flags are preserved.
Point-based splits retain their existing replay contract. Profiling currently
refuses all writes because its independent state clones do not certify arbitrary
alias relationships.

Stable `SplitUnsupportedError.context.reason` values include:

| Reason | Meaning |
| --- | --- |
| `partition_node_invalid` | A selection is not a canonical compute-node ID |
| `partition_call_split` | Outputs of one native invocation were separated |
| `partition_reverse_dependency` | Suffix computation feeds the requested prefix |
| `partition_dependency_unresolved` | A recorded data dependency cannot be resolved |
| `partition_mutable_state_unsupported` | A buffer observation cannot be assigned safely |
| `partition_effect_order_unproven` | Mutation or alias ordering lacks effect evidence |
| `partition_mutation_dispatch_unsupported` | Runtime tensor overrides invalidate a fresh-allocation proof |
| `partition_custom_effects_unproven` | Custom callable side effects are not certified |
| `partition_shared_trainable_cross_device` | Tied parameter replicas lack gradient consolidation |

Same-device tied parameters are supported. Heterogeneous training with a shared
parameter on both sides refuses until a gradient consolidation contract exists.
State migration and optimizer reassociation remain caller-visible; TorchLens
does not merge optimizer states or perform an implicit step.

In training and live-state runtimes, captured plain Tensor attributes (for example
a detection head's unregistered `stride`) participate in the same live-state
contract as registered buffers.
Compact handles retain their qualified names and follow attribute replacement;
missing live values refuse instead of using stale capture data. Ownership reports
classify these sources as buffers, including their stable names.

Batch-dependent `repeat`/`tile` counts on fixed state are lowered as repeat
factors, separately from output extents. Fixed repeat counts on dynamic inputs
remain fixed. These recipes retain the shape program's sampled-witness disclosure
and do not certify unseen Python control flow.

## Runnable example

Run `python examples/split_native_partition.py` from an installed checkout. The
example constructs an interleaved two-branch network, selects dependency-ready
native calls from `graph_profile()`, analyzes and materializes a noncontiguous
prefix, inspects layout and migration metadata, and compares two Adam steps
against a full-model reference while recutting between steps. It requires no
downloads or GPU and contains no placement solver or network transport.
