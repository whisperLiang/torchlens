# Split runtime repairs

The [SGD FP64 acceptance report](split_training_dag_fp64.md) contains current
validation and source provenance. This page describes the retained mechanisms;
optimizer diagnostic sweeps are not part of the acceptance workflow.

## Exact dynamic operands

Output shape inference alone cannot repair a captured `split(batch_size)` call.
The shape compiler records an exact positional or keyword operand address and
its symbolic extent expression. Native split operations require a symbolic
receiver axis, complete partitioning of retained call outputs and independent
witness agreement. Equal chunks and complete positive section vectors, including
unequal sections, preserve fixed query sizes and unrelated scalar arguments.
Unproved remainder cases retain their refusal.

The B=2 structural/numerical probe remains mandatory. It authorizes sampled
extrapolation, not all future Python branches. B=1/2/4/8 regressions check values
and gradients in compact and retained-trace graphs and across recuts.

## Persistent state after transport

Suffix writes to a transported buffer are published to effective bindings through
the existing source-identity ownership machinery. Sequential suffix microbatches
use the same publication path. Publication rebinds storage rather than overwriting
a tensor saved for prefix autograd, preserving BatchNorm running state and backward
version counters. Recuts inherit the effective state.

Writes still require buffer provenance, direct operand identity and unchanged
effect order. Custom dispatch, unproved aliases and unsupported effects refuse.

## Generator-device semantics

Equal CPU/CUDA seeds do not imply equal Dropout masks. Recognized stochastic calls
execute on their captured generator device with the saved per-call RNG/autocast
context; ordinary copies preserve autograd connections to segment placement.
Other calls retain their own placement. Compact graphs retain the scalar capture
metadata needed for this policy.

The policy has transfer and compute costs. Native callable identity controls it;
custom functions receive no exemption by name. Explicit generators, random out
arguments and missing RNG state refuse unless their state/alias contract is proved.

## Training storage effects

The graph-local index caches structural origins and RAW/WAR/WAW dependencies.
Independent storage resources remain separate; overlapping or unknown sources
retain conservative checks. Views propagate possible origins, fresh allocations
need exact native identity proofs, and mutable boundary aliases retain their
publication/refusal rules.

Partition search closes data and required effect dependencies within the one
GraphPartition planner. Prefix-then-suffix execution still refuses a required
suffix-before-prefix edge. Runtime source/alias checks remain live and are not
cached across training steps.

## Acceptance

The isolated driver uses plain SGD(lr=0.01) and fixed numerical checks against
an independent full-model reference. Repartition rebinds current execution
parameters; it does not recapture the graph. The optional deterministic flag
records the requested native algorithm policy; unsupported kernels remain
failures. Runtime dtype and caller-owned optimizer support are unchanged.

[Run the SGD matrix](split_training_dag_fp64.md) for model/device results and
[read the runtime contract](split_native_runtime.md) for placement and ownership.
