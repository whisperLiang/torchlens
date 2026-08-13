# postprocess/ - Implementation Guide

## Ordering Is Derived (design-ppdag-v3)
Step order is NOT hand-maintained. `_contracts.py` holds each step's declared
contract (op-column `writes`/`reads`, `placeholder_probes`, `row_effects`,
`trace_state` tokens) plus the two frozen direction authorities:
`LEGACY_STEP_RANK` (every derived edge orients by rank, never by registry
position) and the reason-bearing `PINNED_ORDER_PAIRS` corpus. `_executor.py`
derives the edges (RAW/WW/WAR, two-sided row barriers, token conflicts, the
step-17 barrier), runs rank-keyed Kahn, and refuses import when the derived
order, registry, rank, or corpus disagree (checks R1/R2/K1 + the token
read-before-write analogue). `tests/test_postprocess_dag.py` freezes the
goldens (multi-writer table, probe set, rank), pins the day-1 findings by
name, and holds K2 (every derived producer->consumer pair must be pinned
with a reviewed reason).

Reordering steps therefore requires: editing the named corpus entry (the
semantic review), re-recording the axes matrix
(`tests/support/postprocess_axes.py`), the byte-identity oracles, and a
warnings/exception-order review (those are pinned only by day-1 identity).
The historical prose invariants (1-3 before 5, 7 before 8, 9/10 before 11,
15.5 before 16, 16.5 before 17, 18/19 before 20) are corpus entries now.

## Executor
`postprocess()` keeps the prologue (pre-0 + step-0 materialize block), the
no-layers early exit, and the freeze epilogue; steps 1-20 run through
`_executor.run_pipeline` over `STEP_REGISTRY`. Every step body resolves its
callable through the `torchlens.postprocess` module namespace AT CALL TIME —
monkeypatching a step function on the module still works and still trips the
audit (the seam test proves it). Audit windows are explicit per-step
boundaries: begin -> run -> end-in-finally -> contract check ->
postconditions OUTSIDE any window; no window survives the loop, so the
freeze seam runs unaudited by construction. Step 18's `should_run` IS the
streaming snapshot point (context-writing, never trace-writing); step 19
gates on the snapshot; `should_run` evaluates exactly once per step.

## Step 5 Conditional Branch Detection
- Implementation is in `control_flow.py` with AST support from `ast_branches.py`.
- Primary data is cond-id-aware: `conditional_records`, `conditional_arm_entry_edges`,
  `conditional_edge_call_indices`, `conditional_arm_children`.
- Legacy THEN/ELIF/ELSE fields are derived compatibility views.
- Backward flood is parent-only; do not make it bidirectional.
- Ternary `IfExp` attribution depends on source `col_offset` when arms share a line.

## Module Suffixes
Capture-time op creation appends module-address information to `equivalence_class`
so identical ops in different modules do not get loop-grouped together. Step 7's
`loop_detection.py` seam adapts Trace ops to the live backend-neutral implementation
in `loop_grouping_adapter.py`; do not duplicate grouping logic in the Trace adapter.

## Step 11 Lookup-Key Finalization
`_build_lookup_keys_and_finalize_retained_layers()` applies lookup-key construction while
preserving dependencies needed for replay/intervention when those modes request them.

## Steps 18-20 Streaming and Release
Streaming bundle finalization and eviction live in `finalization.py`. These steps coordinate
with `_io.streaming.BundleStreamWriter` and lazy out refs. Never evict graph-connected
training outs. Step 20 then releases live parameter references.

## Refresh Projection
There is no `postprocess_fast()` orchestrator. Step 0 reads the sealed
`CapturedRunCore.events` snapshot (cloned with independent mutable dict fields) when a
`CaptureSession` is attached, `RefreshProjector` applies refreshed payloads onto the
existing graph, and the single `postprocess()` entry point preserves the ordered
Step 0-20 contracts. Saved-output summaries are refreshed after Step 11 finalizes the
retained op list.

## Gotchas
- `_build_layer_logs()` merges only selected fields across ops; most fields use first pass.
- `_tracing_finished` is not reset between exhaustive and fast ops.
- Conditional cleanup must update both primary cond-id structures and derived views.
- Changing label formats requires checking visualization, validation, I/O, intervention, and
  bundle supergraph code.
