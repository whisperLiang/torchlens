# postprocess/ - Implementation Guide

## Critical Ordering Dependencies
- Step 0 materializes sealed capture events before graph traversal begins.
- Steps 1-3 must run before Step 5 so conditional attribution sees an orphan-free graph.
- Module suffixes must already be present on `equivalence_class` before Step 7.
- Step 7 must precede Step 8 because label mapping uses recurrent groups.
- Step 9 must precede Step 11 because lookup keys depend on finalized module hierarchy info.
- Step 10 must rename global refs before lookup-key finalization.
- Step 11.5 resolves source variable names through `ast_branches.py` after final ops exist.
- Step 15.5 must precede Step 16 because `Module.layers` points to `Layer` keys.
- Step 16.5 computes `graph_shape_hash` before `_set_tracing_finished` changes access behavior.
- Steps 18-19 are only for streamed out bundles.
- Step 20 releases live parameter references after all logs and optional streams are finalized.

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
There is no `postprocess_fast()` orchestrator. `CaptureSession` and `TraceProjector`
prepare refresh events, and the single `postprocess()` entry point preserves the ordered
Step 0-20 contracts. Saved-output summaries are refreshed after Step 11 finalizes the
retained op list.

## Gotchas
- `_build_layer_logs()` merges only selected fields across ops; most fields use first pass.
- `_tracing_finished` is not reset between exhaustive and fast ops.
- Conditional cleanup must update both primary cond-id structures and derived views.
- Changing label formats requires checking visualization, validation, I/O, intervention, and
  bundle supergraph code.
