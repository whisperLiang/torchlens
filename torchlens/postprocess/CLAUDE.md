# postprocess/ - Graph Cleanup and Finalization

## What This Does
Transforms raw capture records into user-facing `Trace` state. The current full pipeline
has stable contract steps 0-20: graph traversal, conditional attribution, buffer fixes, loop
detection, labeling, finalization, streaming bundle finalization, and optional out
eviction plus parameter-reference release. Step order is load-bearing.

## Files

| File | Steps | Purpose |
|------|-------|---------|
| `__init__.py` | orchestrator | Full `postprocess()` pipeline and step contracts |
| `_materialize.py` | 0 | Project capture events into raw `Op` state |
| `graph_traversal.py` | 1-4 | Output nodes, output ancestors, orphan removal, distances |
| `ast_branches.py` | 5 support, 11.5 | Conditional AST indexing and source variable names |
| `control_flow.py` | 5-6 | Conditional attribution and buffer dedup |
| `loop_detection.py` | 7 adapter | Adapt Trace state and apply recurrence assignments |
| `loop_grouping_adapter.py` | 7 implementation | Backend-neutral recurrence grouping |
| `labeling.py` | 8-11 | Final labels, renaming, lookup keys, retained layer lists, field ordering |
| `finalization.py` | 12-20 | Undecorate, params, layers, modules, hash, streaming finalization/eviction, ref release |
| `incremental.py` | fastlog enrichment | Adds module paths and param addresses to sparse recordings |

## Step Contracts (M10)

Every step's `PostprocessStepContract` declares its exact op-store COLUMN
write set (`writes`). Under `TORCHLENS_POSTPROCESS_ASSERTIONS` a
zero-cost-when-off audit (class-swap instrumentation on the op row store)
verifies each step writes only declared columns; a new column write fails the
tripwire and widening a declared set is a reviewed contract diff. The audit
covers assignment/deletion AND in-place container mutation (per-step
order-canonical content fingerprints of mutable dict/list/set cells on rows
that existed at step begin — unordered containers hash sorted element
fingerprints, so equal content never false-positives on iteration order).
Whole-row lifecycle is audited separately from column writes: row creation is
a step's produces contract, and row REMOVAL (op husking releases every cell)
is checked against the contract's explicit `removes_rows` sanction — an
unsanctioned removal fails with a precise message, and a sanctioned one
(step 3 orphan removal) does not read as a wall of column writes. Read sets
are not audited (named remaining slice), and mutables nested inside
non-builtin custom objects are the disclosed residual. Transient
build scratch lives in three named per-phase workspaces (`ir/workspaces.py`),
not a flat `TraceBuildState` (deleted in M10): `RawGraphWorkspace` (capture
ingress + steps 0-11, also the backend `finalize_forward_session` ownership
token), `ModuleCaptureWorkspace` (module prep/stack capture, consumed at step
16), and `WrapperRuntimeWorkspace` (wrapper hot path). All three drop at the
transient-state cleanup seam.

## The Ordered Steps

| Step | Function | What |
|------|----------|------|
| pre-0 | `_resolve_output_parent_labels` | Pair each output tensor with its graph parent; late-log returned-but-never-traced buffers as source events |
| 0 | `materialize_from_events` | Rebuild raw `Op` state from sealed capture events |
| 1 | `_add_output_layers` | Create dedicated output nodes (skips unattributable outputs) |
| 2 | `_find_output_ancestors` | Mark nodes connected to model output |
| 3 | `_remove_orphan_nodes` | Drop unconnected raw nodes |
| 4 | `_mark_layer_depths` | Optional input/output distance metadata |
| 5 | `_mark_conditional_branches` | AST/bool/event/edge conditional attribution |
| 6 | `_fix_buffer_layers` | Deduplicate and reconnect buffers |
| 7 | `_detect_and_label_loops` or `_group_by_shared_params` | Recurrent grouping |
| 8 | `_map_raw_labels_to_final_labels` | Build raw-to-final label map |
| 9 | `_log_final_info_for_layers` | Write final layer/module fields |
| 10 | `_rename_model_history_layer_names` | Rename global refs (field reorder removed — scrub order is now deterministic) |
| 11 | `_build_lookup_keys_and_finalize_retained_layers` | Build lookup keys and finalize retained layer lists |
| 11.5 | `_populate_var_names` | Resolve source assignment names through `ast_branches.py` |
| 12 | `_undecorate_all_saved_tensors` | Strip TorchLens attrs from saved tensors |
| 13 | `torch.cuda.empty_cache` | Optional CUDA cache clear |
| 14 | `_log_time_elapsed` | Capture timing |
| 15 | `_finalize_param_logs` | Build and complete ParamLogs |
| 15.5 | `_build_layer_logs` | Build aggregate LayerLogs |
| 16 | `_build_module_logs` | Build ModuleLogs |
| 16.5 | `compute_graph_shape_hash` | Hash graph shape before pass-finished behavior changes |
| 17 | `_set_tracing_finished` | Switch Trace to user-facing behavior |
| 18 | `_finalize_streamed_bundle` | Finalize streamed out bundle |
| 19 | `_evict_streamed_outs` | Optional in-memory out eviction |
| 20 | `release_param_refs` | Drop live parameter references after finalization |

## Step 5: Conditional Attribution
Step 5 builds AST indexes, classifies terminal scalar bools, materializes dense
`conditional_records`, runs a backward flood from branch bools, attributes forward arm edges,
then derives legacy THEN/ELIF/ELSE views. Canonical structures are:
- `Trace.conditional_records`
- `Trace.conditional_arm_entry_edges`
- `Trace.conditional_edge_call_indices`
- `conditional_arm_children` on `Op` and `Layer`

## equivalence_class module suffix
Module containment comes from op-creation stack snapshots, and op creation appends
the canonical module path to `equivalence_class`. No postprocess pass infers or
propagates `modules`.

## Loop Detection
`loop_detection.py` builds a backend-neutral `RecurrenceGroupingGraph` and applies the
assignments returned by `loop_grouping_adapter.py`. The adapter owns the live frontier,
adjacency, parameter-free false-positive guard, grouping, and pass assignment behavior.

## Refresh Projection
There is no standalone `postprocess_fast()` orchestrator. Refresh captures run through the
full `postprocess()` entry point with the established Trace state; Step 0 reads the sealed
`CapturedRunCore.events` snapshot (cloned with independent mutable dict fields) when a
`CaptureSession` is attached, and `RefreshProjector` applies refreshed payloads onto the
existing graph. Saved-output counters are refreshed after retained layers are finalized;
module aggregation remains part of the ordered full pipeline.
