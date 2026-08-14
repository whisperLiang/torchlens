# postprocess/ - Graph Cleanup and Finalization

## What This Does
Transforms raw capture records into user-facing `Trace` state. The current full pipeline
has stable contract steps 0-20: graph traversal, conditional attribution, buffer fixes, loop
detection, labeling, finalization, streaming bundle finalization, and optional out
eviction plus parameter-reference release. Step order is load-bearing.

## Files

| File | Steps | Purpose |
|------|-------|---------|
| `__init__.py` | orchestrator | `postprocess()` prologue/epilogue, audit helpers, re-exports |
| `_contracts.py` | contracts | Step contracts, frozen rank, pinned-pair corpus, capture baseline |
| `_executor.py` | derivation + executor | Edge derivation, rank-keyed Kahn, import checks, StepSpec registry, run_pipeline |
| `_materialize.py` | 0 | Project capture events into raw `Op` state |
| `graph_traversal.py` | 1-4 | Output nodes, output ancestors, orphan removal, distances |
| `ast_branches.py` | 5 support, 11.5 | Conditional AST indexing and source variable names. Hot/cold FileIndex split: parsed ASTs (`_HeavyAst`) are released at the postprocess epilogue (`release_parsed_asts()`); span data + node-free projected calls persist, and unprojected-scope queries re-parse from RETAINED source (never disk), failing closed on anomaly |
| `control_flow.py` | 5-6 | Conditional attribution and buffer dedup |
| `loop_detection.py` | 7 adapter | Adapt Trace state and apply recurrence assignments |
| `loop_grouping_adapter.py` | 7 implementation | Backend-neutral recurrence grouping |
| `labeling.py` | 8-11 | Final labels, renaming, lookup keys, retained layer lists, field ordering |
| `finalization.py` | 12-20 | Undecorate, params, layers, modules, hash, streaming finalization/eviction, ref release |
| `incremental.py` | fastlog enrichment | Adds module paths and param addresses to sparse recordings |

## Step Contracts and the Derived Order (M10 + design-ppdag-v3)

Every step's `PostprocessStepContract` (`_contracts.py`) declares its exact
op-store COLUMN write set AND read set, plus `placeholder_probes` (reviewed
reads that legally observe the step-0 placeholder), `row_effects`
(`creates`/`deletes` whole-row sanctions — `creates` also carries row-clone
read legality), and `trace_state` tokens (closed 20-token vocabulary,
`r:`/`w:` stored form, `rw:` construction shorthand). The step order is
DERIVED from these declarations (`_executor.py`): every RAW/WW/WAR,
two-sided row-barrier, token, and barrier edge orients by the frozen
`LEGACY_STEP_RANK`, rank-keyed Kahn reproduces the registry exactly (import
checks R1/R2), and the semantic direction authority is the reason-bearing
`PINNED_ORDER_PAIRS` corpus (import check K1; test-side K2). A coordinated
rank+registry reversal passes the drift checks by construction — only the
corpus catches it, by named reviewed entry.

Under `TORCHLENS_POSTPROCESS_ASSERTIONS` a zero-cost-when-off audit
(class-swap instrumentation on the op row store) verifies each step writes
only declared columns; `TORCHLENS_POSTPROCESS_READ_AUDIT=enforce`
additionally verifies reads stay inside declared reads+probes (the write
audit keeps a read-free class so enforcing writes never pays a `cell_get`
override). The audit covers assignment/deletion AND in-place container
mutation (per-step order-canonical content fingerprints of mutable
dict/list/set cells on rows that existed at step begin). Whole-row lifecycle
is separate: creation is a produces contract; REMOVAL (husking releases and
re-reads every cell) checks the `row_effects` `deletes` sanction, and
released-row reads/deletes are row-lifecycle events, not column accesses.
`Op.copy()`'s whole-schema getattr loop is tagged by `row_clone_scope` as
the row-clone access kind — legal only on `creates` steps, no per-column
edges (the row barrier carries ordering). Recording mode additionally tags
writes content-effective vs no-op (a permanent no-op writer cannot
discharge a read-before-write finding). The recording/enforcement axes
matrix lives in `tests/support/postprocess_axes.py`; per-axis enforcement
and the phantom-declaration/no-op-writer union reports run in
`tests/test_postprocess_enforcement.py`. Known day-1 findings are pinned by
name in `tests/test_postprocess_dag.py` (step 3 reads `label`/`layer_label`
as data — root-cause pending, never silenced). Disclosed residuals:
mutables nested inside non-builtin custom objects; kind-table cells; the
`is OpRowStore` swap guard silently skips fork `OpStoreView`s and sealed
stores (sealing happens after step 20, outside every window). Transient
build scratch lives in three named per-phase workspaces (`ir/workspaces.py`),
not a flat `TraceBuildState` (deleted in M10): `RawGraphWorkspace` (capture
ingress + steps 0-11, also the backend `finalize_forward_session` ownership
token), `ModuleCaptureWorkspace` (module prep/stack capture, consumed at step
16), and `WrapperRuntimeWorkspace` (wrapper hot path). All three drop at
step 17.5 (the contracted container-adoption + workspace-drop seam).

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
| 11.75 | executor `_run_step_11_75` | Resolve deferred retention decisions through the attached `CaptureSession` (saves selected payloads; runs only when a capture session is attached) |
| 12 | `_undecorate_all_saved_tensors` | Strip TorchLens attrs from saved tensors |
| 13 | `torch.cuda.empty_cache` | Optional CUDA cache clear |
| 14 | `_log_time_elapsed` | Capture timing |
| 15 | `_finalize_param_logs` | Build and complete ParamLogs |
| 15.5 | `_build_layer_logs` | Build aggregate LayerLogs |
| 16 | `_build_module_logs` | Build ModuleLogs |
| 16.5 | `compute_graph_shape_hash` | Hash graph shape before pass-finished behavior changes |
| 17 | `_set_tracing_finished` | Switch Trace to user-facing behavior |
| 17.5 | executor `_run_step_17_5` | Adopt container records; drop the three per-phase workspaces |
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
