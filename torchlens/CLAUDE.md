# torchlens/ - Core Package

## What This Is
TorchLens extracts outs and metadata from backend-resolved captures. PyTorch eager capture is the
stable default; MLX, JAX, tinygrad, Paddle, and TensorFlow are technical-preview backends. `import torchlens`
exposes the public API and compatibility shims, but torch wrapping is lazy: the first torch capture
prepares the model and calls `wrap_torch()` from `backends/torch/`.

## Architecture Overview

```
import torchlens
  |- exposes 96 top-level public names in __all__
  |- eagerly imports the core capture/intervention surface, fastlog/options, and
  |  the HuggingFace autoroute bridge; compat, export, report, stats,
  |  validation, and viz stay lazy
  |
trace(model, input, save=..., intervene=..., lookback=..., storage=...)
  |- backends/registry.py      - resolve torch / MLX / JAX / tinygrad / Paddle / TensorFlow backend
  |- backends/torch/model_prep.py - ensure torch is wrapped, prepare modules/buffers/params
  |- capture/trace.py          - run forward pass with active logging
  |- backends/torch/ops.py     - build raw torch op records during wrapper calls
  |- postprocess/              - current 20-step graph cleanup/finalization pipeline
  +- returns Trace

tl.record(model, input, save=...)
  |- uses the same wrapper hot path
  |- stores predicate-selected RecordContext/ActivationRecord values
  +- returns Recording; Recording.to_trace() materializes full structure
```

Selective `layers_to_save` uses a predicate-backed single pass when early labels are
sufficient and falls back to the two-pass strategy for final-label-only selectors such as
negative indexes, integer selectors, output labels, identity labels, and gradient
selection. String selectors keep the legacy substring contract. Unqualified recurrent
labels save all passes; pass-qualified labels such as `"attn:2"` save one 1-based pass.
Prefer `save=tl.func(...)`, `save=tl.in_module(...)`, and composed predicates for new
single-pass selective capture. The old `keep_op=`/`keep_module=` `record()` alias
kwargs are removed; `save=` is the only predicate spelling and `default_module=`
gates module-boundary event recording (uniformly — ALL module enter/exit events;
predicate-gated module-event selection has no public spelling).

Common unified capture examples:

```python
relu_trace = tl.trace(model, x, save=tl.func("relu"))
paddle_trace = tl.trace(paddle_model, paddle_x, backend="paddle")
tf_trace = tl.trace(tf_model, tf_x, backend="tf")
windowed = tl.trace(
    model,
    x,
    save=tl.func("conv2d") & tl.followed_by(tl.func("relu")),
    lookback=4,
    lookback_payload_policy="detached_raw",
)
patched = tl.trace(
    model,
    x,
    save=tl.func("attn"),
    intervene=tl.when(tl.func("attn"), tl.scale(0.5)),
)
streamed = tl.trace(model, x, save=tl.in_module("encoder"), storage=tl.to_disk("run.tlspec"))
recording = tl.record(model, x, save=tl.func("relu"))
trace_from_recording = recording.to_trace()
trace.draw(show_containers="nodes")
```

Provisional semantic I/O surface (review-day names):

```python
log = tl.trace(model, x, output_style="classification", output_head="logits")
log.output_table(top_n=5)
log.summary(level="output")
log.to_pandas(include_decoded_output_summary=True)

input_log = tl.trace(model, raw_text, transform=text_to_tensor, save_raw_input="small")
input_log.draw(show_input_transform_summary=True)

mds_layers = tl.in_module("block1") | tl.in_module("block2")
image_log = tl.trace(
    model,
    image_list,
    transform=image_batch_to_tensor,
    save=mds_layers,
    save_raw_input=True,
    output_style="classification",
)
image_log.model_profile
image_log.output_table(top_n=5)
tl.repgeom.mds_evolution(image_log, save=mds_layers, min_n=8)
tl.repgeom.rdm_evolution(image_log, save=mds_layers)
tl.viz.feature_map_evolution(image_log, save=mds_layers)
tl.repgeom.scree_evolution(image_log, save=mds_layers)
image_log.draw(node_spec_fn=tl.repgeom.mds_scatter_node_spec(max_thumbnails=8))
image_log.draw(node_spec_fn=tl.repgeom.rdm_node_spec(max_stimuli=8))
image_log.draw(node_spec_fn=tl.viz.feature_map_node_spec())
image_log.draw(node_spec_fn=tl.repgeom.scree_node_spec())
```

Sprint B annotation/MDS names are provisional until review-day signoff. `Trace.model_profile`
is computed, not persisted. `tl.repgeom.mds_evolution(...)` requires the target batch
activations to have been saved at capture time; use a curated `save=` subset, not `save="all"`,
for image batches. `Trace._annotation_blobs` is public-provisional only for render-time
annotation payloads and compatibility review.
Sprint C RDM, feature-map, and scree node visuals are PIL-only render-time images composed
from `tl.viz.render_*` primitives and are provisional until review-day signoff.

`backward_ready=True` is the public opt-in for losses built from saved outs. It keeps
floating tensors graph-connected, preserves user `requires_grad`, and rejects incompatible
detaching or disk-only out storage.
`inference_only=True` is the opt-in no-grad capture path for forward-only analysis; it is mutually
exclusive with backward-related capture because it discards the autograd graph.

## Top-Level Modules

| Path | Purpose |
|------|---------|
| `__init__.py` | Top-level API, 94-name `__all__`, deprecation shims, `peek`/`extract` helpers |
| `_state.py` | Global logging toggle, active log, decoration maps, prepared-model registry; no torchlens imports |
| `_trace_state.py` | Small runtime state enum exposed through `torchlens.io` |
| `_errors.py`, `errors/` | Public and legacy exception classes |
| `_io/`, `io/` | Portable `.tlspec` save/load, manifest, lazy tensor refs, public I/O helpers |
| `options.py` | Capture, save, visualization, replay, intervention, and streaming option groups |
| `observers.py` | `tap()` and `record_span()` observer helpers |
| `report/` | `report.explain(log)` and capture-time scalar logging |
| `stats/` | Streaming stats and `aggregate()` over dataloaders |
| `types.py`, `accessors/` | Moved type/accessor aliases for non-top-level public names |

## Subpackages
- `capture/` - real-time forward and backward operation logging.
- `data_classes/` - `Trace`, `Layer`, `Op`, module/param/buffer/grad logs. The declared
  record schema carries per-field `StorageBinding` axes (generated `_schema_bindings.py`,
  regenerate with `tools/generate_record_schema.py`); Trace fields have a declared
  component ownership map (`_trace_components.py`).
- `_trace_core/` - private columnar store substrate (columns, pools, edge-occurrence
  table, payload arena, overlays, `TraceCore`). The M5 Op seam is LIVE: every captured
  `Op` is a two-word `(_core, _row)` facade over the per-trace `OpRowStore`
  (`op_store.py`) held at `trace._trace_core` (declared `FieldPolicy.DROP`); rows are
  row-major lists while building and seal after step 20 (columnar transpose with
  numeric packing at >=512 rows). `Op.copy()`, pickle restore, fork shells, and
  preview backends use detached single-row stores. M6 relations are LIVE
  (`relation_views.py`): on FINISHED traces the relation accessors return IMMUTABLE
  views — `tuple` for label sequences (`parents`, `children`, `modules`,
  `module_call_stack`, conditional child lists, ...), `frozenset` for label sets
  (`input_ancestors`, `output_descendants`, `root_ancestors`,
  `internal_source_ancestors`) — an authorized public type break (JMT 2026-08-12):
  in-place mutation raises, assignment still works and normalizes to the view type,
  and equal views may be shared across records. `parents`/`children` live in the
  core's canonical dataflow edge-occurrence table (CSR by edge id) and rematerialize
  lazily; dict-shaped relation metadata (`parent_arg_positions` etc.) stays mutable.
  M7a group views are LIVE: `equivalent_ops`/`recurrent_ops` cells hold ONE shared
  `GroupRef` per membership group (`groups.py`); reads resolve to the group's cached
  immutable view (`frozenset`/`tuple`) and removal scrub rebinds the group row once.
  M7b shared-fact blocks (`fact_blocks.py`): the call-level container facts
  (`code_context`, `non_tensor_pos_args`, `non_tensor_kwargs`, `func_non_tensor_args`,
  `func_config`, `arg_names`) live ONCE per FunctionCall group and `param_shapes` once
  per distinct value (the ParamAlias block); member cells hold the `_FACT` sentinel and
  the facade hydrates the exact public container type per row on first read (fresh
  mutable copy, cached back — per-row isolation is unchanged). Sibling outputs of one
  wrapped call also share ONE journal-side `FunctionCallRef`.
  During postprocess the staging containers remain real mutable builtins; legacy
  list/set state normalizes on load.
  M8 remaining kinds: `Layer` is an AGGREGATE FACADE — the ~86 representative
  fields the dict era copied from the first pass are class-level mirror
  descriptors reading through to `ops[0]` (with the exact copy-time
  normalizations); writes shadow per-layer in `__dict__`, deletes tombstone,
  and cleanup/removal materialize mirrors before husking the backing ops.
  `Param`/`Buffer`/`FuncCallLocation`/`ModuleCall`/`Module` are row facades
  over per-trace kind tables (`record_rows.py`, `TraceCore.kind_rows`,
  adopted by the build passes and sealed with the core; preview backends and
  loads stay detached-backed). The canonical label -> op-row index binds at
  the freeze as `TraceCore.label_rows`.
  M9 backward epochs: `GradFn`/`GradFnCall`/`BackwardPass` are row facades;
  each successful backward projection binds an atomic `BackwardEpoch`
  (`TraceCore.backward_epochs`) — a full rebuild atomically replaces the
  epoch list, a clean tail fold extends the live epoch — whose per-kind row
  stores back the projected records. The lazy watermark/revision
  invalidation stays trace-side and byte-identical; loaded/preview traces
  keep detached-backed backward records.
  M10: `TraceBuildState` is GONE — its transient fields dissolved into three
  named per-phase workspaces (`ir/workspaces.py`: `RawGraphWorkspace` for
  capture ingress + steps 0-11, `ModuleCaptureWorkspace` for module
  prep/stack capture consumed at step 16, `WrapperRuntimeWorkspace` for the
  wrapper hot path), each dropped at the transient-state cleanup seam; the
  backend `finalize_forward_session` protocol takes the raw-graph workspace
  as its ownership token. Each `POSTPROCESS_STEP_CONTRACTS` entry (v2)
  declares its exact op-store COLUMN write AND read sets plus
  `placeholder_probes`, `row_effects` (creates/deletes row sanctions), and
  closed-vocabulary `trace_state` tokens; the step order is DERIVED from
  these declarations by rank-keyed Kahn (`postprocess/_executor.py`), with
  the frozen `LEGACY_STEP_RANK` and the reason-bearing `PINNED_ORDER_PAIRS`
  corpus as the two-key direction authority — a coordinated rank+registry
  reversal slips the drift checks by construction and only the corpus
  catches it. `TORCHLENS_POSTPROCESS_ASSERTIONS` arms a zero-cost-when-off
  write audit and `TORCHLENS_POSTPROCESS_READ_AUDIT=enforce` the read side
  (class-swap instrumentation in `op_store.py`), scoped by per-step
  begin/run/end/assert windows (postconditions run outside any window; no
  window survives the loop). Declared-never-observed writes and reads live
  in reason-bearing phantom-exemption ledgers, no-op writers are pinned and
  cannot discharge a read, and day-1 findings are pinned by name; widening
  any set is a reviewed contract diff.
  M11: `Trace.fork()` is COPY-ON-WRITE (`data_classes/_trace_fork.py`): the
  fork core wraps the sealed op store and every kind table in per-fork
  `OpStoreView`s (own overlay; base overlay/rows snapshot at fork;
  eager fork-time isolation of exact builtin mutable containers with
  tensor/callable identity preserved; GroupRef translation to cloned group
  tables; record/accessor translation for cell-held references), fork
  records are fresh two-word shells at the SAME rows, and only Layer shadow
  dicts, record extras, and the policy-driven trace-side remainder are
  copied. The object-graph forkcopier (typed deepcopy engine) is deleted;
  differentiable replay forks drop the deep-cone/shallow-rest split. The
  facade identity cache is weak-valued (strong side table only for
  non-weakref-able `Op`, whose weakref refusal aliases-v1 pins); the
  standalone compaction passes are folded into the core freeze seam
  (`data_classes/_compaction.py`); `TraceCore.transaction()` checkpoints
  every mutation surface atomically. Architecture of record:
  `docs/reference/trace_core_design.md`.
- `backends/torch/` - torch function wrapping, explicit wrap/unwrap, module prep.
- `fastlog/` - sparse predicate recording with RAM/disk storage and recovery.
- `postprocess/` - graph cleanup, conditionals, loop detection, labeling, finalization.
- `validation/` - forward replay, backward validation, metadata invariants, `.tlspec` schema checks.
- `visualization/` - Graphviz rendering, rank layout, NodeSpec, themes, overlays, bundle diff.
- `intervention/` - selectors, sites, hooks, helpers, Bundle, fork/replay/rerun/save.
- `intervention/_super/` - internal Bundle-level Super* aligned views and accessors.
- `intervention/_topology/` - internal bundle supergraph and topology diff support.
- `bridge/`, `compat/`, `callbacks/` - optional integrations and migration facades.
- `notebook/`, `neuro/` - appliance package boundaries gated by extras.

## Key Concepts

### Toggle Architecture
- Lazy wrapping: `wrap_torch()` installs wrappers on first capture or explicit call.
- Persistent wrappers: after wrapping, calls only pay a `_state._logging_enabled` check when
  logging is off.
- `active_logging(trace)` enables logging during the forward; `pause_logging()` protects
  internal TorchLens tensor ops from recursive capture.
- Stale `from torch import cos` style references are recovered by the rescue re-run
  (`backends/torch/rescue.py`, disclosed via `trace.rescue_rerun`); the protocol-invisible
  constructors keep targeted module-attr patching via the mechanical belt (`backends/torch/belt.py`).

### Module Containment
Module containment is captured via a wrap-forward stack helper at
`backends/torch/module_stack.py`. Both fastlog and exhaustive modes share the helper. Each
captured op snapshots the stack at op-creation time; downstream postprocess only appends
the canonical module-path suffix to `equivalence_class` for loop detection. This replaces
the older tensor-entry/exit thread-replay system removed in v2.18 (sprint
module-containment-refactor).

### Data Flow
1. Decoration intercepts torch function calls.
2. Barcode nesting detection identifies bottom-level operations.
3. `capture/` builds raw `Op` entries.
4. `postprocess/` removes orphans, marks conditionals, detects loops, labels nodes, builds logs.
5. `Trace` exposes lookup, visualization, validation, save/load, intervention, and summary helpers.

### Journal Producer (single since producer unification P7)
Torch captures journal decomposed `OpRecord` rows (`ir/op_record.py`: `OpCore` + typed
facets, strict protocol with legacy flat-name properties) through the ONE commit tail
`capture/projections.py::commit_op`; step 0 ingests them via the generated scatter
(`ingest_op_records`). The op lane is genuinely append-only: post-commit knowledge rides
the typed `OpAmendment` lane (nine exact-set families, `append_amendment` the single
writer) and every amended-state read folds through `CaptureEvents.amended_op_records()`.
grad-fn handles live only in the journal side index (`grad_fn_handles_by_label_raw`).
The legacy `OpEvent` torch producer and its `TORCHLENS_CAPTURE_PRODUCER` switch were
deleted (P7); preview backends keep emitting compat `OpEvent`s until S15 and adapt at the
one ingest boundary (`op_record_from_event`), with `PATH_TO_FLAT` as the amendment fold
guard and `_clone_op_event_for_replay` record-shape-aware, all retained-with-schedule.

### Portable Artifacts
`tl.save()` and `tl.load()` route through `_io/bundle.py`. Unified `.tlspec` directories have
`manifest.json` plus safetensors blobs; public schema validation lives in `validation/__init__.py`.
Runnable saves are sparse by default and produce `sparse_recorded_taken_path_v2` descriptors with
REQUIRED explicit execution-context records (per-call `CallExecutionContext` + capture-scoped
`AmbientExecutionContext`), restored at replay or refused typed; legacy v1 artifacts load
analysis-only. `include_weights=True` bundles the full capture-time
`state_dict` (named parameters plus persistent buffers) as a separate `state_dict_v1` blob family;
the sparse core still contains no tensor values. Load binds it through the same strict state
contract used by `Trace.load_state_dict()`, while explicit user state overrides it at run time.
Used non-persistent buffers always ship in the REQUIRED `runnable_nonpersistent_buffer_v1` family
(declared state; not gated on either include flag; disclosed at save).
`include_activations=True` independently writes capture-time `save=`-selected
`out`/`transformed_out` values as `selected_activation_v2`, including physical
`InputAttestationFingerprint` eligibility records. Loaded values are available through
`Trace.archived_activations` for inspection and eligible byte-exact attestation only; the sparse
scheduler never consumes them. Original-input, capture-equivalent real-state runs report
`attested` or fail transactionally with `numeric_attestation_failed`; changed-input (logical or
physical), random-state, and nondeterministic-capture-context runs report `not_applicable`, and
`attested` always implies `verified`.
The frozen `ReadinessStatus`, `RunProvider`, `StateSource`, `PathFaithfulness`, `DivergencePolicy`,
`NumericAttestationStatus`, and `RunnableErrorCode` vocabularies live in `torchlens.runnable` and are
documented exhaustively in `docs/reference/runnable_tlspec_contract.md`. r37 additions:
`state_alias_topology_unsupported` (save-time state-topology refusal; tied live-identity state
stages as one alias-group allocation) and `context_field_invalid` (parse-time closed-vocabulary
context refusal); zero-tensor-leaf and instance-stateful container outputs refuse at save with
`missing_output_container_contract` via the one per-kind capability table.
Non-torch preview backends use `payload_policy="array_payloads"` when their codecs can materialize
payloads; Paddle bf16 payloads carry logical dtype metadata because NumPy transports them as
`uint16`. TensorFlow preview payloads also use `array_payloads` for dense numeric/bool forward
arrays and preserve `tf.bfloat16` logical dtype metadata.
Intervention specs can be saved at audit, executable-with-callables, or portable levels.

### Appliances
The appliance subfolders `notebook` and `neuro` are part of the 2.x package layout. They
currently enforce their extras by importing required dependencies, but export no public
objects yet.
