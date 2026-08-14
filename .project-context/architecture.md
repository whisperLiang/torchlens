# TorchLens Architecture

## Module Map

### `torchlens/_state.py` (~208 lines)
Global toggle, session state, context managers. Single source of truth for `_logging_enabled`
bool checked by every decorated wrapper. Also stores pre-computed lookup tables, WeakSet of
prepared models, active ModelLog reference. **Must never import other torchlens modules**
(prevents circular deps).

### `torchlens/user_funcs.py` (~664 lines)
Public API: `log_forward_pass()`, `show_model_graph()`, `validate_forward_pass()`,
`get_model_metadata()`, `validate_batch_of_models_and_inputs()`. Orchestrates the two-pass
strategy when selective layers requested.

### `torchlens/constants.py` (~645 lines)
7 FIELD_ORDER tuples (canonical field sets for LayerPassLog, ModelLog, etc.), function
discovery sets (~90 IGNORED_FUNCS, ORIG_TORCH_FUNCS listing ~2000 functions to decorate).

### `torchlens/decoration/` (2 files, ~1,710 lines)
- `torch_funcs.py` — One-time decoration of ~2000 torch functions. Core interceptor with
  barcode nesting detection, in-place detection, DeviceContext bypass.
- `model_prep.py` — Two-phase model preparation (permanent `_prepare_model_once` + per-session
  `_prepare_model_session`). Module forward decorator with exhaustive/fast-path split.

### `torchlens/capture/` (7 files, ~4,960 lines)
Real-time tensor operation logging during forward pass.
- `trace.py` — Forward-pass orchestration, session setup/cleanup
- `output_tensors.py` — Core logging: builds LayerPassLog entries, exhaustive/fast dispatch
- `source_tensors.py` — Logs input and buffer tensors as source nodes
- `tensor_tracking.py` — Barcode system, parent-child links, backward hooks
- `arg_positions.py` — O(1) tensor extraction via 3-tier lookup (639 static entries)
- `salient_args.py` — Extracts significant function args for metadata
- `flops.py` — Per-operation FLOPs computation (~290 ops)

### `torchlens/postprocess/` (6 files, ~3,179 lines)
26-step pipeline (declared contract keys `0`..`20` plus fractional inserts `11.5`, `11.75`,
`15.5`, `16.5`, `17.5` in `_contracts.py::POSTPROCESS_STEP_CONTRACTS`). Order is critical —
many steps depend on prior output.
- `graph_traversal.py` — Steps 1-4: output layers, ancestor marking, orphan removal, distance flood
- `control_flow.py` — Steps 5-6: six-phase conditional attribution (AST indexing, bool
  classification, event materialization, backward flood, forward arm attribution, derived
  views), buffer cleanup
- `loop_detection.py` — Step 7: isomorphic subgraph expansion, layer assignment
- `labeling.py` — Steps 8-11: label generation, rename, trim/reorder, lookup keys
- `finalization.py` — Steps 12-19: undecorate, ParamLog, ModuleLog, LayerLog, mark complete

### `torchlens/data_classes/` (10 files, ~3,821 lines)
- `model_log.py` — ModelLog: top-level container, 70+ attrs
- `layer_pass_log.py` — LayerPassLog: per-pass entry (~85+ fields)
- `layer_log.py` — LayerLog: aggregate class grouping passes
- `buffer_log.py` — BufferLog(LayerPassLog): buffer-specific computed properties
- `module_log.py` — ModuleLog, ModulePassLog, ModuleAccessor
- `param_log.py` — ParamLog (lazy grad via `_param_ref`)
- `func_call_location.py` — Structured call stack frame with lazy properties
- `internal_types.py` — FuncExecutionContext, VisualizationOverrides
- `interface.py` — ModelLog query methods: `__getitem__`, `to_pandas()`, 7-step lookup cascade
- `cleanup.py` — Post-session teardown, cycle breaking

### `torchlens/validation/` (3 files, ~2,795 lines)
- `core.py` — BFS orchestration, forward replay, perturbation checks
- `exemptions.py` — 4 data-driven exemption registries + 16 posthoc checks
- `invariants.py` — 18 metadata invariant categories (A-R): structural + semantic

### `torchlens/visualization/` (3 files, ~2,777+ lines)
- `_render_dot.py` — Graphviz rendering orchestration (validation, request resolution, RenderIR population, DOT emission); node/edge/subgraph emission lives in `_render_nodes.py`, `_render_edges.py`, `_render_leaf.py`, with IF/THEN labels and the override system in `_render_common.py`
- `elk_layout.py` — ELK-based layout for large graphs, Worker thread, sfdp fallback
- `dagua_bridge.py` — ModelLog → DaguaGraph conversion for dagua renderer

### `torchlens/utils/` (7 files, ~950 lines)
Stateless helpers: arg handling, tensor ops (safe_copy, tensor_nanequal), RNG capture/restore,
barcode hashing, object introspection, display formatting, collection manipulation.

## Data Flow

```
import torchlens
  → decorate_all_once()       # wraps ~2000 torch functions permanently
  → sweep_stale_belt_references()  # patches stale refs to the derived protocol-invisible set
  # (other `from torch import cos` style stale refs are recovered by the rescue re-run:
  #  an escape signal triggers ONE re-run with a TorchFunctionMode net that redirects
  #  stale calls to their wrappers; disclosed via trace.rescue_rerun)

log_forward_pass(model, input)
  → _prepare_model_once(model)   # permanent: tl_module_address, forward wrappers
  → _prepare_model_session(model) # per-call: requires_grad, buffers, session attrs
  → active_logging(model_log)    # enables _logging_enabled toggle
  →   model(input)               # forward pass — each torch op hits decorated wrapper
  →     torch_func_decorator     # barcode nesting → bottom-level ops logged
  →       log_function_output_tensors_exhaustive()  # builds LayerPassLog entry
  →       OR log_function_output_tensors_fast()     # reuses prior graph structure
  → postprocess(model_log)       # 26-step pipeline
  →   Steps 1-4: graph cleanup (outputs, ancestors, orphans, distances)
  →   Steps 5-6: control flow (Step 5a-5f conditional attribution, buffer dedup)
  →   Step 7: loop detection (isomorphic subgraph expansion)
  →   Steps 8-11: labeling (raw→final labels, rename, reorder, lookup keys)
  →   Steps 12-19: finalization (undecorate, ParamLog, ModuleLog, LayerLog)
  → return ModelLog
```

Key types flowing between modules:
- `Dict[str, Dict]` — raw tensor dict during capture (`_raw_tensor_dict` on ModelLog)
- `LayerPassLog` — per-pass tensor operation entry (~85+ fields)
- `LayerLog` — aggregate grouping passes of the same layer
- `ModuleLog` / `ModulePassLog` — per-module metadata
- `ParamLog` — per-parameter metadata with lazy gradient access

## Key Abstractions

### Toggle Architecture
Single `_logging_enabled` bool in `_state.py`. Wrappers check it on every call — when False,
one branch check, negligible overhead. No re-wrapping/un-wrapping per forward pass.

### Two-Pass Strategy
When user requests specific layers (not "all"/"none"), Pass 1 runs exhaustive to discover full
graph structure, Pass 2 runs fast saving only requested activations. Counter alignment between
passes maintained via identical increment logic.

### Conditional Branch Attribution (Step 5)
Step 5 now runs as six ordered phases:
1. 5a builds AST file indexes for source files referenced by terminal bool frames.
2. 5b classifies terminal scalar bools and records structural `ConditionalKey`s.
3. 5c materializes dense `ModelLog.conditional_records` IDs and rewrites bool metadata.
4. 5d runs the backward-only flood that marks branch-start parents.
5. 5e attributes ops and forward edges to branch arms, populating
   `conditional_arm_entry_edges` and `conditional_arm_children`.
6. 5f derives legacy THEN/ELIF/ELSE views and records `conditional_edge_passes` for
   rolled-mode divergence.

Primary branch metadata is cond-id-aware:
- `ModelLog.conditional_records` stores the canonical event records.
- `ModelLog.conditional_arm_entry_edges` stores arm-entry edges keyed by `(cond_id, branch_kind)`.
- `ModelLog.conditional_edge_passes` stores pass numbers for rolled edges whose arm labels
  vary across passes.
- `conditional_arm_children` on `LayerPassLog` / `LayerLog` stores per-node branch children.

Legacy `conditional_then_entry_edges`, `conditional_elif_entry_edges`, `conditional_else_entry_edges`,
`conditional_then_children`, `conditional_elif_children`, and `conditional_else_children`
are derived views computed from those primary structures.

### Barcode Nesting Detection
Random 8-char barcodes detect bottom-level vs wrapper functions. Barcode set on tensor before
call; if unchanged after → no nested torch calls → log it. If changed → nested call already
logged it.

### Operation Equivalence Types
Structural fingerprint: `{func_name}_{arg_hash}[_outindex{i}][_module{origin}]`, plus the
module-stack suffix appended at op creation. Used by loop detection (Step 7) to group
operations into layers.

### LayerLog Delegation
Single-pass layers: `__getattr__` delegates to `passes[1]`. Multi-pass per-pass fields:
raises **ValueError** (not AttributeError, to avoid Python's property/__getattr__ trap).

## Dependency Graph
```
_state.py          ← imported by everything (no outgoing torchlens imports)
constants.py       ← imported by capture/, postprocess/, data_classes/
utils/             ← imported by capture/, postprocess/, data_classes/, validation/
decoration/        → calls capture/ (via decorated wrappers)
                   → reads _state.py
capture/           → creates data_classes/ entries (LayerPassLog)
                   → reads _state.py, constants.py
postprocess/       → mutates data_classes/ entries
                   → reads constants.py
data_classes/      → references _state.py (TYPE_CHECKING only)
validation/        → reads data_classes/, calls original torch funcs
visualization/     → reads data_classes/ (LayerLog, ModelLog)
user_funcs.py      → orchestrates decoration/, capture/, postprocess/, validation/, visualization/
```

## Known Complexity

### Loop Detection (postprocess/loop_detection.py)
Most complex single module. BFS expansion of isomorphic subgraphs, iso group refinement with
direction-aware neighbor connectivity, adjacency union-find for layer assignment. Module
suffixes are present before loop detection, and `_rebuild_pass_assignments` clears stale
assignments after repeated expansion rounds. ~826 lines.

### Exhaustive/Fast-Path Split (capture/output_tensors.py)
Two parallel code paths that must maintain counter alignment. Fast path skips most metadata
but must match exhaustive path's operation ordering exactly.

### ELK Layout (visualization/elk_layout.py)
Node.js subprocess with V8 heap sizing, Worker thread to prevent stack overflow, stress
algorithm with O(n^2) memory (NEVER use for >100k nodes), Kahn's topological sort for seeding.

### Circular References (data_classes/)
ModelLog ↔ LayerPassLog ↔ ModelLog cycles. ModuleLog ↔ ModelLog cycles. ParamLog pins
nn.Parameter. All rely on Python's cyclic GC. Explicit `cleanup()` available.

## Conditional Attribution Limits

Fully attributed in eager Python `forward()`:
- `if` / `elif` / `else` chains
- Ternary `IfExp` (`x if cond else y`)

Classified only, not branch-attributed:
- `assert`
- standalone `bool(x)`
- comprehension filters
- `while`
- `match` guards

Documented false negatives:
- pure Python predicates such as `if self.training:` or `if python_bool:`
- `if tensor.item() > 0:`
- shape/metadata predicates such as `if x.shape[0] > 0:`
- functional conditionals such as `torch.where`

Unsupported / source-unavailable cases:
- Jupyter or REPL cells, `exec`, `eval`
- `torch.compile`, `torch.jit.script`, `torch.jit.trace`
- `nn.DataParallel` / `DistributedDataParallel`
- monkey-patched `forward` implementations

Deferred:
- dagua conditional-edge rendering
- ELK conditional rendering
- while-loop body attribution

### DeviceContext Bypass (decoration/torch_funcs.py)
Python wrappers bypass C-level TorchFunctionMode dispatch. Factory functions need manual
device kwarg injection when `torch.device('meta')` context is active (HuggingFace use case).

## Top-level module inventory

Complete inventory of `torchlens/` top-level entries (every root `*.py` module and every
package directory; `schemas/` included as the one non-package data directory). Roles are
one-liners derived from each module's docstring or a skim of its contents.

### Root modules

| module | role |
| --- | --- |
| `__init__.py` | Public package entry: lazy attribute surface (`_LAZY_ATTRS`), `__all__`, no torch side effects at import |
| `_capture_state_helpers.py` | Internal model state, cache, and input helpers for public trace capture |
| `_chunked_capture_helpers.py` | Internal chunked-forward helper functions for public trace capture |
| `_chunking.py` | Input chunking helpers for forward-only chunked capture |
| `_deprecations.py` | Shared helpers for additive public-API deprecations |
| `_distributed.py` | Detection of distributed (DTensor / device-mesh / TP / PP) model state, shared by capture entry and compat report |
| `_errors.py` | Shared internal exception types and actionable-message helpers built on `errors/_base` |
| `_fast_run.py` | Explicit guarded fast paths for repeated static-model execution (`run(fast=True)`) |
| `_input_coerce.py` | Duck-typed ergonomic input coercion for TorchLens entry points |
| `_input_walk.py` | Single-sourced model-input boundary traversal (normative dispatch for all input-tree walkers) |
| `_literals.py` | Shared `Literal` aliases for public option strings |
| `_robustness.py` | Tensor-variant detection and pre-flight guards for `trace` (meta/sparse/fake refusals) |
| `_runnable_attestation.py` | Numeric attestation and nondeterminism checks for runnable execution |
| `_runnable_call_arguments.py` | Sparse-call argument decoding and binding |
| `_runnable_call_outputs.py` | Sparse-call output binding and mutation checks |
| `_runnable_execution.py` | Transactional execution providers for the unified `Trace.run` surface |
| `_runnable_input_aliases.py` | Input alias topology and non-tensor tree contracts |
| `_runnable_input_metadata.py` | Input structure, literal, and metadata witness helpers |
| `_runnable_input_sites.py` | Live input sites and metadata contract checks |
| `_runnable_output_contracts.py` | Output reconstruction and post-execution contracts |
| `_runnable_path_faithfulness.py` | Path-faithfulness and state comparison helpers |
| `_runnable_providers.py` | Public provider entry points and run finalization |
| `_runnable_seam.py` | Narrow ownership seam between `Trace` and sparse runnable internals |
| `_runnable_state.py` | Non-executing state binding and allocation for sparse runnable traces |
| `_runnable_state_context.py` | State contracts and captured execution contexts |
| `_runnable_transaction.py` | Loaded-sparse transaction execution and allocation checks |
| `_runnable_verification.py` | Seed, RNG, attestation, and fork utilities for runnable runs |
| `_runnable_witness_contracts.py` | Control, shape, and host-escape witness checks |
| `_save_budget.py` | Running budget for retained activation bytes with typed refusal (`SaveBudgetExceededError`) |
| `_source_links.py` | Source-location link formatting helpers |
| `_split_rebind.py` | Compatibility helpers for behavior-preserving module decomposition |
| `_state.py` | Global state for toggle-gated decoration; single source of truth for capture-control mutable state |
| `_trace_selector_helpers.py` | Internal predicate and selector helpers for public trace capture |
| `_trace_state.py` | Run-state ownership for intervention execution (deliberately outside `intervention/`) |
| `_training_validation.py` | Shared validation helpers for training-compatible capture modes |
| `_transport.py` | Device/layout-aware host transport for tensor digest and codec paths |
| `_user_public_impls.py` | Private implementations backing public user-facing utility commands |
| `captured_run.py` | Shared public base types for TorchLens captured runs |
| `constants.py` | FIELD_ORDER tuples (canonical field sets) and torch function discovery sets |
| `facets.py` | Lazy alias stub: self-replaces in `sys.modules` with canonical `torchlens.semantic.facets` |
| `hash.py` | Provisional public structural-hash helpers |
| `observers.py` | User observer helpers: taps, scalar logs, record spans |
| `options.py` | Grouped option dataclasses (`CaptureOptions`, ...) for public TorchLens APIs |
| `quantities.py` | Numeric quantity types with unit-aware display |
| `runnable.py` | Frozen type contracts (enums and schema shapes) for sparse runnable `.tlspec` artifacts |
| `types.py` | Public type aliases and rarely used data classes |
| `user_funcs.py` | Public API entry points; contains every user-facing function |

### Packages

| module | role |
| --- | --- |
| `_io/` | Portable save/load implementation: scrubs a Trace to metadata plus safetensors blobs, writes/rehydrates directory bundles |
| `_trace_core/` | Private per-trace semantic store substrate (numpy-backed typed columns, per-trace interning) |
| `accessors/` | Accessor classes for TorchLens log collections |
| `attribution/` | Input-attribution methods for TorchLens |
| `autoroute/` | Auto-routing registries for model input and output handling (`autoroute.input`, `autoroute.output`) |
| `backends/` | Backend Protocol, public registry exports, and per-backend adapters |
| `bridge/` | External-tool bridge namespace (Captum, HF, SHAP, SAE Lens, LIT, profiler, ...) |
| `bundle/` | Single `Bundle` type for intervention-ready TorchLens model logs |
| `callbacks/` | Callback integration namespace with lazy Lightning support |
| `capture/` | Real-time tensor operation capture: source/output tensor logging, family tracking, forward-pass orchestration |
| `compat/` | Compatibility adapters and runtime support reports (`tl.compat.report`) |
| `data_classes/` | Core data structures representing a logged forward pass |
| `debug/` | Power-user debugging helpers for completed traces (`bisect_nan`, `hot_path`, ...) |
| `distributed/` | Distributed capture opt-in: arming, group lifetime identity, and evidence |
| `errors/` | Public TorchLens exception classes (base hierarchy, runnable errors, legacy path aliases) |
| `examples/` | Example-loading namespace for small TorchLens artifacts |
| `experimental/` | Experimental APIs with unstable naming and behavior |
| `export/` | Static export helpers for TorchLens logs |
| `fastlog/` | Lightweight predicate-recording namespace (`tl.record`) |
| `intervention/` | Import surface for intervention selectors, hooks, reruns, and bundles |
| `io/` | Public I/O and administrative helpers (log admin, intervention-spec save) delegating to `_io` and `user_funcs` |
| `ir/` | Internal backend-neutral IR for capture unification (op records, events, selectors, workspaces) |
| `merged/` | Cross-rank trace merging (rung C1): `merge_ranks`, `MergedTrace`, frozen merge vocabularies |
| `neuro/` | Extras-gated neuroscience namespace, import-inert, no public objects yet |
| `notebook/` | Extras-gated notebook namespace, import-inert, no public objects yet |
| `partial/` | Partial capture helpers for failed TorchLens forward ops |
| `postprocess/` | Postprocessing pipeline cleaning the model log after the forward pass |
| `receptive_field/` | Lazy public namespace for receptive-field analysis |
| `repgeom/` | Import-clean representation geometry helpers (NumPy + torch only), provisional |
| `report/` | Reporting helpers for TorchLens observer metadata |
| `schemas/` | JSON Schema documents for `.tlspec` manifests (v1/v2); data directory, not a Python package (no `__init__.py`) |
| `semantic/` | Semantic facet views: canonical home of facets (registry, recipes, patching) |
| `stats/` | Streaming statistics for out aggregation |
| `utils/` | Focused utility modules: RNG, tensor ops, argument handling, introspection, collections, hashing, display |
| `validation/` | Validation subpackage: saved outs, backward capture, and metadata invariants (the tripwire) |
| `visualization/` | Computational graph visualization via Graphviz (DOT rendering, ELK layout, dagua bridge) |
| `viz/` | Visualization convenience namespace: activation/tensor plots (montage, heatmaps, feature-map evolution) plus `bundle_diff` re-export |

### Dual homes (pending adjudication)

Four name pairs currently have two homes. Facts as of this writing (verified by reading);
the SF-33 adjudication decision will be recorded here when made -- this note does not
decide it.

- `torchlens/io` + `torchlens/_io`: `_io/` implements the portable save/load path
  (Trace scrub to metadata plus safetensors blobs, bundle write and rehydration);
  `io/` is the public-facing helper namespace (log administration such as `list_logs`
  and `reset_naming_counter`, intervention-spec save) that delegates to `_io` and
  `user_funcs`.
- `torchlens/viz` + `torchlens/visualization`: `visualization/` is the graph-rendering
  engine (Graphviz DOT emission, ELK layout, dagua bridge) behind `draw` and
  `show_model_graph`; `viz/` is a user-facing convenience namespace for
  activation/tensor plotting (montage, text tables, feature-map evolution, heatmaps,
  image scatter, line plots) and re-exports `bundle_diff` from `visualization/`.
- `torchlens/errors/` + `torchlens/_errors.py` (+ `intervention/errors.py`): `errors/`
  is the public exception package (`_base` hierarchy, runnable exception classes,
  legacy exception-path aliases); `_errors.py` holds shared internal exception types
  and actionable-message helpers built on `errors._base`; `intervention/errors.py`
  owns the intervention error catalog (severity tags, field-formatted messages) and
  imports from both.
- `torchlens/facets.py` + `torchlens/semantic/`: `semantic/` is the canonical
  implementation (`semantic/facets.py` plus `recipes` and `patching`); top-level
  `facets.py` is a lazy import-alias stub that replaces itself in `sys.modules` with
  `torchlens.semantic.facets` so the two module objects are identity-equal.
