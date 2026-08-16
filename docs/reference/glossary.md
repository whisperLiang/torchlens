# TorchLens glossary

This glossary fixes the public meaning of TorchLens terms. It describes the current 2.x API;
removed spellings are listed separately in [Deprecations](deprecations.md).

## Capture products

**Trace**
: The finalized record of one executed path: operations, dataflow edges, module context, selected
  payloads, and capture metadata. `tl.trace(...)` returns a `Trace`.

**Recording**
: A sparse, torch-only event stream returned by `tl.record(model, inputs, save=...)`.
  `Recording.to_trace()` materializes full graph structure; reading a payload that was not selected
  by `save=` fails explicitly.

**PartialTrace**
: A recoverable prefix of a failed `tl.trace(...)` capture. A raised capture exception may expose
  `exc.partial_log`; recover it with `tl.partial.from_failed_capture(exc)`. A partial trace is not
  silently treated as complete.

**Failed partial Recording**
: A `Recording` with `status="partial_error"` and `failed=True`. Use
  `on_forward_error="attach_partial"` to attach it as `exc.partial_recording`, or
  `on_forward_error="return_partial"` to return it. It carries string-only error metadata,
  `n_ops_completed`, and best-effort `last_event_*` fields. It cannot be passed to
  `Recording.to_trace()` or `Recording.log_backward()`.

**Capture outcome**
: The frozen `Trace.outcome`, `Recording.outcome`, or `PartialTrace.outcome`. Its status is one of
  `COMPLETE`, `HALTED`, `ABORTED_NONFINITE`, `FAILED`, `UNATTESTED`, or `UNKNOWN`; failed outcomes
  also identify the phase. See [Capture outcomes](capture_outcomes.md).

**Episode capture**
: One wrapped multi-step generation run captured as a single product
  (`capture_kind=episode`): `tl.trace(episode_root, x, episode=EpisodeSpec(stepped_module=...))`
  stamps the declaration and lands a per-step status ledger (header + rows with
  `complete`/`interrupted`/`absent` statuses, emitted tokens, and the managed-RNG entry
  seed) at `trace.annotations["episode"]`. A diagnostic-tier product for tens of steps
  (cost is superlinear in step count); the ledger is a disclosure, never a settlement
  authority, and its persistence is gated until the coordinated schema bump. All episode
  spellings are provisional (no deprecation shim owed). See
  [Episode capture](episode_capture.md).

## Graph records

**Op**
: One executed callable invocation. Ops are the dataflow graph's nodes and have unique full labels
  such as `relu_1_2`.

**Layer**
: The user-facing grouping of equivalent or recurrent Ops. A recurrent layer may have multiple
  passes, addressed with a one-based suffix such as `linear_1:2`.

**Module**
: A captured framework module or backend-neutral module-like object. A module contains Ops but is
  not itself an Op.

**Parameter / Buffer**
: Registered model state. Parameters are trainable state; buffers are non-parameter state and may
  have multiple captured versions. A training-mode `BatchNorm2d` collapse label can say
  “5 layers total” because it counts the parameter leaves `weight` and `bias` plus the buffer
  leaves `running_mean`, `running_var`, and `num_batches_tracked`; it is not claiming five
  BatchNorm calls.

**GradFn**
: A first-class live backward-autograd node captured during backward logging. Portable artifacts
  do not contain PyTorch's live autograd graph.

**Facet**
: A named semantic view into one captured value, such as an attention projection or an LSTM state.

### Documented-unstable ATen profile INDEX

The wave-0 ATen execution-profile core is gated off for ordinary callers and its fields remain
`FieldPolicy.DROP`. Only the pytest prerelease switch can persist and reload these records; ordinary
v7 artifacts contain no primitive profile. The public recorder, entity accessors, and absent-profile
refusal remain unavailable until the S2-owned capability amendment lands. The names below are the
exact surface already introduced by the gated core. Each is documented unstable and may be renamed
or removed without a compatibility alias. No validation or honesty tripwire may be weakened.

<!-- ATEN-UNSTABLE-INDEX:START -->

| Surface | Exact spelling or token | Stability |
| --- | --- | --- |
| Facades, record kind, disclosure, and provenance | `AtenOp`, `OpRef`, `SuperAtenOp`, `primitive_op`, `mode_paused_interior`, `exact_via_aten`, `heuristic` | unstable -- no deprecation shim owed |
| Field-order contract | `PRIMITIVE_OP_FIELD_ORDER` | unstable -- no deprecation shim owed |
| Primitive fields | `label`, `sequence`, `capture_phase`, `forward_pass_index`, `backward_epoch_index`, `owner_func_call_id`, `parent_op_refs`, `parent_grad_fn_call_ref`, `owner_status`, `decomposition_slot`, `namespace`, `operator`, `overload`, `schema`, `schema_fingerprint`, `module_call_stack`, `input_tensor_facts`, `output_tensor_facts`, `mutation_kind`, `view_copy_kind`, `autocast_context`, `dispatch_key_context`, `grad_fn_ref`, `grad_fn_link_status`, `grad_fn_link_provenance`, `algorithmic_flops`, `flop_status`, `flop_formula_source`, `flop_formula_version`, `outcome`, `exception_type`, `execution_context` | unstable -- no deprecation shim owed |
| Observer-gap fields | `kind`, `capture_phase`, `sequence_before`, `sequence_after`, `owner_func_call_id`, `parent_op_refs`, `reason` | unstable -- no deprecation shim owed |
| Redundant Op-reference fields | `op_row_index`, `op_label`, `func_call_id` | unstable -- no deprecation shim owed |
| Tensor-fact fields | `container_path`, `tensor_impl_capability`, `logical_version`, `storage_alias_group`, `shape`, `stride`, `dtype`, `device`, `layout`, `requires_grad` | unstable -- no deprecation shim owed |
| Execution-context fields | `pytorch_version`, `backend`, `device_model`, `device_capability`, `grad_mode`, `inference_mode`, `module_training_summary`, `autocast`, `deterministic_algorithms`, `tf32_matmul_policy`, `sdpa_policy`, `compile_stance`, `owner_thread_coverage`, `completeness_witness_mode` | unstable -- no deprecation shim owed |
| Super comparison fields | `comparison_status`, `has_observation_gap` | unstable -- no deprecation shim owed |
| Invariant contracts | `primitive_op_invariants`, `non_torch_primitive_op_inert` | unstable -- no deprecation shim owed |
| Switch-active load failures | `primitive_op_schema_invalid`, `primitive_op_fk_invalid` | unstable -- no deprecation shim owed |
| Capture phases | `forward`, `backward`, `setup` | unstable -- no deprecation shim owed |
| Mutation classes | `none`, `in_place`, `out_variant`, `metadata_only`, `unknown` | unstable -- no deprecation shim owed |
| View/copy classes | `view`, `copy`, `alias`, `unknown` | unstable -- no deprecation shim owed |
| Owner classes | `forward_op`, `backward_grad_fn_call`, `orphan`, `unresolved` | unstable -- no deprecation shim owed |
| Grad-link classes | `linked`, `unlinked`, `conflict`, `not_applicable` | unstable -- no deprecation shim owed |
| Dispatcher outcomes | `returned`, `raised` | unstable -- no deprecation shim owed |
| FLOP evidence classes | `formula_exact`, `estimated`, `unsupported` | unstable -- no deprecation shim owed |
| Super alignment classes | `all_present_same_schema`, `all_present_different_schema`, `sparse`, `coverage_indeterminate` | unstable -- no deprecation shim owed |
| Execution and disclosure tokens | `forced_eager`, `strict_subclass_constructor` | unstable -- no deprecation shim owed |
| Temporary label grammar | `aten_<sequence>` | unstable -- no deprecation shim owed |

<!-- ATEN-UNSTABLE-INDEX:END -->

`AtenOp` is one value-free dispatcher call measured during a concrete capture. Its label is opaque,
capture-local, and intentionally excluded from universal Trace string lookup. `OpRef` is a redundant
dense foreign key whose row index, Op label, and function-call witness must all agree. A
`mode_paused_interior` entry says only that TorchLens paused its owned dispatch observer around a
strict Tensor-subclass constructor; recorded rows and counts on that parent are lower bounds, and no
synthetic primitive row is created for the unseen interior.

`SuperAtenOp` aligns observed rows positionally by `decomposition_slot`. Its `comparison_status`
distinguishes equal-schema coverage, different-schema coverage, proven sparse membership, and
coverage that is indeterminate because at least one member has an observation gap. Positional
alignment is evidence, not semantic equivalence.

## Selection and storage

**Predicate**
: A site selector built with `tl.func`, `tl.module`, `tl.in_module`, `tl.label`, and boolean or
  temporal composition. `save=` is the only selective sparse-capture spelling; removed
  `keep_op=` and `keep_module=` arguments raise `TypeError`.

**Lookback**
: A bounded window of earlier metadata or detached payloads retained so a later temporal predicate
  can select its predecessors.

**Save mode**
: The activation-retention policy: `copy` isolates the retained value, `reference` preserves the
  live autograd-capable object, and `view` intentionally leaves aliasing visible.

**Streaming storage**
: A `storage=tl.to_disk(...)` capture that writes selected payloads while capture proceeds. It is
  distinct from saving a completed in-memory Trace.

**Capture cache**
: The opt-in `cache=True` content-hash store of finished captures. A hit requires the model's
  tensor content (including device and `requires_grad`), training flags, module tree and
  `forward` code, plain instance attributes, user-registered module hooks, inputs, and capture
  configuration to all match. Entries are HMAC-authenticated, bounded by entry count and bytes,
  and `tl.clear_capture_cache()` empties the cache while preserving its secret.

## Replay and intervention

**Intervention**
: A value or gradient edit performed at a selected live site. `tl.when(predicate, helper)` binds a
  selector to helpers such as `tl.zero_ablate()` or `tl.scale(...)`.

**Bundle**
: A named collection of aligned Traces, constructed with `tl.bundle(...)` or `tl.Bundle(...)`, for
  cross-run comparison.

**Runnable artifact**
: A `.tlspec` saved at `level="runnable"` with a sparse taken-path descriptor. Its source capture
  must use `intervention_ready=True`; this flag supplies replay templates even when no intervention
  is planned.

**Readiness**
: The non-executing runnable state `ready` or `unavailable`.

**Path faithfulness**
: A runnable result verdict: `verified`, `diverged`, or `unverifiable`. `unverifiable` means the run
  completed without enough evidence for the stronger claim; it is not a pass.

**Numeric attestation**
: Byte-level comparison of eligible archived selected activations. Its status is `attested`,
  `numeric_attestation_failed`, `not_applicable`, or `not_present`; attestation never upgrades path
  faithfulness.

## Capture honesty and visualization

**Capture verification reason**
: The machine-readable `Trace.capture_verification_reason` explaining why a live capture is not
  positively verified, for example `mode_rescue_rerun`, `escape_rescue_unrecovered`,
  `owner_thread_tripwire_changed`, or `dynamo_region_not_logged`.

**Rescue re-run**
: A single disclosed second forward under a `TorchFunctionMode` net that redirects detected stale
  pre-wrap torch references to their wrappers. The trace records `trace.rescue_rerun` and reason
  `mode_rescue_rerun`.

**Collapse**
: A rendering-only condensation of module detail. `collapse="none"`, `"auto"`, `"max"`, or a
  float in `[0, 1]` never changes the underlying Trace. Traces above the preflight compute
  ceiling `COLLAPSE_OPTIMIZER_MAX_OPS` (2000 ops) decline smart collapse with a
  `TorchLensWarning`: `draw()` renders uncollapsed, `Trace.collapse_plan()` refuses typed
  (`collapse_plan_unavailable`), and `Trace.collapse_schedule()` degrades to its single
  full-graph step.

**Run folding**
: `fold_repeats=True` replaces eligible runs of distinct same-class sibling modules with a
  representative and an honest `+N more` label. It differs from recurrence, where `(xN)` means the
  same parameters executed repeatedly.

**Receptive field / Projective field**
: Influence geometry through the captured DAG toward inputs / toward outputs. Entity-level
  `op.receptive_field` and `op.projective_field` pair with Trace-level
  `Trace.receptive_fields()` and `Trace.projective_fields()` tables.

**Encoding channel** *(unstable — no deprecation shim owed)*
: A declarative value source (record field name, scalar builtin, or callable `node -> value`)
  mapped to a visual channel on `draw()`. v1 ships the color channel; size and rank channels
  follow. Channels are presentation-only: they never change the Trace or the collapse plan, and
  they require the Graphviz dot layout (`layout="auto"` forces dot when a channel is active;
  explicit `layout="rank"` refuses with `encoding_requires_dot_layout`).

**color_by** *(unstable — no deprecation shim owed; keyword-only)*
: `draw(color_by=...)` fills eligible operation nodes from a colorblind-safe sequential ramp,
  normalized linear min-max over the visible nodes (the legend states the transform). Missing,
  non-finite, or rolled-ambiguous values leave nodes unencoded with a legend note. On rolled
  multi-pass layers, field sources resolve through a name-keyed rolled-aggregate allowlist:
  per-pass-varying and first-pass-only sources are never painted as if uniform, exact cross-pass
  totals encode with a mandatory aggregation legend line, and unclassified sources refuse
  (`encoding_source_invalid`). Wrong-typed values refuse (`encoding_value_invalid`); a raising
  callable refuses with the original exception chained (`encoding_callable_error`).

**size_by** *(unstable — no deprecation shim owed; keyword-only)*
: `draw(size_by=...)` sizes eligible operation nodes from a scalar field, the closed `"dims"`
  shape token (numel of the non-batch output shape — the D4 default mapping, applied because D4
  is unruled), or a callable `node -> scalar`. Emitted sizes are width/height MINIMUMS under
  `fixedsize=false`: a label can never be truncated by an encoding and fonts never scale; the
  encoded area is clamped to 4x the default node area. Strictly opt-in — plain `draw()` keeps
  uniform boxes. On rolled multi-pass layers size REFUSES where color degrades
  (`size_by_rolled_varying`): a size source that cannot be certified single-valued
  (marker-varying, first-pass projection, varying shape under `"dims"`) has no honest "n/a"
  rendering. Exact cross-pass totals (`total_*`) encode with a mandatory aggregation legend line;
  callables bypass the rolled table with a legend disclosure.

**scale (size channel)** *(unstable — no deprecation shim owed; keyword-only)*
: The size-channel scale transform: `"sqrt"` (default; compresses dynamic range) or `"linear"`
  (the literal area motif). Log is rejected by design (flattens 512-vs-4096). Supplied without
  `size_by` it refuses (`scale_requires_size_by`); an unknown token refuses
  (`encoding_scale_invalid`). Every legend drawn states the active scale.

**show_legend tri-state**
: `show_legend` accepts `None` (default, AUTO: no legend unless an encoding channel is active,
  then a channel-only disclosure legend), `True` (full theme legend, plus channel rows when
  active), and `False` (no legend, honored even with channels active — the encoding is then
  undisclosed). The `None` value is *(unstable — no deprecation shim owed)* pending ratification.

## Backend-neutral identity

**Backend**
: `Trace.backend`, such as `torch`, `jax`, `mlx`, `tinygrad`, `paddle`, or `tf`.

**Module identity mode**
: `Trace.module_identity_mode`, describing how the backend identifies module-like ownership.

**Backend address / resolver status**
: Portable origin and resolution fields for backend records. They travel with `param_source`,
  `dtype_ref`, and `device_ref`; none should be inferred from a torch-only object when a
  backend-neutral field exists.

## Site selectors

**Label selectors**
: `tl.label` (exact final label), `tl.contains` (label substring), and `tl.regex` (label
  regex pattern) select ops by their public labels. `tl.where` filters a table with a
  predicate callable.

**Structural selectors**
: `tl.func` / `tl.module` / `tl.in_module` select by callable or module context;
  `tl.head` selects one attention head; `tl.facet` selects a semantic facet view;
  `tl.func_transform` selects `torch.func` transform boundary ops; `tl.output` selects
  model outputs, and `tl.output_at` / `tl.input_at` select a nested output or model-input
  path.

**Temporal composition**
: `tl.followed_by` (retroactive successor) and `tl.preceded_by` (lookback predecessor)
  compose with boolean operators to form temporal predicates over the op stream.

**Backward selectors**
: `tl.grad_fn` (backward grad_fn), `tl.grad_fn_label` (exact grad_fn label),
  `tl.grad_input` / `tl.grad_output` (backward event tensors), `tl.in_backward_pass`
  (one backward pass number), and `tl.without_op` (grad_fns without a paired forward op;
  the old `tl.intervening` spelling is a deprecated alias that warns).

## Intervention helpers

**Value helpers**
: `tl.zero_ablate`, `tl.scale`, `tl.add`, `tl.clamp`, `tl.noise` (Gaussian noise),
  `tl.mean_ablate` (replace with a source mean), `tl.resample_ablate` (sample replacement
  values from a source tensor), `tl.replace_with` (fixed value), `tl.swap_with` (another
  site's tensor), `tl.steer` (add a scaled steering direction), `tl.project_onto` /
  `tl.project_off` (keep or remove the component along a direction), and
  `tl.splice_module` (call a module as a black-box forward splice). Availability outside
  the torch backend is narrower; see the per-backend rosters in the backends guide.

**Backward helpers**
: `tl.bwd_hook` builds a live/rerun-only backward hook; `tl.grad_zero`, `tl.grad_scale`,
  `tl.grad_clamp`, `tl.grad_clip`, and `tl.grad_noise` edit gradient tensors during the
  backward pass.

**Replay verbs**
: `tl.do` applies a one-shot intervention to a captured log; `tl.push` pushes an edit
  downstream through the recorded graph (DAG replay) and `tl.push_from` pushes from a
  pre-mutated site; `tl.run` performs a full-forward run with the log's active
  intervention spec; `tl.sweep` captures one intervened trace per swept replacement
  value. `tl.replay`, `tl.replay_from`, and `tl.rerun` are deprecated aliases of `push`,
  `push_from`, and `run`.

## Extraction, observers, and admin

**Extraction helpers**
: `tl.pluck` returns the saved out for one layer, `tl.extract` for many layers, and
  `tl.extract_dataset` extracts outs from an iterable dataset in batches. `tl.peek` and
  `tl.batched_extract` are deprecated aliases that warn.

**Observers**
: `tl.tap` creates a tap observer for a site; `tl.span` records a named observer span
  around captures or hook execution (`tl.record_span` is its deprecated alias);
  `tl.record_kpi_in_graph` records a user KPI on the active capture graph;
  `tl.register_tensor_connection` registers a manual parent-child tensor edge during
  capture; `tl.decide_recording_of_batch` retroactively keeps or discards a captured
  batch log.

**Validation entry**
: `tl.validate(model, x, scope=...)` validates a model/input pair for a requested scope
  (for example `"saved"` or `"receptive_field"`), capturing what it needs itself.

**Session admin**
: `tl.release_model` releases a traced model from persistent TorchLens preparation
  (restoring whole-model pickle / `torch.save` serializability); `tl.clear_capture_cache`
  empties the capture cache; `tl.list_logs` / `tl.reset_naming_counter` manage log
  bookkeeping.

## Persistence, containers, and namespaces

**Save / load**
: `tl.save` persists a `Trace` into a portable `.tlspec` directory bundle at a chosen
  level; `tl.load` loads a `.tlspec` object with eager tensor materialization.
  `tl.PayloadLoadHints` carries backend-specific payload materialization hints
  (`tl.JaxPayloadLoadHint` is the JAX-specific form).

**Options**
: `tl.options` groups the public option dataclasses: `CaptureOptions`, `SaveOptions`,
  `VisualizationOptions`, `ReplayOptions`, `InterventionOptions`, `StreamingOptions`.

**Structural hash**
: `tl.hash` is the provisional structural-hash namespace; `tl.assert_unchanged` asserts
  a model still matches a pinned address-free structural hash.

**Container registration**
: `tl.register_container` registers a custom container type (flatten/unflatten pair) for
  capture and reconstruction. `tl.Container` is the computed view over a captured Python
  output container.

**Capture-product bases**
: `tl.CapturedRun` is the shared base for uncooked and cooked capture projections;
  `tl.ActivationLookup` is the protocol for raw-label/pass/address activation lookup
  consumers.

**Namespaces**
: `tl.fastlog` is the sparse predicate-recording namespace behind `tl.record`;
  `tl.facets` is the semantic facet namespace (canonical home `torchlens.semantic`);
  `tl.export` holds static export helpers; `tl.bundle(...)` / `tl.Bundle` build aligned
  Trace collections and `tl.show_bundle_graph` renders a bundle's graph.

## Quantities and typed errors

**Quantities**
: `tl.Quantity` is the marker base for numeric quantity wrappers with unit-aware
  display: `tl.Bytes` (memory), `tl.Duration` (seconds), `tl.Flops` (floating-point
  operations), `tl.Macs` (multiply-accumulates).

**Lookup and reentrancy errors**
: `tl.AmbiguousOpLookupError` is raised when a bare Op lookup matches multiple
  pass-qualified Ops; `tl.ReentrantTraceError` when a trace is started while another
  trace is active.

## Distributed capture and cross-rank merging

**Distributed arming**
: `tl.distributed.arm()` opts a process into first-class capture of explicit
  `torch.distributed` collectives (required at process start for MPMD / spawn-rank
  programs; SPMD processes may arm lazily at capture entry). Sharded state (DTensor,
  tensor/pipeline parallel) still refuses typed at capture entry.

**Collective boundary op**
: A captured op whose portable `annotations["collective"]` carries the
  `collective_boundary_v1` payload: correlation key, role-indexed dual geometry, event
  disclosures, witness fields, and lifetime evidence.

**merge_ranks / MergedTrace**
: `tl.merge_ranks([trace_or_path, ...])` stitches N rank-local captures into a
  `MergedTrace` presenter (never a `Trace` subclass) at their explicit collective
  boundaries; loads rerun the derivation and refuse tampered artifacts typed.

**merge_report**
: `tl.merge_report(...)` is the graph-free merge diagnostic that derives alignment and
  consistency verdicts without constructing a merged graph, and never raises on
  conflicts.

**Merge vocabularies**
: `MergeAlignment`, `BoundaryConsistency`, `MergeValueStatus`, and `MergedErrorCode` are
  frozen vocabularies in `torchlens.merged`, release-gated against
  [the merged-trace contract](merged_trace_contract.md).

## Unstable surfaces (documented-unstable; no deprecation shim owed)

Spellings below shipped ahead of their naming-session/S2 ratification under
the megasprint provisional-name protocol: they may rename WITHOUT deprecation
shims, by declared contract. Each carries the same tag at its definition.

**site_key (`Op.site_key`)** — *unstable — no deprecation shim owed*
: The portable structural-position identity of one retained op
  (`site_key_v1`): a `"s1|"`-prefixed string of the pass-free module address
  stack, the normalized op type, the output slot, and a 1-based occurrence
  ordinal within one pass-qualified innermost module call instance, minted
  from raw records at grouping time on every backend. Policy-independent
  (identical whether grouping ran, degraded, or was off), process-portable
  (no barcodes, ids, or arg hashes), and a BRIDGING relation: two captures of
  the same program agree on site keys even when their layer labels disagree.
  It proves POSITION, never source identity — cross-capture joins carry a
  per-call-instance cardinality guard, a source-location witness, and a
  three-tier verdict (corroborated / positional / refused). Editing the
  model's `forward()` changes downstream sites: keys bridge captures of the
  SAME program, never a program diff. The `"s1"` prefix makes any future
  re-keying a visible schema event. `FieldPolicy.DROP` under tlspec v7
  (prerelease-registered; persists at the coordinated bump).

**Layer.site_key** — *unstable — no deprecation shim owed*
: The layer's single site key iff every op in the layer shares exactly one.
  A site-SPANNING layer (within-call-instance recurrence such as transformer
  residual-add pairs, or root-context loops) refuses typed
  `layer_site_ambiguous` — read per-pass keys via `.ops[k].site_key`.
  Site-uniformity and pass-uniformity are DIFFERENT axes: a reused-module
  multi-pass layer has one key. Legacy pre-site-key artifacts refuse typed
  `site_key_unavailable`, never a `None` read.

**Layer.site_peers** — *unstable — no deprecation shim owed*
: Layers sharing any of this layer's site keys within the trace (the
  reused-relu cohort surface), computed live per call and never persisted.
  Refuses typed `site_key_unavailable` on keyless layers — a legacy artifact
  never collapses into a `None`-key peer-of-everything.

**Layer.shape_summary** — *unstable — no deprecation shim owed*
: Derived (never persisted) data string summarizing output shapes ACROSS
  PASSES of one layer: `None` for single-pass and shape-uniform layers; one
  varying axis renders `"A->B"` (monotone) or `"A-B"` (min-max); multi-axis
  or rank-varying layers render first-to-last full shapes
  (`"2x64x8x8->2x512x4x4"`). Distinct from the internal module-run fold
  summary (`ModuleRepeatFold.shape_summary`), which summarizes across a
  repeated MODULE RUN. The string legitimately contains `->`; renderers must
  HTML-escape it (escape-at-render, never assert-absence).

**grouping= (trace kwarg) / trace.grouping** — *unstable — no deprecation
shim owed; S2-gated vocabulary*
: Closed-vocabulary grouping-policy knob: `"structural"` (default — today's
  recurrence grouping), `"strict_shapes"` (reserved; refuses typed until its
  own reviewed design lands), `"fold_sites"` (the D1 within-capture folding
  axis; refuses typed on plain captures until an affirmative D1 ruling).
  Unknown values refuse `grouping_invalid`; legal-but-not-entry-legal values
  refuse `grouping_policy_unavailable`. The mirror field `trace.grouping`
  records the requested value. Distinct from the display-only `fold_repeats`
  viz knob, which folds repeated module runs at render time and never
  changes grouping.

**trace.grouping_policy** — *unstable — no deprecation shim owed; S2-gated
vocabulary*
: The persisted, load-validated `grouping_policy_v1` stamp recording HOW the
  trace was grouped: `policy` (the step-7 grouping that actually ran —
  `structural` / `params_only` / later `fold_sites`), `requested` (the knob
  mirror), `folded_sites` and `site_join` (two distinct site-granular axes:
  step-7 folds vs product-layer joins), `detector`, `effective`, and
  `settlement_note`. Loads validate against the exact writer key set, closed
  vocabularies, and coherence rules C1–C8; parse failure or incoherence
  warns once and settles to THE canonical degraded representation
  (`policy="unknown"`, `settlement_note="grouping_stamp_<reason>"`), which
  round-trips byte-stable and stays degraded — verdicts only worsen across
  persistence. Legacy pre-stamp artifacts settle silently to
  `grouping_stamp_legacy`; degraded stamps refuse stamp-consuming
  operations typed. `FieldPolicy.DROP` under tlspec v7
  (prerelease-registered).

**L1 grouping refusal codes** — *unstable — no deprecation shim owed;
S2-gated*
: `layer_site_ambiguous`, `site_key_unavailable`, `grouping_invalid`,
  `grouping_policy_unavailable`.

**structure_only (capture kwarg) / trace.structure_only** — *unstable — no
deprecation shim owed*
: `tl.trace(model, x, capture=CaptureOptions(structure_only=True))` runs the
  capture under the structure-only contract: the op graph, module hierarchy,
  parameter geometry, and per-op shape/dtype are recorded with every
  value-bearing claim a HYPOTHESIS; value payloads are never retained,
  value-requiring consumers refuse typed through
  `torchlens.capture.structure_only.require_structure_only_capability`, and
  value-dependent branches refuse with the user's source line
  (device-neutral). The mirror field `trace.structure_only` declares the
  mode. Capability contract:
  [structure_only_capabilities.md](structure_only_capabilities.md).

**Trace.discharge_against(real_trace)** — *unstable — no deprecation shim owed*
: Discharges a structure-only trace's hypotheses against an ordinary settled
  COMPLETE capture of the same graph. Returns a frozen `StructureDischarge`
  (per-claim table + overall corroborated/refuted verdict, positional join
  licensed by graph-shape digest equality); a REFUTED discharge flips
  hypothesis consumers to typed refusals. Neither trace is mutated.

**StructureClaimStatus (hypothesis / corroborated / refuted)** — *unstable —
no deprecation shim owed*
: The tri-state evidence class of a structure-only trace's value-bearing
  claims; never silently promoted.

**Structure-only refusal codes** — *unstable — no deprecation shim owed;
S2-gated*
: `structure_only_option_conflict`, `structure_only_values_unsupported`,
  `value_dependent_branch_unsupported`, `meta_kernel_unavailable`,
  `structure_only_{save,runnable,replay,validation,backward,episode}_unsupported`,
  `structure_only_refuted_hypothesis`, `structure_only_discharge_precondition`,
  `structure_only_type_invalid`.

**PredicateProtocol** — *unstable — no deprecation shim owed*
: `torchlens.ir.predicate_registry.PredicateProtocol` — the frozen callable
  signature for the capture-lifecycle `save`/`halt`/`until` predicate slots:
  one positional concrete `RecordContext`, returning a normalized decision.
  The S4 seam contract; normative page:
  [predicate_runtime.md](predicate_runtime.md). The `intervene=` and grad
  slots are outside the protocol.

**coerce_predicate(value, \*, slot)** — *unstable — no deprecation shim owed*
: The single documented coercion door for predicate consumers
  (`torchlens.ir.predicate_registry`). Closed value domain {raw callable,
  registered-name str}; raw callables (including `BaseSelector` instances and
  `followed_by` composites) return BY IDENTITY, registered names return a
  slot-aware enforcing wrapper carrying
  `__torchlens_cache_key__ = ("registered", name, version)`. Slot vocabulary
  `save | halt | until` is closed (S2-owned); unknown slots refuse typed.

**Selection / ResolvedSelection / `__selection__`** — *unstable — no
deprecation shim owed (do/Selection spellings slate-ratified subject to D7)*
: `tl.Selection` is the composable selection QUERY — a frozen AST over leaf
  terms (selector / receptive-field box / gradient-RF / facet / param / unit)
  and boolean combinators, trace-independent; `selection.resolve(trace)`
  returns a `tl.ResolvedSelection` — frozen, trace-bound, an ordered tuple of
  `SiteEntry(site_key, mask, provenance)` rows. TWO-LEVEL DENOTATION:
  (touched-site family, selected-element set); zero-mask entries are retained
  first-class; `.empty`/`__bool__` are ELEMENT-level; `bool()` on the QUERY
  type refuses typed (`selection_bool_ambiguous`). Anything region-shaped
  implements `__selection__`; `SiteEntry.mask` returns a FRESH materialization
  (mutation cannot alter the selection). Masks are exact AS SETS; producer
  inexactness rides the closed `provenance.relation` lattice
  (`exact | upper_bound | lower_bound | unknown`). `ResolvedSelection` is
  session-time only, never persisted. Selection kinds `ACT | PARAM | EDGE`
  are a closed vocabulary; mixed kinds refuse `selection_kind_incompatible`.

**Selection operators `| & - ~` (+ reflected)** — *ratified set (slate 5.4);
semantics unstable-documented*
: Same-site operands compose as masks; different-site yields a MULTI-SITE
  selection; `-` never un-touches sites (`fam(A-B) = fam(A)`); `~` is the
  touched-site mask complement (never predicate negation and never
  model-universe). NO `__xor__`: `(a - b) | (b - a)` spells it.
  `BaseSelector` keeps its shipped composite semantics; `selector - selector`
  desugars to `and(a, not(b))`; a selector composed with a region producer
  defers to the Selection algebra.

**tl.units / tl.params / tl.random_selection** — *unstable — no deprecation
shim owed*
: Stage-1 producer constructors: `units(site, indices)` (explicit site +
  index set), `params(name, mask=None)` (named-parameter element region),
  `random_selection(like=, within=, seed=)` (seeded size-matched control
  sampled without replacement inside `within`; too-small populations refuse
  `selection_unresolvable` / `population_too_small`).

**SelectionError / selection refusal codes** — *unstable — no deprecation
shim owed; S2-gated*
: One carrier class (`torchlens.selection.SelectionError`, catalogued in the
  intervention error catalog) for the closed codes
  `selection_trace_mismatch`, `selection_bool_ambiguous`,
  `selection_kind_incompatible`, `selection_unresolvable` (closed reason set
  `site_not_in_trace | value_not_saved | non_tensor_site | no_index_space |
  mask_shape_mismatch | facet_write_mask_unavailable | population_too_small`),
  and `selection_apply_invalid` (stage 2).

**tl.Edit / do(selection, edit)** — *Edit ratified (slate 5.5, subject to D7
default-keep); mask-application semantics documented-unstable*
: `tl.Edit` is the public edit-object type; `HelperSpec` is its deprecated
  alias (stable surface, no removal scheduled). `trace.do(selection, edit)`
  applies an edit to a resolved selection under the NORMATIVE
  MASK-APPLICATION CONTRACT: the edit hook computes its full replacement
  exactly as today (helpers stay mask-oblivious), then the ENGINE applies
  `torch.where(mask, edited, original)` on a FRESH tensor — never in-place
  on, never a view aliasing, the stored capture value. Whole-site masks
  short-circuit the scatter (exactly today's behavior). No broadcasting in
  v1; shape/dtype/device/broadcast mismatches and ineligible sites refuse
  `selection_apply_invalid` (closed reason set
  `shape | dtype | device | broadcast | not_maskable`). Learned-parameter
  edits refuse typed (D3 activation-path narrowing, default keep). Each
  Selection-targeted do() appends an audit record (query repr + resolve
  digest + per-site relations) to `trace.intervention_audit` (DROP-gated,
  session-time under v7).

**tl.patch_from(source_trace)** — *unstable — no deprecation shim owed*
: Edit factory patching targeted sites from another trace's recorded
  post-capture values (activation patching); with a Selection target only
  the selected elements are patched. Portability `opaque_audit`: persisted
  args carry source-trace IDENTITY only (never the Trace, never tensors);
  values bind at do() time session-side; no executable-save path in v1.

**trace.edges / edge substitution** — *unstable — no deprecation shim owed;
S2/S3-gated*
: `trace.edges` returns the dataflow edge family (one `EdgeUseRecord` per
  parent→child occurrence; requires an `intervention_ready` capture, else
  `edge_provenance_unavailable`). The canonical occurrence address is
  `(child_func_call_id, arg_kind, arg_path)`. Edge records lift as EDGE-kind
  selections (whole-edge granularity; `~` complements within the trace's
  edge family). `do(edge_selection, edit)` replaces the value CONSUMED on
  the edge — only the child's consumption changes; `parent.out` stays
  producer truth. Ships on the replay/push engine ONLY
  (`edge_intervention_engine_unsupported` otherwise; the rerun-engine design
  is an escalated named future). Storage fork: the substituted value lives
  in the DROP-gated `Op.edge_substitutions` store (+
  `Op.edge_replacement_stamps`, `FireRecord.edge_address`); capture truth
  (`saved_args`, `out_versions_by_child`, `parent.out`) is retained
  unmodified — the pre-edit snapshot that makes divergence decidable.
  Validation: every tier-(ii) entry must be corroborated (FireRecord +
  stamp) else FAIL; corroborated children are RE-EXECUTED with the
  substituted value spliced at the address and must match (verdict
  `edge_intervention_boundary` — a different check, never no check).
  v7 PERSISTENCE BOUNDARY: an edge-intervened trace refuses
  `edge_intervention_save_unsupported` at ALL four save levels while the
  pre-release switch is inactive (session-only in production until the
  wave-3 bump); the refusal precedes `artifact_save_level_unsupported`.

**TapObserver.values(masked=True)** — *unstable — no deprecation shim owed*
: `tap(resolved_selection)` stores each firing site's mask on the
  `TapRecord`; `values(masked=True)` returns fresh masked copies (selected
  elements). `values()` stays exactly the shipped full-snapshot behavior.

**register_predicate(name, \*, replace=False)** — *unstable — no deprecation
shim owed*
: Registers a plain predicate callable under a name for later
  `coerce_predicate` acceptance; returns the function truly unchanged (no
  attribute stamped, nothing the restricted loader consults). Duplicate user
  names refuse `predicate_name_conflict` without `replace=True`; builtin
  names are never replaceable; name misses at coercion refuse
  `predicate_unregistered`. The registry is INERT until consuming surfaces
  adopt name acceptance.

**color_by (draw kwarg)** — *unstable — no deprecation shim owed; keyword-only*
: The v1 encoding-channel value source on `Trace.draw` (L5 channel core). See
  the "color_by" entry above for semantics.

**size_by / scale (draw kwargs)** — *unstable — no deprecation shim owed; keyword-only*
: The wave-1 size encoding channel on `Trace.draw` (D4 default-applied: sqrt +
  conservative area-only mapping + typed refusal on rolled varying sources).
  See the "size_by" and "scale (size channel)" entries above for semantics.

**show_legend=None AUTO value** — *unstable — no deprecation shim owed*
: The tri-state AUTO value on the stable `show_legend` kwarg: no legend unless
  an encoding channel is active, then a channel-only disclosure legend.
  `True`/`False` keep their stable historical meanings.

**Encoding refusal codes** — *unstable — no deprecation shim owed*
: `encoding_source_invalid`, `encoding_value_invalid`,
  `encoding_callable_error`, `encoding_requires_dot_layout`,
  `size_by_rolled_varying`, `scale_requires_size_by`, `encoding_scale_invalid`.
