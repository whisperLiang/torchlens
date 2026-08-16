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
