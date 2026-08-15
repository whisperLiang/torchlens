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
  float in `[0, 1]` never changes the underlying Trace.

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
