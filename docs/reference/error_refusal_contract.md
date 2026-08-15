# Error refusal contract

User-reachable TorchLens refusals are subclasses of `torchlens.errors.TorchLensError` and retain
their historical built-in exception compatibility where applicable. Callers should branch on
`exc.fields["code"]`, never on message text. Each refusal covered by this contract also carries a
non-empty `exc.fields["remedy"]`, and its human-readable message ends with that remedy.

The argument and capture-context classes are resolved lazily from `torchlens.errors`; they do not
add names to the top-level `torchlens` namespace:

- `InvalidArgumentError(ConfigurationError, ValueError)`
- `ArgumentTypeError(ConfigurationError, TypeError)`
- `ArgumentConflictError(ConfigurationError, ValueError)` — VALUE-combination
  conflicts (well-typed options whose values are mutually exclusive, e.g.
  `save_outs_to` + `out_sink`) were historically raw `ValueError`s; the typed
  refusal preserves that catchability.
- `KeywordConflictError(ConfigurationError, TypeError)` — KEYWORD/call-surface
  conflicts (deprecated kwarg + replacement, grouped option + flat field, two
  exclusive call surfaces) were historically raw `TypeError`s (Python's
  duplicate-keyword convention); the typed refusal preserves that catchability.
  The split is site-by-site per git history, never a blanket base.
- `CaptureContextError(CaptureError, RuntimeError)`
- `DiagnosticSeverityError(ConfigurationError, ValueError)`
- `PayloadUnavailableError(CaptureError, ValueError)`
- `RecordBindingError(CaptureError, RuntimeError)`

## Stable codes

| Code | Refusal | Remedy class |
|---|---|---|
| `activation_not_saved` | Requested op activation was not retained by `save=` | Re-run the capture with `save=` covering the op |
| `annotation_backend_unsupported` | Tensor annotation on a non-torch trace | Pass JSON-serializable data instead |
| `annotation_namespace_invalid` | `annotations["user"]` is no longer a dict | Restore the user namespace to a dict |
| `annotation_not_json_serializable` | Annotation data is neither JSON nor a tensor | Convert arrays to `torch.Tensor` |
| `annotation_payload_missing` | `annotate()` received neither data nor image | Pass `data=`, `image=`, or both |
| `annotation_tensor_not_portable` | Annotation tensor fails the payload codec | Pass a dense, codec-supported tensor |
| `artifact_kind_mismatch` | Specialized loader received another artifact kind | Use the matching loader or generic `io.load` |
| `artifact_save_level_invalid` | `.tlspec` save level is unknown | Choose a documented save level |
| `artifact_save_level_unsupported` | Artifact kind cannot provide the requested save level | Choose a level supported by that kind |
| `artifact_version_below_floor` | Artifact predates the rehydration floor (`tlspec_version` < 6 / torchlens < 2.33) | Load and re-save it with a torchlens release that still reads it |
| `ambiguous_op_lookup` | Accessor key matches multiple pass-qualified objects | Use a full address, pass label, or call index |
| `auto_environment_unsupported` | `TORCHLENS_AUTO=1` requested implicit capture | Unset it and call `auto_capture()` |
| `backward_capture_conflict` | `save_grads` conflicts with `backward_ready=False` | Enable or omit `backward_ready` |
| `backward_graph_unavailable` | Backward drawing without a captured backward graph | Call `log_backward(loss)` first |
| `backward_pass_filter_invalid` | Backward pass filter is not a positive one-based int | Pass a positive pass number |
| `batch_items_invalid` | Export batch-items count is negative | Pass a non-negative integer |
| `batch_render_invalid` | Batch render policy is unknown or malformed | Choose a documented batch_render policy |
| `backend_ambiguity` | Auto-resolution found multiple equal backend matches | Pass `backend=` explicitly |
| `backend_capability_conformance` | Advertised backend capability has no implementation | Disable it or register its implementation |
| `backend_error` | Base-class default of the backend registry family — never raised directly; every live registry refusal carries one of the specific `backend_*`/`unknown_backend` codes below | Branch on the specific backend codes; this row exists only so an unmigrated future subclass is still documented |
| `backend_mismatch` | Explicit backend cannot handle the model or inputs | Select the owning backend |
| `backend_payload_unsupported` | Backend payload has no supported codec | Save metadata only or use another backend |
| `backend_runtime_compatibility` | Runtime cannot materialize serialized backend data | Install a compatible runtime or analyze only |
| `backend_unsupported` | Backend does not implement the requested capability | Omit it or use another backend |
| `buffer_visibility_invalid` | Unsupported `show_buffers` value | Choose a documented visibility policy |
| `bundle_member_payload_missing` | Bundle member retained no tensor at this node | Query a node with stored tensors |
| `bundle_member_unknown` | Bundle member name is not in the bundle | Pass a known member name |
| `bundle_shape_mismatch` | Bundle members have incompatible shapes | Query members with matching shapes |
| `bundle_stack_incomplete` | Not every bundle member retained a tensor | Stack where every member has a tensor |
| `bundle_diff_layout_invalid` | Bundle diff layout is unsupported | Pass `layout='paired'` |
| `bundle_diff_members_invalid` | Bundle diff sides are missing or identical | Name two distinct members |
| `bundle_statistic_invalid` | Bundle statistic is unknown | Choose `mean`, `std`, `var`, or `norm` |
| `code_panel_callable_return_invalid` | Callable code panel returned a non-string at render time (`ArgumentTypeError`) | Return the panel text as a string |
| `code_panel_model_collected` | Callable code panel needs the live model | Use a built-in code_panel mode |
| `code_panel_option_invalid` | Code panel mode literal is unknown (`InvalidArgumentError`) | Pass a documented mode or a callable |
| `code_panel_side_invalid` | Code panel side is unknown | Pass `side='right'` or `'left'` |
| `custom_callable_import_path_missing` | Custom function registry key lacks its `import_path` reference (`InvalidArgumentError`) | Supply `import_path='module:qualname'` on the registry key entry |
| `dagua_renderer_not_opted_in` | Experimental dagua renderer used without opt-in | Import `torchlens.experimental.dagua` first |
| `capture_context_required` | Capture-only helper called outside `trace()` | Call it from the captured forward |
| `child_process_capture_unsupported` | Capture attempted from a non-deliberate child process | Capture from the owning process, or an initialized SPMD rank |
| `unwrap_during_active_capture` | `unwrap_torch()` called while a capture is running | Finish or abort the capture before unwrapping |
| `collapse_level_invalid` | Float collapse level is outside `[0, 1]` | Choose an in-range level |
| `collapse_mode_invalid` | Collapse mode is unsupported — PER-SURFACE DOMAINS: rendering `collapse=` accepts `none`/`auto`/`max`/float, `collapse_plan(mode=)` accepts `auto`/`max`/float, `collapse_order(mode=)` accepts only `auto`/`max` | Choose a mode documented for that surface; the raised remedy names the exact set |
| `collapse_plan_unavailable` | Collapse optimizer declined the render context | Use a supported render context and mode |
| `container_leaf_not_saved` | Container leaf value was not retained | Re-run with `save=` covering the leaves |
| `container_not_reconstructable` | Container spec or backend support is absent | Capture with `capture_container_structure=True` |
| `container_selector_requires_registry` | Snapshot selector on a non-registry view | Call `reconstruct()` without site/role |
| `container_selector_unresolved` | Container selector matched zero or many records | Pass a more specific `site=`/`role=` |
| `container_spec_inadmissible` | Recorded output-container spec is corrupt, tampered, or names an inadmissible type (`ContainerReconstructionError`, `ValueError` lineage; default-deny security tripwire, also raised by public `Op.multi_output_type`) | Re-save the artifact from a trusted capture |
| `container_value_source_invalid` | Container value source is unknown | Pass `values='out'` or `'transformed'` |
| `context_field_invalid` | Persisted execution-context field fails its closed-vocabulary parse | Re-export the artifact; do not hand-edit descriptor context fields |
| `decoded_output_not_classification` | Decoded output is not a batch top-k table | Capture with classification output decoding |
| `decoded_output_unavailable` | Logits were not retained for re-decoding | Capture with retained logits or lower `top_n` |
| `derived_field_assignment_invalid` | Assignment to a derived compatibility field | Do not assign derived fields |
| `compiled_callable_unsupported` | Compiled plain callable has no module capture surface | Pass the original eager module |
| `compile_counts_unavailable` | `tl.debug.count_compiles()` found no Dynamo compile counters in this torch runtime (`CompileCountsUnavailableError`, `RuntimeError` lineage) | Upgrade torch or skip the verification on this runtime |
| `deprecated_argument_conflict` | Deprecated and replacement arguments were both supplied | Remove the deprecated argument |
| `diagnostic_severity_invalid` | Diagnostic severity is outside the closed vocabulary | Choose a documented severity |
| `distributed_payload_witness_unsupported` | Payload witnesses are reserved | Use digest witnesses |
| `distributed_witness_invalid` | Distributed witness mode is unknown | Choose `none` or `digest` |
| `error_constructor_args_conflict` | Diagnostic constructor got message args and fields | Pass a message or named fields, not both |
| `fold_repeats_invalid` | Repeat-fold policy is invalid | Choose `None`, `True`, or `False` |
| `fsdp_capture_unsupported` | `record()` received an FSDP-wrapped model | Record the unsharded module |
| `import_path_invalid` | Custom-callable import reference is malformed | Use the `module:qualname` form |
| `intervening_cluster_invalid` | Intervening-cluster policy is unknown | Choose `upstream`, `outside`, `downstream`, or `own` |
| `intervention_tensor_unsupported` | Intervention save tensor fails the codec | Use dense, codec-supported tensors |
| `layers_not_logged` | Rendering requires a fully-logged trace | Capture with full layer logging |
| `history_size_invalid` | Recorder history size is out of range | Pass an integer in `[0, 1024]` |
| `gradient_not_saved` | Requested gradient payload was not retained | Capture with gradient saving enabled |
| `graph_breaks_normalization_failed` | `tl.debug.graph_breaks()` got a Dynamo explain result of unrecognized shape (`GraphBreaksNormalizationError`, `RuntimeError` lineage) | Report the shape to TorchLens or pin a recognized torch version |
| `graph_breaks_unavailable` | `tl.debug.graph_breaks()` found no `torch._dynamo.explain` in this torch runtime (`GraphBreaksUnavailableError`, `RuntimeError` lineage) | Upgrade torch or skip the correlation on this runtime |
| `graphviz_render_failed` | Graphviz did not produce a usable rendered artifact (`GraphvizRenderError`, `RuntimeError` lineage) | Lower dpi, render direct SVG, or cap the graph size |
| `gradient_pass_ambiguous` | Gradient query spans multiple backward passes | Pick one pass or record positionally |
| `halt_predicate_type_invalid` | `halt` is not callable — MULTICLASS BY SURFACE: `ArgumentTypeError` (`TypeError`) on `tl.trace`, `InvalidArgumentError` (`ValueError`) on `tl.record`, each faithful to its site history | Pass a predicate or `None` |
| `hash_content_type_unsupported` | `tl.hash.content` value cannot be deterministically encoded (`ArgumentTypeError`, `TypeError` lineage) | Pass tensors, arrays, builtin scalars/containers, or `__dict__`-inspectable objects |
| `hash_expected_type_invalid` | `tl.assert_unchanged` pin is neither a string nor `None` (`ArgumentTypeError`, `TypeError` lineage) | Pass the pinned hash string, or `None` to bootstrap a pin |
| `input_tree_cycle` | Model-input tree contains a self-referential container (`InvalidArgumentError`) | Remove the container reference cycle from the model input |
| `input_tree_depth_exceeded` | Model-input tree nesting exceeds the input-boundary depth ceiling (`InvalidArgumentError`) | Flatten the nested input containers before tracing |
| `input_tree_stack_exhausted` | Walking the model-input tree exhausted the Python stack budget before the depth ceiling — capture was entered with most of the interpreter stack already consumed (`InvalidArgumentError`) | Enter capture from a shallower call stack or raise `sys.setrecursionlimit()` |
| `intervention_predicate_type_invalid` | `intervene` is not callable — MULTICLASS BY SURFACE: `ArgumentTypeError` (`TypeError`) on `tl.trace`, `InvalidArgumentError` (`ValueError`) on `tl.record`, each faithful to its site history | Pass `tl.when(...)`, another predicate, or `None` |
| `intervention_action_direction_invalid` | Predicate-side intervention action names an unknown direction (`ArgumentTypeError`; historically `TypeError`, so the live capture path converts it to `PredicateError`) | Choose `forward`, `backward`, or `both` |
| `intervention_action_type_invalid` | Intervention action has an unsupported type | Pass a decision, helper, callable, or `None` |
| `intervention_direction_invalid` | Trace-side intervention direction is unknown (`InvalidArgumentError`; historically `ValueError`) | Choose `forward`, `backward`, or `both` |
| `intervention_engine_invalid` | `do(..., engine=...)` value is unknown | Choose `auto`, `replay`, `rerun`, or `set_only` |
| `intervention_helper_unknown` | Built-in helper name is unknown | Choose a registered helper |
| `jax_control_flow_invalid` | JAX control-flow mode is unknown | Choose `reject`, `unroll`, or `region` |
| `layer_pass_ambiguous` | Per-pass field read on a multi-pass layer | Access the field on one pass via `.ops[k]` |
| `link_format_invalid` | Source-link format is unknown | Choose `terminal`, `html`, or `text` |
| `jax_unroll_range_invalid` | JAX unroll limit is below one | Pass a positive integer |
| `jax_unroll_type_invalid` | JAX unroll limit is not an integer | Pass a positive integer |
| `lookback_invalid` | Lookback is not an integer in `[0, 1024]` | Pass an in-range integer |
| `lookback_payload_policy_invalid` | Lookback payload policy is unknown | Choose a documented payload policy |
| `model_type_unsupported` | Torch capture model is not an `nn.Module` | Pass a module or select its backend |
| `max_pairs_invalid` | Bundle diff pair budget is below one | Pass `max_pairs >= 1` or None |
| `max_predicate_failures_invalid` | Predicate failure budget is not a non-negative int | Pass a non-negative integer |
| `module_focus_empty` | Focused module contains no rendered layers | Focus a module with layers |
| `module_focus_invalid` | Module focus value has the wrong kind or owner | Pass an owned Module or address string |
| `module_focus_not_found` | Module focus address is not in the trace | Pass an existing module address |
| `metric_name_invalid` | Intervention metric name is unknown | Choose a registered metric or callable |
| `metric_shape_mismatch` | Metric operands have different element counts | Pass equal-size operands |
| `metric_tensor_type_invalid` | Metric operand is not a tensor | Pass tensor operands |
| `metric_type_invalid` | Metric selector is neither a name nor callable | Pass a registered name or callable |
| `module_call_ambiguous` | Single-call accessor on a multi-call module | Access one call via `module.calls[N]` |
| `node_label_field_invalid` | Node label field name is unknown | Pass documented label field names |
| `node_overlay_invalid` | Node overlay name is unknown | Choose a supported overlay |
| `op_lookup_index_out_of_range` | Integer layer index is out of range | Pass an in-range index |
| `op_lookup_not_found` | Lookup key matches no layer, op, or module | Use a valid label, index, or address |
| `op_lookup_pass_out_of_range` | Pass qualifier exceeds the recorded pass count | Specify a lower pass number |
| `op_lookup_pass_required` | Bare label names a multi-pass layer | Append a pass qualifier such as `:2` |
| `on_forward_error_invalid` | Forward-error policy is unknown | Choose `raise`, `attach_partial`, or `return_partial` |
| `on_predicate_error_invalid` | Predicate-error policy is unknown | Choose `auto`, `accumulate`, or `fail-fast` |
| `option_group_conflict` | Grouped and flat options set the same field on a merge entrypoint (`ArgumentConflictError`; historically `ValueError`) | Use one option style |
| `option_group_keyword_conflict` | Flat draw kwarg and `VisualizationOptions` field set the same option (`KeywordConflictError`; historically `TypeError`) | Use one option style |
| `option_group_type_invalid` | Grouped option has the wrong object type | Pass the documented options class |
| `output_device_invalid` | Output device policy is unknown | Choose `same`, `cpu`, or `cuda` |
| `output_tree_cycle` | Model-output tree contains a self-referential container, so its occurrence-weighted tensor sum is undefined in `validate_backward_pass` (`InvalidArgumentError`) | Remove the container reference cycle from the model output |
| `output_tree_depth_exceeded` | Model-output tree nesting exceeds the output-boundary depth ceiling in `validate_backward_pass` (`InvalidArgumentError`) | Flatten the nested output containers before validating |
| `record_not_bound` | Record's owning Trace reference is gone | Keep the owning Trace alive |
| `recording_events_not_retained` | `to_trace()` on a disk-recovered Recording | Convert the in-session Recording |
| `recording_backward_halted` | `log_backward()` on a halted Recording (the sparse frontier retained no complete output to root the backward walk) | Re-record without `halt=` or use `trace(halt=...)` for prefix backward |
| `recording_failed_not_convertible` | `to_trace()` on a failed partial Recording | Fix the forward and re-record |
| `recording_halt_frontier_missing` | Halted Recording retained no frontier payload | Save the halt frontier or use `trace(halt=...)` |
| `recording_multipass_not_convertible` | `to_trace()` on a multi-pass Recording | Record one pass per Recording |
| `reentrant_trace` | `tl.trace` was started while another capture was active (`ReentrantTraceError`, `RuntimeError` lineage) | Finish the outer capture before starting another |
| `recording_option_duplicate` | Recording option was specified twice | Pass each option exactly once |
| `recording_option_type_invalid` | Recording option has an unsupported type | Pass the documented type for that option |
| `relation_assignment_type_invalid` | Finished relation field assigned a non-container | Assign list/set/tuple/frozenset or None |
| `renderer_capability_unsupported` | RenderIR requires a capability its renderer lacks (`UnsupportedRendererCapabilityError`, `RuntimeError` lineage) | Use the graphviz renderer or drop the option needing the capability |
| `run_fast_divergence_policy_invalid` | `fast=True` with a non-raise divergence policy | Use `on_divergence='raise'` or drop `fast=` |
| `run_input_missing` | Legacy rerun received no forward input | Pass the input as `log.run(model, x)` |
| `run_fast_requires_inputs` | `fast=True` on the legacy run surface | Call `trace.run(inputs=..., fast=True)` |
| `run_legacy_arguments_conflict` | Unified and legacy run arguments were mixed | Pass one input form only |
| `run_legacy_options_conflict` | Unified run received legacy rerun options | Drop the legacy options |
| `run_source_model_collected` | Live model reference is no longer retained | Pass the model to `trace.run(model, input)` |
| `output_sink_conflict` | Disk storage and callback sink were both configured | Choose one sink |
| `save_mode_invalid` | Activation save mode is unknown | Choose a documented save mode |
| `save_predicate_type_invalid` | `save=` is neither SaveOptions, predicate, selector, nor `None` (`save='all'` lands here) | Pass a predicate or SaveOptions; use `layers_to_save='all'` for exhaustive saves |
| `save_payload_level_conflict` | Optional payload family requires runnable level | Use runnable level or omit that family |
| `selector_function_pattern_type_invalid` | `func()` pattern is not a string | Pass a function-name string |
| `skip_fn_boundary_invalid` | `skip_fn` tried to skip an input or output layer | Return False for boundary layers |
| `spec_format_version_unsupported` | Intervention `.tlspec` format version is unknown | Use a supported format version |
| `summary_fields_invalid` | Summary field names are unknown | Pass documented summary fields |
| `summary_level_invalid` | Summary level is unknown | Pass a documented summary level |
| `summary_option_conflict` | Aliased summary options disagree | Pass one alias, or equal values |
| `stack_ordinals_duplicate` | Stacked ops share an execution ordinal | Narrow the selector to distinct ops |
| `stack_ordinals_unavailable` | Matched ops lack recorded execution ordinals | Select ops with recorded ordinals |
| `stack_output_not_tensor` | Stacked op's saved primary out is not one tensor | Select single-tensor-output ops |
| `stack_selector_no_match` | `trace.stack` selector matched no sites | Pass a selector matching saved ops |
| `stack_shape_mismatch` | Stacked outputs have different shapes | Select ops with identical output shapes |
| `storage_argument_conflict` | `storage` and `streaming` were both supplied | Prefer `storage`, or remove it |
| `structural_hash_mismatch` | Model structural hash differs from the pinned value (`StructuralHashMismatchError`, `AssertionError` lineage) | Inspect the captured traces for the divergence, or re-pin if intentional |
| `sweep_intervention_conflict` | `sweep()` received a second intervention | Express the target through `at` |
| `sweep_names_length_mismatch` | Sweep names and values have different lengths | Pass one name per value |
| `sweep_site_missing` | Sweep site target is missing | Pass `at` |
| `sweep_site_type_invalid` | Sweep site target has an unsupported type | Pass a label, selector, or predicate |
| `sweep_values_empty` | Sweep values iterable is empty | Pass at least one value |
| `sweep_values_missing` | Sweep values iterable is missing | Pass a non-empty iterable |
| `tensor_connection_labels_missing` | Manual edge endpoint lacks a capture label | Use tensors already captured in the active trace |
| `top_n_invalid` | Requested `top_n` is below one | Pass a positive integer |
| `trace_not_finished` | Export requested before the forward pass finished | Wait until `trace(...)` has returned |
| `trace_reference_collected` | Owning Trace was garbage-collected | Keep the Trace alive while reading records |
| `unknown_backend` | Explicit backend name is not registered | Choose a registered backend |
| `unsupported_tensor_variant` | Model/input carries meta, fake, functional, or sparse tensor variants (`UnsupportedTensorVariantError`) | Materialize dense, strided tensors with concrete shapes on a real device |
| `visualization_bool_option_invalid` | A bool-only draw/visualization option received a non-bool (strings such as `'no'` were silently truthy) | Pass True or False |
| `visualization_show_containers_invalid` | `show_containers` is outside its closed vocabulary | Pass False or one of `labels`, `cluster`, `collapsed`, `auto`, `nodes` |
| `visualization_intervention_mode_invalid` | Intervention rendering mode is unknown | Choose `node_mark` or `as_node` |
| `visualization_layout_invalid` | Visualization layout is unknown | Choose `auto`, `dot`, or `rank` |
| `visualization_direction_invalid` | Render direction is unknown | Choose `bottomup`, `topdown`, or `leftright` |
| `visualization_renderer_invalid` | Visualization renderer is unknown | Choose `graphviz` or `dagua` |
| `visualization_theme_invalid` | Visualization theme is unknown | Choose a supported theme |
| `visualization_mode_invalid` | Backend visualization mode is unsupported | Choose a backend-supported mode |
| `visualization_node_style_invalid` | Node style is unknown | Choose a documented style |
| `wrappers_removed_before_capture` | A concurrent `unwrap_torch()` removed the torch wrappers between model preparation and capture admission | Do not call `unwrap_torch()` concurrently with capture entry; re-run `tl.trace` to re-install the wrappers |

## Constant-spelled refusal kinds (distributed collective capture)

Three distributed-capture refusals identify themselves on `exc.fields["kind"]` (one
entry per structured finding) rather than `exc.fields["code"]`, and their identifier
strings are spelled as module-level constants rather than inline `code="..."`
literals. They are part of the same stable public vocabulary: branch on the kind
string, never on message text. The lockstep gate enrolls each constant explicitly
(`tests/test_error_contract_lockstep.py`), so renaming the constant or drifting its
string value fails the gate exactly like an inline code.

| Code | Refusal | Remedy class |
|---|---|---|
| `ambiguous_group_lifetime` | A collective used a process group whose pre-arming lifetime cannot be proven | Call `tl.distributed.arm()` at process start, before any group is created |
| `uncaptured_collective_op` | Arm-time recognizer set-inequality or dispatcher schema scan found a collective the wraps would not capture | Upgrade TorchLens to a build whose recognizer covers the installed torch, or avoid the unrecognized collective in the traced forward |
| `wildcard_recv_unsupported` | A point-to-point receive from `ANY_SOURCE` cannot be attributed to a sender | Pass an explicit source rank to `recv`/`irecv` |

The related `group_lifetime_evidence_conflict` kind is governed by the merged-trace
contract (`docs/reference/merged_trace_contract.md`), where it is also a
`MergedErrorCode` member.

Adding or renaming a code is a public vocabulary change and must update this table and the
corresponding typed-door test in the same change. Constant-spelled kinds must additionally
update the enrollment table in `tests/test_error_contract_lockstep.py`.
