# Error refusal contract

User-reachable TorchLens refusals are subclasses of `torchlens.errors.TorchLensError` and retain
their historical built-in exception compatibility where applicable. Callers should branch on
`exc.fields["code"]`, never on message text. Each refusal covered by this contract also carries a
non-empty `exc.fields["remedy"]`, and its human-readable message ends with that remedy.

The argument and capture-context classes are resolved lazily from `torchlens.errors`; they do not
add names to the top-level `torchlens` namespace:

- `InvalidArgumentError(ConfigurationError, ValueError)`
- `ArgumentTypeError(ConfigurationError, TypeError)`
- `ArgumentConflictError(ConfigurationError, TypeError)`
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
| `ambiguous_op_lookup` | Accessor key matches multiple pass-qualified objects | Use a full address, pass label, or call index |
| `auto_environment_unsupported` | `TORCHLENS_AUTO=1` requested implicit capture | Unset it and call `auto_capture()` |
| `backward_capture_conflict` | `save_grads` conflicts with `backward_ready=False` | Enable or omit `backward_ready` |
| `batch_items_invalid` | Export batch-items count is negative | Pass a non-negative integer |
| `backend_ambiguity` | Auto-resolution found multiple equal backend matches | Pass `backend=` explicitly |
| `backend_capability_conformance` | Advertised backend capability has no implementation | Disable it or register its implementation |
| `backend_error` | Generic backend registry request failed | Pass a compatible registered backend |
| `backend_mismatch` | Explicit backend cannot handle the model or inputs | Select the owning backend |
| `backend_payload_unsupported` | Backend payload has no supported codec | Save metadata only or use another backend |
| `backend_runtime_compatibility` | Runtime cannot materialize serialized backend data | Install a compatible runtime or analyze only |
| `backend_unsupported` | Backend does not implement the requested capability | Omit it or use another backend |
| `buffer_visibility_invalid` | Unsupported `show_buffers` value | Choose a documented visibility policy |
| `capture_context_required` | Capture-only helper called outside `trace()` | Call it from the captured forward |
| `collapse_level_invalid` | Float collapse level is outside `[0, 1]` | Choose an in-range level |
| `collapse_mode_invalid` | Collapse mode is unsupported | Choose `none`, `auto`, `max`, or a float |
| `collapse_plan_unavailable` | Collapse optimizer declined the render context | Use a supported render context and mode |
| `container_leaf_not_saved` | Container leaf value was not retained | Re-run with `save=` covering the leaves |
| `container_not_reconstructable` | Container spec or backend support is absent | Capture with `capture_container_structure=True` |
| `container_selector_requires_registry` | Snapshot selector on a non-registry view | Call `reconstruct()` without site/role |
| `container_selector_unresolved` | Container selector matched zero or many records | Pass a more specific `site=`/`role=` |
| `container_value_source_invalid` | Container value source is unknown | Pass `values='out'` or `'transformed'` |
| `decoded_output_not_classification` | Decoded output is not a batch top-k table | Capture with classification output decoding |
| `decoded_output_unavailable` | Logits were not retained for re-decoding | Capture with retained logits or lower `top_n` |
| `derived_field_assignment_invalid` | Assignment to a derived compatibility field | Do not assign derived fields |
| `compiled_callable_unsupported` | Compiled plain callable has no module capture surface | Pass the original eager module |
| `deprecated_argument_conflict` | Deprecated and replacement arguments were both supplied | Remove the deprecated argument |
| `diagnostic_severity_invalid` | Diagnostic severity is outside the closed vocabulary | Choose a documented severity |
| `distributed_payload_witness_unsupported` | Payload witnesses are reserved | Use digest witnesses |
| `distributed_witness_invalid` | Distributed witness mode is unknown | Choose `none` or `digest` |
| `fold_repeats_invalid` | Repeat-fold policy is invalid | Choose `None`, `True`, or `False` |
| `gradient_not_saved` | Requested gradient payload was not retained | Capture with gradient saving enabled |
| `gradient_pass_ambiguous` | Gradient query spans multiple backward passes | Pick one pass or record positionally |
| `halt_predicate_type_invalid` | `halt` is not callable | Pass a predicate or `None` |
| `intervention_predicate_type_invalid` | `intervene` is not callable | Pass `tl.when(...)`, another predicate, or `None` |
| `intervention_action_type_invalid` | Intervention action has an unsupported type | Pass a decision, helper, callable, or `None` |
| `intervention_direction_invalid` | Intervention direction is unknown | Choose `forward`, `backward`, or `both` |
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
| `metric_name_invalid` | Intervention metric name is unknown | Choose a registered metric or callable |
| `metric_shape_mismatch` | Metric operands have different element counts | Pass equal-size operands |
| `metric_tensor_type_invalid` | Metric operand is not a tensor | Pass tensor operands |
| `metric_type_invalid` | Metric selector is neither a name nor callable | Pass a registered name or callable |
| `module_call_ambiguous` | Single-call accessor on a multi-call module | Access one call via `module.calls[N]` |
| `module_containment_engine_invalid` | Internal containment engine value is unknown | Choose a supported engine |
| `op_lookup_index_out_of_range` | Integer layer index is out of range | Pass an in-range index |
| `op_lookup_not_found` | Lookup key matches no layer, op, or module | Use a valid label, index, or address |
| `op_lookup_pass_out_of_range` | Pass qualifier exceeds the recorded pass count | Specify a lower pass number |
| `op_lookup_pass_required` | Bare label names a multi-pass layer | Append a pass qualifier such as `:2` |
| `option_group_conflict` | Grouped and flat options set the same field | Use one option style |
| `option_group_type_invalid` | Grouped option has the wrong object type | Pass the documented options class |
| `output_device_invalid` | Output device policy is unknown | Choose `same`, `cpu`, or `cuda` |
| `record_not_bound` | Record's owning Trace reference is gone | Keep the owning Trace alive |
| `relation_assignment_type_invalid` | Finished relation field assigned a non-container | Assign list/set/tuple/frozenset or None |
| `run_fast_divergence_policy_invalid` | `fast=True` with a non-raise divergence policy | Use `on_divergence='raise'` or drop `fast=` |
| `run_fast_requires_inputs` | `fast=True` on the legacy run surface | Call `trace.run(inputs=..., fast=True)` |
| `run_legacy_arguments_conflict` | Unified and legacy run arguments were mixed | Pass one input form only |
| `run_legacy_options_conflict` | Unified run received legacy rerun options | Drop the legacy options |
| `run_source_model_collected` | Live model reference is no longer retained | Pass the model to `trace.run(model, input)` |
| `output_sink_conflict` | Disk storage and callback sink were both configured | Choose one sink |
| `save_mode_invalid` | Activation save mode is unknown | Choose a documented save mode |
| `save_payload_level_conflict` | Optional payload family requires runnable level | Use runnable level or omit that family |
| `selector_function_pattern_type_invalid` | `func()` pattern is not a string | Pass a function-name string |
| `stack_ordinals_duplicate` | Stacked ops share an execution ordinal | Narrow the selector to distinct ops |
| `stack_ordinals_unavailable` | Matched ops lack recorded execution ordinals | Select ops with recorded ordinals |
| `stack_output_not_tensor` | Stacked op's saved primary out is not one tensor | Select single-tensor-output ops |
| `stack_selector_no_match` | `trace.stack` selector matched no sites | Pass a selector matching saved ops |
| `stack_shape_mismatch` | Stacked outputs have different shapes | Select ops with identical output shapes |
| `storage_argument_conflict` | `storage` and `streaming` were both supplied | Prefer `storage`, or remove it |
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
| `visualization_intervention_mode_invalid` | Intervention rendering mode is unknown | Choose `node_mark` or `as_node` |
| `visualization_layout_invalid` | Visualization layout is unknown | Choose `auto`, `dot`, or `rank` |
| `visualization_mode_invalid` | Backend visualization mode is unsupported | Choose a backend-supported mode |
| `visualization_node_style_invalid` | Node style is unknown | Choose a documented style |

Adding or renaming a code is a public vocabulary change and must update this table and the
corresponding typed-door test in the same change.
