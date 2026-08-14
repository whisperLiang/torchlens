# backends/torch/ - Agent Notes

## Wrapper Boundaries
- `wrappers.py` owns persistent torch/function decoration. `_logging_enabled` must stay
  the runtime gate; wrappers remain installed after first capture.
- `torch.func` / functorch transform builders return boundary callables. They should attach
  transform metadata, lazy source locations, and replay callables without tracing inside the
  transformed function.
- Direct-call transform wrappers such as `torch.autograd.functional.jacobian` follow the same
  boundary-node contract.

## Provenance
- `model_prep.py` tags registered buffers and plain module tensor attributes with buffer
  addresses before capture.
- `ops.py` records `unattributed_tensor_args` only for tensor arguments with no TorchLens
  input/op/buffer provenance. This is warn-first diagnostics, not a parent-edge substitute.
- Foreign tensors in output position keep existing output binding behavior and should not
  create provenance warnings.

## Journal Producer (single since producer unification P7)
- Every torch capture freezes decomposed `OpRecord` rows (`OpCore` + typed facets) through
  the ONE commit tail `capture/projections.py::commit_op` (freeze -> atomic append): the
  three exhaustive `_make_layer_log_entry` sites via `ExhaustiveOpDraft` and the sparse
  `append_projected_event` sites via `SparseOpDraft`. The legacy `OpEvent` producer, its
  dual-path env switch, `_op_event_from_log`, and
  `_event_from_record` were deleted in P7.
- Post-commit knowledge never mutates the op lane: it rides the typed `OpAmendment` lane
  (`CaptureEvents.append_amendment`, nine exact-set families) and folds through the ONE
  reducer `amended_op_records()`. grad-fn handles live ONLY in the journal side index
  (`grad_fn_handles_by_label_raw`); records never carry them.
- Preview backends keep emitting compat `OpEvent`s until S15 and adapt at the one ingest
  boundary (`op_record_from_event`); `OpEvent`, `PATH_TO_FLAT` (amendment fold guard),
  and `_clone_op_event_for_replay` are retained-with-schedule (S15) and guard-tested in
  `tests/producer_parity/test_p7_single_producer.py`.
