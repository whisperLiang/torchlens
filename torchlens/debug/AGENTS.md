# debug/ - Implementation Guide

Power-user diagnostics for completed traces. Imported lazily as `tl.debug` and
deliberately NOT in the top-level `__all__`. Everything here is read-only analysis;
nothing mutates a trace or the capture pipeline.

## _nan.py
- `find_nan(model, x, **trace_kwargs)` captures fresh and localizes the first
  nonfinite value; `find_nan_in_trace(trace)` / `bisect_nan(trace)` work on an
  existing trace.
- Results are `FindNanResult` / `BisectNanResult`; unsaved ancestors become an
  explicit `uncertainty_zone`, never a silent guess.

## _cost.py
- `hot_path(trace, by="flops")` ranks ops by cost metric into a pandas DataFrame.
- `theoretical_op_bytes(op)` estimates input/parameter traffic from metadata only.

## _compile_counter.py
- `count_compiles()` is a context manager yielding `CompileCounts`; raises
  `CompileCountsUnavailableError` when Dynamo counters are not available.

## _graph_breaks.py
- `graph_breaks(model, x, **trace_kwargs)` returns a `GraphBreakReport` of
  `GraphBreak` records correlated back to trace op locations.
- Snapshots and restores model state around its probe run; typed failures are
  `GraphBreaksUnavailableError` / `GraphBreaksNormalizationError`.

## _audit.py
- `audit_trace(trace)` returns a `TraceAudit` of `AuditFinding`s; accepts a
  `PartialTrace` and routes it through `_audit_partial_trace()`.

## _graph.py
- `lineage(...)` -> `LineageResult`; `compare(...)` diffs two traces;
  `dead_neurons(trace, dim=..., threshold=...)` scans saved activations.

## _dtype_range.py
- `dtype_range_audit(...)` -> `DTypeRangeAudit`: overflow headroom, subnormal
  fractions, and narrowing casts per op.

## _gradients.py
- `gradient_flow_audit(...)` builds a gradient-magnitude DataFrame; returns an
  annotated empty frame with a message rather than raising when grads are absent.

## _infer_input_shape.py
- `infer_input_shape(model, **kwargs)` -> `InferInputShapeResult`; iterative
  probe-and-refine search driven by shape-error parsing and module priors.

## _recompute.py
- `recompute_candidates(trace, budget_gb=...)` ranks ops worth recomputing vs saving.

## _common.py
- Shared private helpers (`_ordered_ops`, `_resolve_op`, `_require_pandas`, ...).

## Gotchas
- pandas is required lazily per call (`_require_pandas()`), not at import.
- Several helpers (`find_nan`, `graph_breaks`, `infer_input_shape`) run fresh
  forwards/captures; they are not safe on models with destructive side effects.
- Keep result objects typed dataclasses; callers branch on fields, not messages.
