"""Plain-language reports for completed TorchLens logs."""

from __future__ import annotations

from collections import Counter
from typing import Any, Literal
import traceback

import torch

from ..data_classes._nonfinite import first_nonfinite_layer, nonfinite_layers

Audience = Literal["researcher", "practitioner", "auto"]
ExplainFormat = Literal["text", "json"]


def explain(
    log: Any,
    audience: Audience = "auto",
    format: ExplainFormat = "text",
) -> str | dict[str, Any]:
    """Explain a TorchLens log in plain language.

    Parameters
    ----------
    log:
        Completed ``Trace``-like object.
    audience:
        Report style. ``"researcher"`` includes graph-pattern detail,
        ``"practitioner"`` emphasizes operational status, and ``"auto"``
        selects a balanced report.
    format:
        ``"text"`` for the existing prose report or ``"json"`` for a
        structured dictionary.

    Returns
    -------
    str | dict[str, Any]
        Multi-section prose, or the stable flat schema documented below.

    Raises
    ------
    ValueError
        If ``audience`` or ``format`` is not supported.

    Notes
    -----
    The JSON schema is identified by ``schema="torchlens.explain.v1"`` and
    contains these stable snake-case keys: ``schema``, ``audience``,
    ``capture_status``, ``model_class``, ``layer_count``, ``operation_count``,
    ``saved_tensor_count``, ``total_tensor_count``, ``has_backward_pass``,
    ``exception_type``, ``exception_message``, ``last_completed_op_label``,
    ``last_completed_op_shape``, ``last_completed_op_dtype``,
    ``last_completed_op_device``, ``failing_boundary``, and
    ``first_nonfinite``. Evidence unavailable from the supplied log is reported
    as ``"unknown"`` rather than inferred.
    """

    if audience not in {"researcher", "practitioner", "auto"}:
        raise ValueError("audience must be 'researcher', 'practitioner', or 'auto'.")
    if format not in {"text", "json"}:
        raise ValueError("format must be 'text' or 'json'.")

    if _is_partial_trace(log):
        diagnosis = _partial_diagnosis(log)
        if format == "json":
            return _partial_json(log, audience, diagnosis)
        return _partial_text(diagnosis)

    if format == "json":
        return _full_json(log, audience)

    lines = [
        "TorchLens report",
        "",
        "Model summary",
        *_model_summary_lines(log),
        "",
        "Capture summary",
        *_capture_summary_lines(log),
        "",
        "Backward summary",
        *_backward_summary_lines(log),
        "",
        "Anomalies",
        *_anomaly_lines(log),
        "",
        "Interventions",
        *_intervention_lines(log),
        "",
        "Notable patterns",
        *_pattern_lines(log, audience=audience),
    ]
    return "\n".join(lines)


def _is_partial_trace(log: Any) -> bool:
    """Return whether ``log`` is a :class:`PartialTrace` instance.

    Parameters
    ----------
    log:
        Candidate capture object.

    Returns
    -------
    bool
        Whether the object is a failed partial capture wrapper.
    """

    from ..partial import PartialTrace

    return isinstance(log, PartialTrace)


def _last_partial_op(log: Any) -> Any | None:
    """Return the last recorded completed operation in a partial capture.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    Any | None
        Last raw operation, or ``None`` when none completed.
    """

    raw_layers = tuple(getattr(log, "raw_layers", ()))
    return raw_layers[-1] if raw_layers else None


def _partial_boundary(log: Any) -> str:
    """Return the nearest evidence-backed failure boundary description.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    str
        Recorded exception field or traceback location, otherwise ``"unknown"``.
    """

    exception = log.original_exception
    fields = getattr(exception, "fields", {})
    if isinstance(fields, dict) and (fields.get("layer") or fields.get("op")):
        return f"layer={fields.get('layer', 'unknown')}, op={fields.get('op', 'unknown')}"
    extracted = traceback.extract_tb(exception.__traceback__)
    forward_frames = [frame for frame in extracted if frame.name == "forward"]
    if forward_frames:
        frame = forward_frames[-1]
        return f"forward at {frame.filename}:{frame.lineno}"
    if extracted:
        frame = extracted[-1]
        return f"{frame.name} at {frame.filename}:{frame.lineno}"
    return "unknown"


def _partial_diagnosis(log: Any) -> dict[str, Any]:
    """Collect evidence-backed fields for a failed partial capture.

    Parameters
    ----------
    log:
        Partial capture wrapper.

    Returns
    -------
    dict[str, Any]
        Last-op, boundary, exception, and non-finite evidence.
    """

    op = _last_partial_op(log)
    output = getattr(op, "out", None) if op is not None else None
    label = (
        str(getattr(op, "_label_raw", getattr(op, "_layer_label_raw", "unknown")))
        if op is not None
        else "unknown"
    )
    shape = getattr(op, "shape", None) if op is not None else None
    dtype = getattr(op, "dtype", None) if op is not None else None
    device = getattr(op, "device", None) if op is not None else None
    if isinstance(output, torch.Tensor):
        shape = tuple(output.shape) if shape is None else shape
        dtype = output.dtype if dtype is None else dtype
        device = output.device if device is None else device
    exception = log.original_exception
    return {
        "last_completed_op_label": label,
        "last_completed_op_shape": tuple(shape) if shape is not None else "unknown",
        "last_completed_op_dtype": str(dtype) if dtype is not None else "unknown",
        "last_completed_op_device": str(device) if device is not None else "unknown",
        "failing_boundary": _partial_boundary(log),
        "exception_type": type(exception).__name__,
        "exception_message": str(exception),
        "first_nonfinite": str(log.first_nonfinite()),
    }


def _partial_text(diagnosis: dict[str, Any]) -> str:
    """Render a partial-capture diagnosis as text.

    Parameters
    ----------
    diagnosis:
        Evidence fields from :func:`_partial_diagnosis`.

    Returns
    -------
    str
        Human-readable failed-capture report.
    """

    return "\n".join(
        [
            "TorchLens report",
            "",
            "Capture status",
            "- This is a partial capture; only operations completed before the failure are known.",
            "",
            "Failure diagnosis",
            (
                "- Last completed op: "
                f"{diagnosis['last_completed_op_label']} "
                f"(shape={diagnosis['last_completed_op_shape']}, "
                f"dtype={diagnosis['last_completed_op_dtype']}, "
                f"device={diagnosis['last_completed_op_device']})."
            ),
            f"- Failing boundary: {diagnosis['failing_boundary']}.",
            (
                "- Captured exception: "
                f"{diagnosis['exception_type']}: {diagnosis['exception_message']}"
            ),
            f"- First non-finite evidence: {diagnosis['first_nonfinite']}",
        ]
    )


def _base_json(log: Any, audience: Audience) -> dict[str, Any]:
    """Return fields common to complete and partial JSON reports.

    Parameters
    ----------
    log:
        Capture object.
    audience:
        Requested audience label.

    Returns
    -------
    dict[str, Any]
        Common stable-schema fields.
    """

    return {
        "schema": "torchlens.explain.v1",
        "audience": audience,
        "capture_status": "complete",
        "model_class": getattr(log, "model_class_name", type(log).__name__),
        "layer_count": _safe_len(getattr(log, "layer_list", None)),
        "operation_count": int(getattr(log, "num_ops", 0) or 0),
        "saved_tensor_count": int(getattr(log, "num_saved_ops", 0) or 0),
        "total_tensor_count": int(getattr(log, "num_tensors", 0) or 0),
        "has_backward_pass": bool(getattr(log, "has_backward_pass", False)),
        "exception_type": "unknown",
        "exception_message": "unknown",
        "last_completed_op_label": "unknown",
        "last_completed_op_shape": "unknown",
        "last_completed_op_dtype": "unknown",
        "last_completed_op_device": "unknown",
        "failing_boundary": "unknown",
        "first_nonfinite": _first_nonfinite_summary(log),
    }


def _full_json(log: Any, audience: Audience) -> dict[str, Any]:
    """Build the stable JSON schema for a completed trace.

    Parameters
    ----------
    log:
        Completed trace.
    audience:
        Requested audience label.

    Returns
    -------
    dict[str, Any]
        Stable full-trace report dictionary.
    """

    return _base_json(log, audience)


def _partial_json(
    log: Any,
    audience: Audience,
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    """Build the stable JSON schema for a partial trace.

    Parameters
    ----------
    log:
        Partial capture wrapper.
    audience:
        Requested audience label.
    diagnosis:
        Evidence fields from :func:`_partial_diagnosis`.

    Returns
    -------
    dict[str, Any]
        Stable failed-capture report dictionary.
    """

    result = _base_json(log, audience)
    result.update(diagnosis)
    result.update(
        {
            "capture_status": "partial",
            "model_class": getattr(log.trace, "model_class_name", type(log.trace).__name__),
            "layer_count": len(log.raw_layers),
            "operation_count": len(log.raw_layers),
            "saved_tensor_count": sum(
                bool(getattr(op, "has_saved_activation", False)) for op in log.raw_layers
            ),
            "total_tensor_count": len(log.raw_layers),
        }
    )
    return result


def _first_nonfinite_summary(log: Any) -> str:
    """Return a saved-output non-finite summary without speculation.

    Parameters
    ----------
    log:
        Completed trace-like object.

    Returns
    -------
    str
        First recorded non-finite detail, or a scoped clean statement.
    """

    layer = first_nonfinite_layer(log, kind="saved")
    if layer is None:
        return "No non-finite values found in saved outputs."
    return _first_nonfinite_detail(log, str(getattr(layer, "layer_label", "unknown")))


def _first_nonfinite_detail(log: Any, saved_label: str) -> str:
    """Return the log's own non-finite detail, or a scoped fallback.

    ``log.first_nonfinite()`` re-asks the trace under its own scan contract and can
    itself raise ``ValueError`` on a selective-save trace (it reads unsaved ``.out``
    payloads, saved or not, and revalidating a memo reads them too). explain() must
    not propagate that: fall back to the scoped saved-output statement so the
    report always honors its ``unknown``/clean contract.

    Parameters
    ----------
    log:
        Completed trace-like object.
    saved_label:
        Label of the saved output already found non-finite.

    Returns
    -------
    str
        First recorded non-finite detail, or a scoped fallback statement.
    """

    scoped = f"saved output {saved_label} is non-finite"
    if not hasattr(log, "first_nonfinite"):
        return scoped
    try:
        return str(log.first_nonfinite())
    except (ValueError, RuntimeError, TypeError):
        return scoped


def _model_summary_lines(log: Any) -> list[str]:
    """Return model architecture and cost summary lines.

    Parameters
    ----------
    log:
        Model log to summarize.

    Returns
    -------
    list[str]
        Bullet lines for the model section.
    """

    model_class_name = getattr(log, "model_class_name", type(log).__name__)
    num_params = int(getattr(log, "num_params", 0) or 0)
    trainable_params = int(getattr(log, "num_params_trainable", 0) or 0)
    frozen_params = int(getattr(log, "num_params_frozen", 0) or 0)
    total_flops = int(getattr(log, "total_flops_forward", getattr(log, "total_flops", 0)) or 0)
    module_count = _safe_len(getattr(log, "modules", None))
    return [
        f"- Architecture: {model_class_name}.",
        (
            f"- Parameters: {_format_count(num_params)} total "
            f"({_format_count(trainable_params)} trainable, {_format_count(frozen_params)} frozen)."
        ),
        f"- Forward FLOPs: {_format_count(total_flops)}.",
        f"- Modules represented: {_format_count(module_count)}.",
    ]


def _capture_summary_lines(log: Any) -> list[str]:
    """Return layer, operation, pass, and tensor capture summary lines.

    Parameters
    ----------
    log:
        Model log to summarize.

    Returns
    -------
    list[str]
        Bullet lines for the capture section.
    """

    layer_count = _safe_len(getattr(log, "layer_list", None))
    operation_count = int(getattr(log, "num_ops", 0) or 0)
    tensor_total = int(getattr(log, "num_tensors", 0) or 0)
    tensor_saved = int(getattr(log, "num_saved_ops", 0) or 0)
    pass_counts = [
        int(value)
        for value in (getattr(log, "layer_num_calls", {}) or {}).values()
        if isinstance(value, int)
    ]
    max_ops = max(pass_counts, default=1)
    return [
        f"- Layers logged: {_format_count(layer_count)}.",
        f"- Operations logged: {_format_count(operation_count)}.",
        f"- Tensors saved: {_format_count(tensor_saved)} of {_format_count(tensor_total)}.",
        f"- Maximum observed ops for one layer: {_format_count(max_ops)}.",
    ]


def _backward_summary_lines(log: Any) -> list[str]:
    """Return backward-pass capture summary lines.

    Parameters
    ----------
    log:
        Model log to summarize.

    Returns
    -------
    list[str]
        Bullet lines for the backward section.
    """

    if not bool(getattr(log, "has_backward_pass", False)):
        return ["- No backward passes are recorded on this log."]
    pass_count = int(getattr(log, "num_backward_passes", 0) or 0)
    grad_fn_count = int(getattr(log, "num_grad_fns", 0) or 0)
    grad_fn_call_count = int(getattr(log, "num_grad_fn_calls", 0) or 0)
    saved_grad_count = _safe_len(getattr(log, "saved_grad_ops", None))
    total_backward_memory = getattr(log, "total_backward_memory", None)
    memory_line = (
        f"- Unique backward payload memory: {total_backward_memory}."
        if total_backward_memory is not None
        else "- Unique backward payload memory: unavailable."
    )
    return [
        f"- Backward passes: {_format_count(pass_count)}.",
        (
            f"- GradFn records: {_format_count(grad_fn_count)} nodes, "
            f"{_format_count(grad_fn_call_count)} calls."
        ),
        f"- Op gradient records saved: {_format_count(saved_grad_count)}.",
        memory_line,
    ]


def _anomaly_lines(log: Any) -> list[str]:
    """Return NaN/Inf anomaly lines.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    list[str]
        Bullet lines describing non-finite outs.
    """

    nonfinite_labels = [
        str(getattr(layer, "layer_label", "unknown"))
        for layer in nonfinite_layers(log, kind="saved")
    ]
    if not nonfinite_labels:
        return ["- No NaN or Inf values were found in saved outs."]
    first = nonfinite_labels[0]
    return [
        f"- {len(nonfinite_labels)} saved out(s) contain NaN or Inf values.",
        f"- First affected layer: {first}.",
        f"- Detail: {_first_nonfinite_detail(log, first)}",
    ]


def _intervention_lines(log: Any) -> list[str]:
    """Return intervention summary lines.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    list[str]
        Bullet lines describing applied intervention recipes.
    """

    spec = getattr(log, "_intervention_spec", None)
    target_specs = tuple(getattr(spec, "target_value_specs", ()) or ())
    hook_specs = tuple(getattr(spec, "hook_specs", ()) or ())
    history = list(getattr(log, "state_history", []) or [])
    if not target_specs and not hook_specs and not history:
        return ["- No interventions are recorded on this log."]
    return [
        f"- Target-value edits: {_format_count(len(target_specs))}.",
        f"- Hook edits: {_format_count(len(hook_specs))}.",
        f"- Recorded intervention operations: {_format_count(len(history))}.",
    ]


def _pattern_lines(log: Any, *, audience: Audience) -> list[str]:
    """Return notable graph-pattern lines.

    Parameters
    ----------
    log:
        Model log to inspect.
    audience:
        Requested report audience.

    Returns
    -------
    list[str]
        Bullet lines for graph patterns.
    """

    lines = [
        _shared_parameter_line(log),
        _recurrent_loop_line(log),
        _dynamic_control_flow_line(log),
    ]
    if audience in {"researcher", "auto"}:
        lines.append(_operation_mix_line(log))
    if audience == "practitioner":
        lines.append(_operational_status_line(log))
    return lines


def _shared_parameter_line(log: Any) -> str:
    """Return a shared-parameter pattern line.

    A parameter is *shared* (weight-tied across modules, or reused across
    recurrent passes) exactly when the same parameter object participates in
    more than one operation. The ground-truth signal is therefore the parameter's
    usage count -- ``num_calls`` / distinct ``used_by_ops`` -- NOT
    ``co_parent_params``, which lists the sibling params of the SAME op
    (weight+bias of one Linear) and so is both a false positive for every plain
    multi-param op and a false negative for genuine weight tying (a tied weight is
    the sole param of each op, so ``co_parent_params`` is empty). ``used_by_layers``
    also cannot detect tying: two tied modules roll into one equivalent layer
    label, so only the op-level usage count separates sharing from independence.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Shared-parameter summary.
    """

    shared = 0
    for param_log in getattr(log, "param_logs", []) or []:
        used_by_ops = set(getattr(param_log, "used_by_ops", ()) or ())
        try:
            num_calls = int(getattr(param_log, "num_calls", 0) or 0)
        except (TypeError, ValueError):
            num_calls = 0
        if len(used_by_ops) > 1 or num_calls > 1:
            shared += 1
    if shared:
        return (
            f"- Shared parameters: {_format_count(shared)} parameter object(s) "
            "are used by more than one operation (weight tying or recurrent reuse)."
        )
    return "- Shared parameters: none reported."


def _recurrent_loop_line(log: Any) -> str:
    """Return a recurrent-loop pattern line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Recurrent-loop summary.
    """

    max_loops = int(getattr(log, "max_layer_op_count", 1) or 1)
    if max_loops > 1:
        return f"- Recurrent loops: at least one layer was observed across {max_loops} ops."
    return "- Recurrent loops: no repeated layer ops were detected."


def _dynamic_control_flow_line(log: Any) -> str:
    """Return a dynamic-control-flow pattern line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Dynamic-control-flow summary.
    """

    event_count = _safe_len(getattr(log, "conditional_records", None))
    has_branching = bool(getattr(log, "has_conditional_branching", False))
    if event_count or has_branching:
        return (
            f"- Dynamic control flow: {_format_count(event_count)} conditional event(s) recorded."
        )
    return "- Dynamic control flow: no conditional events were recorded for this input."


def _operation_mix_line(log: Any) -> str:
    """Return a compact operation mix line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Most common operation types.
    """

    names = [
        str(getattr(layer, "func_name", "unknown"))
        for layer in getattr(log, "layer_list", []) or []
        if str(getattr(layer, "func_name", "none")) != "none"
    ]
    if not names:
        return "- Operation mix: no operation names were available."
    common = ", ".join(f"{name} x{count}" for name, count in Counter(names).most_common(5))
    return f"- Operation mix: {common}."


def _operational_status_line(log: Any) -> str:
    """Return a practitioner-oriented operational status line.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    str
        Operational status summary.
    """

    cache_hit = bool(getattr(log, "capture_cache_hit", False))
    streamed = sum(
        1
        for layer in getattr(log, "layer_list", []) or []
        if getattr(layer, "out_ref", None) is not None
        or getattr(layer, "grad_ref", None) is not None
    )
    return f"- Operational status: cache_hit={cache_hit}, streamed_ops={streamed}."


def _safe_len(value: Any) -> int:
    """Return ``len(value)`` or zero when unavailable.

    Parameters
    ----------
    value:
        Object to measure.

    Returns
    -------
    int
        Length or zero.
    """

    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def _format_count(value: int) -> str:
    """Return an integer with comma separators.

    Parameters
    ----------
    value:
        Integer value to format.

    Returns
    -------
    str
        Formatted value.
    """

    return f"{value:,}"
