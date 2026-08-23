"""Machine-readable trace dump for agent consumers (``Trace.to_agent_json``).

DOCUMENTED-UNSTABLE spelling (naming ratification deferred to the UI/naming
sprint). The dump describes the SAME public surface a human drives -- it is a
navigation aid, never a parallel API: every record points back at the live
spelling (``trace[label]``, ``tl.func(...)``, ``tl.report.explain(...)``) an
agent should call next.

Honesty contract (report/AGENTS.md): capture outcome/verification facts are
carried verbatim from the same source ``explain()`` uses; a rescued, ceilinged,
or structure-only capture must stay visible in the dump.
"""

from __future__ import annotations

from typing import Any

from ._explain import _capture_verification, _safe_len

#: Schema identifier for the agent trace dump.
AGENT_TRACE_SCHEMA = "torchlens.agent_trace.v1"

#: Static self-description embedded in every dump so an agent can navigate the
#: trace without reading prose docs first.
_GUIDE: dict[str, Any] = {
    "purpose": (
        "Structural dump of one captured forward pass. Use it to discover "
        "layer labels, graph edges, module structure, and capture-honesty "
        "facts, then drive the live object with the spellings below."
    ),
    "payloads": (
        "Tensor values are never inlined here. Read a saved activation as "
        "trace[<layer_label>].out; ops with saved=false raise on payload "
        "reads -- re-capture with a wider save= predicate (e.g. "
        "tl.trace(model, x, save=tl.func('relu')))."
    ),
    "navigation": {
        "ops": (
            "Execution-ordered operation records. 'label' is the unique "
            "pass-qualified id (layer_label:pass); 'layer_label' addresses "
            "the rolled layer via trace[layer_label]. 'parents'/'children' "
            "hold layer_labels of adjacent ops in the dataflow graph."
        ),
        "modules": (
            "Module hierarchy rows keyed by dotted address; 'address_parent'"
            " / 'address_children' give the containment tree."
        ),
        "capture": (
            "Settled capture facts. capture_verified=false means parts of "
            "the forward may be missing or unattributed -- treat every count "
            "as a lower bound. structure_only=true means shapes/dtypes are "
            "hypotheses, not measurements."
        ),
        "truncation": (
            "Non-null when max_ops dropped op rows; counts disclose exactly what was omitted."
        ),
    },
    "next_steps": {
        "summary": "trace.summary()",
        "plain_language_report": "tl.report.explain(trace)",
        "budgeted_report": "tl.report.explain(trace, max_tokens=500)",
        "one_activation": "trace[<layer_label>].out",
        "receptive_field": "trace[<layer_label>].receptive_field",
        "draw_graph": "trace.draw()",
        "environment_diagnosis": "tl.compat.report(model, x).to_markdown()",
    },
}


def _quantity_int(value: Any) -> int | None:
    """Coerce a TorchLens quantity (Flops/Bytes) to a plain integer.

    Parameters
    ----------
    value:
        Quantity-like object or ``None``.

    Returns
    -------
    int | None
        Integer value, or ``None`` when unavailable or non-numeric.
    """

    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_shape(shape: Any) -> list[int] | None:
    """Return a JSON-safe copy of a recorded output shape.

    Parameters
    ----------
    shape:
        Recorded shape tuple, or ``None``.

    Returns
    -------
    list[int] | None
        Plain list of dimension sizes, or ``None`` when unrecorded.
    """

    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except (TypeError, ValueError):
        return None


def _site_key_or_none(op: Any) -> str | None:
    """Return the op's structural site key, or ``None`` when unavailable.

    ``Op.site_key`` is a plain persisted field (``str | None``); legacy
    keyless artifacts carry ``None``, which the dump discloses as ``null``.

    Parameters
    ----------
    op:
        Operation record.

    Returns
    -------
    str | None
        ``site_key_v1`` string, or ``None`` on keyless artifacts.
    """

    key = getattr(op, "site_key", None)
    return str(key) if key is not None else None


def _op_entry(op: Any) -> dict[str, Any]:
    """Build one JSON-safe operation row.

    Parameters
    ----------
    op:
        Operation record from ``trace.layer_list``.

    Returns
    -------
    dict[str, Any]
        Flat JSON-serializable record for the op.
    """

    device_ref = getattr(op, "device_ref", None)
    dtype = getattr(op, "dtype", None)
    return {
        "label": str(getattr(op, "label", getattr(op, "layer_label", "unknown"))),
        "layer_label": str(getattr(op, "layer_label", "unknown")),
        "pass_index": int(getattr(op, "pass_index", 1) or 1),
        "num_passes": int(getattr(op, "num_passes", 1) or 1),
        "func_name": str(getattr(op, "func_name", "unknown")),
        "shape": _json_shape(getattr(op, "shape", None)),
        "dtype": str(dtype) if dtype is not None else None,
        "device": str(getattr(device_ref, "name", device_ref)) if device_ref else None,
        "parents": [str(p) for p in getattr(op, "parents", ()) or ()],
        "children": [str(c) for c in getattr(op, "children", ()) or ()],
        "module_call_stack": [str(m) for m in getattr(op, "module_call_stack", ()) or ()],
        "module_address": getattr(op, "atomic_module_address", None),
        "saved": bool(getattr(op, "has_saved_activation", False)),
        "site_key": _site_key_or_none(op),
        "num_params": int(getattr(op, "num_params", 0) or 0),
        "flops_forward": _quantity_int(getattr(op, "flops_forward", None)),
    }


def _module_entry(address: str, module: Any) -> dict[str, Any]:
    """Build one JSON-safe module hierarchy row.

    Parameters
    ----------
    address:
        Dotted module address key from ``trace.modules``.
    module:
        Module record.

    Returns
    -------
    dict[str, Any]
        Flat JSON-serializable record for the module.
    """

    parent = getattr(module, "address_parent", None)
    return {
        "address": str(address),
        "class_name": str(getattr(module, "class_name", type(module).__name__)),
        "num_calls": _safe_len(getattr(module, "calls", None)),
        "address_parent": str(parent) if parent is not None else None,
        "address_children": [str(c) for c in getattr(module, "address_children", ()) or ()],
        "num_params": int(getattr(module, "num_params", 0) or 0),
    }


def _op_labels(accessor: Any) -> list[str]:
    """Return pass-qualified labels from an op accessor or label sequence.

    Parameters
    ----------
    accessor:
        ``TraceOpAccessor`` (yields op records) or plain label sequence.

    Returns
    -------
    list[str]
        Pass-qualified op labels.
    """

    if accessor is None:
        return []
    labels: list[str] = []
    for item in accessor:
        labels.append(str(getattr(item, "label", item)))
    return labels


def build_agent_json(log: Any, *, max_ops: int | None = None) -> dict[str, Any]:
    """Build the self-describing machine-readable dump of a finished trace.

    Parameters
    ----------
    log:
        Completed ``Trace``.
    max_ops:
        Optional cap on emitted op rows (execution order, first ``max_ops``
        kept). Omission is disclosed in the ``truncation`` block, never
        silent.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dump under the ``torchlens.agent_trace.v1`` schema.

    Raises
    ------
    ValueError
        If ``max_ops`` is not a positive integer.
    """

    if max_ops is not None and (
        isinstance(max_ops, bool) or not isinstance(max_ops, int) or max_ops < 1
    ):
        raise ValueError(
            "max_ops must be a positive integer (the cap on emitted op rows); "
            "omit it to dump every op."
        )

    ops = list(getattr(log, "layer_list", []) or [])
    truncation: dict[str, Any] | None = None
    if max_ops is not None and len(ops) > max_ops:
        truncation = {
            "ops_included": max_ops,
            "ops_omitted": len(ops) - max_ops,
            "policy": "first max_ops rows in execution order",
            "note": (
                "Op rows were dropped to honor max_ops; counts above remain "
                "the full-capture truth. Re-dump without max_ops for the "
                "complete graph."
            ),
        }
        ops = ops[:max_ops]

    capture = {
        **_capture_verification(log),
        "backend": str(getattr(log, "backend", "unknown")),
        "model_class": str(getattr(log, "model_class_name", type(log).__name__)),
        "structure_only": bool(getattr(log, "structure_only", False)),
        "grouping": str(getattr(log, "grouping", "unknown")),
        "has_backward_pass": bool(getattr(log, "has_backward_pass", False)),
        "device_summary": getattr(log, "backend_runtime_device_summary", None),
    }

    modules_map = getattr(log, "modules", {}) or {}
    module_rows = [
        _module_entry(address, module)
        for address, module in modules_map.items()
        if isinstance(address, str)
    ]

    return {
        "schema": AGENT_TRACE_SCHEMA,
        "schema_stability": (
            "documented-unstable: field names may be renamed by the naming "
            "ratification sprint; branch on 'schema' before hard-coding."
        ),
        "guide": _GUIDE,
        "capture": capture,
        "counts": {
            "layers": _safe_len(getattr(log, "layer_labels", None)),
            "operations": int(getattr(log, "num_ops", 0) or 0),
            "tensors_total": int(getattr(log, "num_tensors", 0) or 0),
            "tensors_saved": int(getattr(log, "num_saved_ops", 0) or 0),
            "parameters": int(getattr(log, "num_params", 0) or 0),
            "modules": len(module_rows),
        },
        "inputs": _op_labels(getattr(log, "input_ops", None)),
        "outputs": _op_labels(getattr(log, "output_ops", None)),
        "layer_labels": [str(label) for label in getattr(log, "layer_labels", []) or []],
        "ops": [_op_entry(op) for op in ops],
        "modules": module_rows,
        "truncation": truncation,
    }
