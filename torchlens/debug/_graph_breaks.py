"""Correlate TorchDynamo graph breaks with eager TorchLens operations."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

from torch import nn

from .._input_coerce import _coerce_input_args
from ..utils import _torch_compat
from ..utils.arg_handling import normalize_input_args


class GraphBreaksUnavailableError(RuntimeError):
    """Raised when the running torch does not expose Dynamo explain."""


class GraphBreaksNormalizationError(RuntimeError):
    """Raised when a Dynamo explain result has an unsupported shape."""


@dataclass(frozen=True)
class GraphBreak:
    """One Dynamo graph break and its nearest eager operation sites.

    Attributes
    ----------
    reason:
        Reason string supplied by Dynamo.
    source_file:
        User source file supplied by Dynamo, if available.
    line_number:
        One-indexed source line supplied by Dynamo, if available.
    matched_op_labels:
        Eager TorchLens operation labels at the smallest same-file line distance.
    unmatched_reason:
        Evidence-backed explanation when no eager operation could be correlated.
    """

    reason: str
    source_file: str | None
    line_number: int | None
    matched_op_labels: tuple[str, ...]
    unmatched_reason: str | None


@dataclass(frozen=True)
class GraphBreakReport:
    """Structured Dynamo graph-break correlation result.

    Attributes
    ----------
    breaks:
        Graph breaks in Dynamo's reported order.
    """

    breaks: tuple[GraphBreak, ...]

    def __len__(self) -> int:
        """Return the number of graph breaks.

        Returns
        -------
        int
            Number of normalized break records.
        """

        return len(self.breaks)


def _model_call_args(
    model: nn.Module,
    x: Any,
    trace_kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Resolve model call arguments consistently with eager tracing.

    Parameters
    ----------
    model:
        Model whose forward signature resolves input-container ambiguity.
    x:
        Public TorchLens input value.
    trace_kwargs:
        Trace keyword arguments, including optional ``input_kwargs``.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Positional and keyword arguments for Dynamo explain.
    """

    coerced = _coerce_input_args(model, x)
    args = tuple(normalize_input_args(coerced, model))
    input_kwargs = trace_kwargs.get("input_kwargs")
    kwargs = dict(input_kwargs) if isinstance(input_kwargs, dict) else {}
    return args, kwargs


def _op_locations(op: Any) -> tuple[tuple[str, int], ...]:
    """Return recorded source locations for one eager operation.

    Parameters
    ----------
    op:
        TorchLens operation record.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Absolute source filenames and one-indexed line numbers.
    """

    locations: list[tuple[str, int]] = []
    for frame in getattr(op, "code_context", ()) or ():
        source_file = getattr(frame, "file", None)
        line_number = getattr(frame, "line_number", None)
        if source_file is None or not isinstance(line_number, int):
            continue
        locations.append((os.path.abspath(str(source_file)), line_number))
    return tuple(locations)


def _correlate_break(trace: Any, record: Any) -> GraphBreak:
    """Map one normalized Dynamo break to nearest same-file eager ops.

    Parameters
    ----------
    trace:
        Completed eager TorchLens trace.
    record:
        Version-neutral break record from the compatibility layer.

    Returns
    -------
    GraphBreak
        Public correlation record.
    """

    if record.source_file is None or record.line_number is None:
        return GraphBreak(
            record.reason,
            record.source_file,
            record.line_number,
            (),
            "Dynamo did not provide a source file and line number.",
        )
    source_file = os.path.abspath(record.source_file)
    distances: list[tuple[int, int, str]] = []
    for op in trace.layer_list:
        label = str(getattr(op, "label", getattr(op, "layer_label", "<unknown>")))
        same_file_lines = [line for filename, line in _op_locations(op) if filename == source_file]
        if same_file_lines:
            distances.append(
                (
                    min(abs(line - record.line_number) for line in same_file_lines),
                    int(getattr(op, "ordinal_index", 0)),
                    label,
                )
            )
    if not distances:
        return GraphBreak(
            record.reason,
            record.source_file,
            record.line_number,
            (),
            "No eager operation recorded a source location in the Dynamo break file.",
        )
    nearest_distance = min(distance for distance, _, _ in distances)
    labels = tuple(
        label
        for distance, _, label in sorted(distances, key=lambda item: (item[0], item[1]))
        if distance == nearest_distance
    )
    return GraphBreak(record.reason, record.source_file, record.line_number, labels, None)


def graph_breaks(model: nn.Module, x: Any, **trace_kwargs: Any) -> GraphBreakReport:
    """Report Dynamo graph breaks and correlate them to eager trace operations.

    The Dynamo result is normalized through ``torchlens.utils._torch_compat``
    without torch-version parsing. Correlation uses recorded
    :class:`FuncCallLocation` file/line evidence; empty matches state why no
    correlation was possible.

    Parameters
    ----------
    model:
        Eager PyTorch module to inspect.
    x:
        Model input accepted by :func:`torchlens.trace`.
    **trace_kwargs:
        Additional keyword arguments forwarded to :func:`torchlens.trace`.

    Returns
    -------
    GraphBreakReport
        Normalized breaks and nearest eager operation labels.

    Raises
    ------
    GraphBreaksUnavailableError
        If this torch runtime lacks ``torch._dynamo.explain``.
    GraphBreaksNormalizationError
        If the runtime returns an unrecognized explain-result shape.
    """

    if not _torch_compat.HAS_DYNAMO_EXPLAIN:
        raise GraphBreaksUnavailableError(
            "torch._dynamo.explain is unavailable in this torch runtime"
        )
    args, kwargs = _model_call_args(model, x, trace_kwargs)
    try:
        raw_explanation = _torch_compat.run_dynamo_explain(model, args, kwargs)
    except RuntimeError as exc:
        if _torch_compat.get_dynamo_explain() is None:
            raise GraphBreaksUnavailableError(str(exc)) from exc
        raise
    try:
        normalized = _torch_compat.normalize_dynamo_explain_output(raw_explanation)
    except _torch_compat._DynamoExplainOutputError as exc:
        raise GraphBreaksNormalizationError(str(exc)) from exc

    from .. import trace as capture_trace

    eager_trace = capture_trace(model, x, **trace_kwargs)
    return GraphBreakReport(tuple(_correlate_break(eager_trace, record) for record in normalized))


__all__ = [
    "GraphBreak",
    "GraphBreakReport",
    "GraphBreaksNormalizationError",
    "GraphBreaksUnavailableError",
    "graph_breaks",
]
