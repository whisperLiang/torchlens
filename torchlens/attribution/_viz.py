"""Graph overlay bridging attribution results onto ``Trace.draw`` encoding channels."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, TypeAlias

from torch import Tensor

from torchlens.attribution._core import AttributionError, AttributionResult

_OverlayReduce: TypeAlias = Literal["abs_sum", "abs_mean", "sum", "max"]

_REDUCERS: dict[str, Callable[[Tensor], float]] = {
    "abs_sum": lambda t: float(t.detach().abs().sum().item()),
    "abs_mean": lambda t: float(t.detach().abs().mean().item()),
    "sum": lambda t: float(t.detach().sum().item()),
    "max": lambda t: float(t.detach().max().item()),
}


def _reduce_value(value: Any, reduce: str) -> float:
    """Reduce one attribution value to a finite color scalar.

    Parameters
    ----------
    value
        Attribution tensor or plain real scalar.
    reduce
        Closed-vocabulary reduction applied to tensor values.

    Returns
    -------
    float
        Scalar magnitude for the color channel.

    Raises
    ------
    AttributionError
        If ``value`` is neither a tensor nor a real scalar.
    """

    if isinstance(value, Tensor):
        return _REDUCERS[reduce](value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AttributionError(
            "overlay values must be attribution tensors or real scalars; "
            f"received {type(value).__name__}"
        )
    return float(value)


def _entry_items(source: Any) -> list[tuple[str, Any]]:
    """Normalize ``source`` into ``(layer key, raw value)`` pairs.

    Parameters
    ----------
    source
        One layer-scoped :class:`AttributionResult`, an iterable of them, or a
        mapping from module name / layer label to result, tensor, or scalar.

    Returns
    -------
    list[tuple[str, Any]]
        Key/value pairs prior to reduction.

    Raises
    ------
    AttributionError
        If a result carries no ``extra["layer"]`` key to anchor it, or the
        source kind is unsupported.
    """

    if isinstance(source, AttributionResult):
        source = (source,)
    if isinstance(source, Mapping):
        items: list[tuple[str, Any]] = []
        for key, value in source.items():
            items.append((key, value.values if isinstance(value, AttributionResult) else value))
        return items
    if isinstance(source, Iterable):
        items = []
        for result in source:
            if not isinstance(result, AttributionResult):
                raise AttributionError(
                    "overlay iterables must contain AttributionResult entries; "
                    f"received {type(result).__name__}"
                )
            layer = result.extra.get("layer")
            if not isinstance(layer, str):
                raise AttributionError(
                    f"{result.method!r} result carries no extra['layer'] anchor; "
                    "input attributions color no graph node -- pass a mapping "
                    "{module_name_or_layer_label: value} instead"
                )
            items.append((layer, result.values))
        return items
    raise AttributionError(
        "overlay source must be an AttributionResult, an iterable of them, or "
        f"a mapping; received {type(source).__name__}"
    )


def _classify_entries(
    trace: Any,
    source: Any,
    reduce: str,
) -> tuple[dict[str, float], dict[str, float]]:
    """Resolve overlay entries into module-address and layer-label value maps.

    Parameters
    ----------
    trace
        Trace whose module outputs and layer labels anchor the keys.
    source
        Overlay source accepted by :func:`overlay`.
    reduce
        Validated closed-vocabulary reduction name.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        Module-address-keyed and layer-label-keyed scalar maps.

    Raises
    ------
    AttributionError
        If a key matches neither a module output nor a layer label.
    """

    known_modules: set[str] = set()
    known_labels: set[str] = set()
    for label in trace.layer_labels:
        known_labels.add(label)
        known_modules.update(trace[label].output_of_modules)

    module_values: dict[str, float] = {}
    label_values: dict[str, float] = {}
    for key, raw in _entry_items(source):
        value = _reduce_value(raw, reduce)
        if key in known_modules:
            module_values[key] = value
        elif key in known_labels:
            label_values[key] = value
        else:
            raise AttributionError(
                f"overlay key {key!r} matches no module output and no layer "
                "label on this trace; use a name from model.named_modules() "
                "or a label from trace.layer_labels"
            )
    return module_values, label_values


def overlay(
    trace: Any,
    source: Any,
    *,
    reduce: _OverlayReduce = "abs_sum",
) -> Callable[[Any], float | None]:
    """Build a ``color_by`` callable painting attribution values onto a graph.

    The returned callable maps rendered nodes to scalar magnitudes:
    nodes that are the output of an attributed module (or match an attributed
    layer label) receive that layer's reduced value, and every other node
    returns ``None`` (drawn unencoded, disclosed by the legend's ``n/a`` note).
    A module fired several times paints its per-layer TOTAL on every node of
    that module; per-pass splits are not claimed.

    Parameters
    ----------
    trace
        Trace the overlay will be drawn on; keys are validated against it.
    source
        One layer-scoped :class:`AttributionResult` (``extra['layer']``
        present), an iterable of them, or a mapping from
        ``model.named_modules()`` name or trace layer label to a result,
        tensor, or real scalar.
    reduce
        Closed-vocabulary reduction for tensor values: ``"abs_sum"``
        (default), ``"abs_mean"``, ``"sum"``, or ``"max"``.

    Returns
    -------
    Callable[[Any], float | None]
        Named callable for ``trace.draw(color_by=...)``.

    Raises
    ------
    AttributionError
        If ``reduce`` is unknown, or a key matches neither a module output
        nor a layer label on ``trace``.
    """

    if reduce not in _REDUCERS:
        raise AttributionError(f"reduce must be one of {sorted(_REDUCERS)}; received {reduce!r}")
    module_values, label_values = _classify_entries(trace, source, reduce)

    def attribution_overlay(node: Any) -> float | None:
        """Return the attributed magnitude for one rendered node."""

        if getattr(node, "is_input", False) or getattr(node, "is_output", False):
            return None
        label = getattr(node, "layer_label", None)
        if label in label_values:
            return label_values[label]
        for address in getattr(node, "output_of_modules", ()) or ():
            if address in module_values:
                return module_values[address]
        return None

    return attribution_overlay


__all__ = ["overlay"]
