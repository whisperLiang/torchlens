"""Live replay validation for the technical-preview MLX backend.

The oracle mirrors the Paddle/tinygrad pattern: every captured operation is
re-invoked from the exact call material recorded at emit time (function,
positional/keyword arguments, output), the replayed output must match the
captured payload, and a parent-perturbation tripwire proves the replay is not
vacuous. The denominator is the captured (whitelisted) op set — MLX cannot
observe unwrapped internals, and that scope is documented rather than hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MLXOpCapture:
    """One captured MLX call retained for live replay validation.

    Parameters
    ----------
    labels_raw:
        Raw TorchLens labels reserved for the call's output arrays, in
        emission order.
    op_name:
        Wrapped operation name.
    func:
        Original (unwrapped) callable invoked by the wrapper.
    args:
        Positional arguments as observed at call time.
    kwargs:
        Keyword arguments as observed at call time.
    output:
        Raw output object returned by the call.
    """

    labels_raw: tuple[str, ...]
    op_name: str
    func: Any
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    output: Any = None


def _is_mlx_array(value: Any) -> bool:
    """Return whether ``value`` is an ``mlx.core.array``.

    Parameters
    ----------
    value:
        Candidate object.

    Returns
    -------
    bool
        True for MLX arrays.
    """

    try:
        import mlx.core as mx
    except ImportError:
        return False
    return isinstance(value, mx.array)


def _iter_output_arrays(output: Any) -> tuple[Any, ...]:
    """Flatten an MLX call output into its array leaves.

    Parameters
    ----------
    output:
        Raw output object (array, tuple/list of arrays, or other).

    Returns
    -------
    tuple[Any, ...]
        Array leaves in deterministic order.
    """

    if _is_mlx_array(output):
        return (output,)
    if isinstance(output, (tuple, list)):
        leaves: list[Any] = []
        for item in output:
            leaves.extend(_iter_output_arrays(item))
        return tuple(leaves)
    return ()


def _payloads_close(a: Any, b: Any) -> bool:
    """Return whether two MLX arrays match within dtype-aware tolerance.

    Parameters
    ----------
    a, b:
        Arrays to compare.

    Returns
    -------
    bool
        True when shapes/dtypes agree and values are close.
    """

    a_np = np.asarray(a)
    b_np = np.asarray(b)
    if a_np.shape != b_np.shape or a_np.dtype != b_np.dtype:
        return False
    if a_np.dtype.kind in ("f", "c"):
        tolerance = 1e-2 if a_np.dtype.itemsize <= 2 else 1e-5
        return bool(np.allclose(a_np, b_np, rtol=tolerance, atol=tolerance, equal_nan=True))
    return bool(np.array_equal(a_np, b_np))


def _perturb_candidates(value: Any) -> tuple[Any, ...]:
    """Return perturbed variants of one MLX array argument.

    Parameters
    ----------
    value:
        MLX array to perturb.

    Returns
    -------
    tuple[Any, ...]
        Candidate replacement arrays.
    """

    import mlx.core as mx

    value_np = np.asarray(value)
    if value_np.dtype.kind == "f":
        return (value + mx.array(0.5, dtype=value.dtype), value * 2)
    if value_np.dtype.kind in ("i", "u"):
        return (value + 1,)
    if value_np.dtype.kind == "b":
        return (mx.logical_not(value),)
    return ()


def _perturbation_changes_output(capture: MLXOpCapture, baseline: tuple[Any, ...]) -> bool:
    """Return whether perturbing one tensor argument changes the replay output.

    Parameters
    ----------
    capture:
        Captured call to perturb.
    baseline:
        Replayed baseline output arrays.

    Returns
    -------
    bool
        True when some perturbation candidate produces a different output, or
        when the call has no perturbable tensor argument (vacuously true —
        constant producers cannot be perturbed through their inputs).
    """

    import mlx.core as mx

    array_positions = [
        index for index, value in enumerate(capture.args) if _is_mlx_array(value)
    ]
    if not array_positions:
        return True
    position = array_positions[0]
    for candidate in _perturb_candidates(capture.args[position]):
        try:
            perturbed_args = (
                *capture.args[:position],
                candidate,
                *capture.args[position + 1 :],
            )
            perturbed = _iter_output_arrays(
                capture.func(*perturbed_args, **capture.kwargs)
            )
            mx.eval(*perturbed)
        except Exception:
            continue
        if len(perturbed) != len(baseline):
            return True
        if any(
            not _payloads_close(p_out, b_out)
            for p_out, b_out in zip(perturbed, baseline)
        ):
            return True
    return False


def _ops_by_label(trace: Any) -> dict[str, Any]:
    """Return materialized trace operations keyed by all known labels.

    Parameters
    ----------
    trace:
        Materialized TorchLens trace.

    Returns
    -------
    dict[str, Any]
        Operations keyed by raw, layer, and pass labels.
    """

    result: dict[str, Any] = {}
    for op in getattr(trace, "layer_list", ()):
        for label in (
            getattr(op, "_label_raw", None),
            getattr(op, "layer_label", None),
            getattr(op, "label", None),
        ):
            if isinstance(label, str):
                result[label] = op
    return result


def _saved_payload(trace: Any, ops_by_label: dict[str, Any], label_raw: str) -> Any:
    """Return the trace's saved payload for one raw label.

    Parameters
    ----------
    trace:
        Trace being validated.
    ops_by_label:
        Label-to-op index from :func:`_ops_by_label`.
    label_raw:
        Raw label whose payload is required.

    Returns
    -------
    Any
        Saved MLX array (public ``op.out`` or the selective-save hidden copy).
    """

    op = ops_by_label.get(label_raw)
    payload = None if op is None else getattr(op, "out", None)
    if payload is None:
        hidden = getattr(trace, "_selective_save_hidden_payloads", {}) or {}
        payload = hidden.get(label_raw)
    if payload is None:
        raise ValueError(f"MLX validation found no saved payload for {label_raw!r}.")
    return payload


def validate_mlx_captures(trace: Any) -> tuple[int, int]:
    """Replay every captured MLX call against the trace's saved payloads.

    Parameters
    ----------
    trace:
        Live MLX trace carrying ``_mlx_op_captures`` replay material.

    Returns
    -------
    tuple[int, int]
        ``(replayed_count, failed_count)`` over the captured op set.
    """

    import mlx.core as mx

    ops_by_label = _ops_by_label(trace)
    replayed_count = 0
    failed_count = 0
    for capture in tuple(getattr(trace, "_mlx_op_captures", ())):
        try:
            replayed = _iter_output_arrays(
                capture.func(*capture.args, **capture.kwargs)
            )
            mx.eval(*replayed)
            if len(replayed) != len(capture.labels_raw) or not replayed:
                failed_count += 1
                continue
            expected = tuple(
                _saved_payload(trace, ops_by_label, label)
                for label in capture.labels_raw
            )
            mx.eval(*expected)
            if any(
                not _payloads_close(r_out, e_out)
                for r_out, e_out in zip(replayed, expected)
            ):
                failed_count += 1
                continue
            if not _perturbation_changes_output(capture, replayed):
                failed_count += 1
                continue
            replayed_count += 1
        except Exception:
            failed_count += 1
    return replayed_count, failed_count
