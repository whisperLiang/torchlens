"""Symbolic shape helpers for split replay."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

from .errors import SplitBoundaryError, SplitErrorContext, SplitUnsupportedError


SymbolicDim = int | str


@dataclass(frozen=True)
class SymbolicShape:
    """Tuple-like symbolic shape representation."""

    dims: tuple[SymbolicDim, ...]

    def __iter__(self) -> Any:
        """Iterate over shape dimensions."""

        return iter(self.dims)

    def __len__(self) -> int:
        """Return the rank."""

        return len(self.dims)

    def as_tuple(self) -> tuple[SymbolicDim, ...]:
        """Return dimensions as a tuple."""

        return self.dims


@dataclass(frozen=True)
class ShapeEnv:
    """Runtime shape environment for a prepared split."""

    batch_symbol: str
    traced_batch_size: int | None
    dynamic_batch: tuple[int, int] | None = None


def infer_traced_batch_size(trace: Any) -> int | None:
    """Infer the leading traced batch dimension from trace input layers.

    Parameters
    ----------
    trace:
        Current-main TorchLens ``Trace``.

    Returns
    -------
    int | None
        Leading batch size, or ``None`` when no tensor input shape is available.
    """

    layer_dict = getattr(trace, "layer_dict_all_keys", {}) or {}
    output_leading_dims: set[int] = set()
    for label in getattr(trace, "output_layers", ()) or ():
        op = layer_dict.get(label)
        shape = getattr(op, "shape", None)
        if shape:
            output_leading_dims.add(int(shape[0]))
    input_leading_dims: list[int] = []
    for label in getattr(trace, "input_layers", ()) or ():
        op = layer_dict.get(label)
        shape = getattr(op, "shape", None)
        if shape:
            leading_dim = int(shape[0])
            input_leading_dims.append(leading_dim)
            if leading_dim in output_leading_dims:
                return leading_dim
    for op in getattr(trace, "layer_list", ()) or ():
        if getattr(op, "is_input", False):
            shape = getattr(op, "shape", None)
            if shape:
                leading_dim = int(shape[0])
                input_leading_dims.append(leading_dim)
                if leading_dim in output_leading_dims:
                    return leading_dim
    if input_leading_dims:
        return input_leading_dims[0]
    return None


def symbolic_shape_from_tensor_ref(
    tensor_ref: Any,
    *,
    batch_symbol: str,
    dynamic_batch: tuple[int, int] | None,
    traced_batch_size: int | None = None,
) -> SymbolicShape | None:
    """Build a symbolic shape from an op, tensor ref, tensor, or raw shape.

    Parameters
    ----------
    tensor_ref:
        Object exposing ``shape`` or a raw shape tuple/list.
    batch_symbol:
        Symbol replacing a dynamic leading batch dimension.
    dynamic_batch:
        Optional dynamic batch range.
    traced_batch_size:
        Leading dimension observed during tracing.

    Returns
    -------
    SymbolicShape | None
        Symbolic shape, or ``None`` when shape metadata is unavailable.
    """

    shape = getattr(tensor_ref, "shape", tensor_ref)
    if shape is None:
        return None
    try:
        dims = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None
    if (
        dynamic_batch is not None
        and traced_batch_size is not None
        and dims
        and dims[0] == traced_batch_size
    ):
        return SymbolicShape((batch_symbol, *dims[1:]))
    return SymbolicShape(dims)


def validate_tensor_against_symbolic_shape(
    value: Any,
    symbolic_shape: SymbolicShape | None,
    *,
    adapter: Any,
    dynamic_batch: tuple[int, int] | None = None,
    batch_symbol: str = "B",
    backend: str = "unknown",
    split_point: str = "",
    label: str | None = None,
) -> None:
    """Validate a tensor-like value against a symbolic shape.

    Parameters
    ----------
    value:
        Runtime tensor-like value.
    symbolic_shape:
        Expected symbolic shape.
    adapter:
        Backend adapter used to query shape.
    dynamic_batch:
        Optional inclusive batch range.
    batch_symbol:
        Batch symbol expected in ``symbolic_shape``.
    backend:
        Backend name for error context.
    split_point:
        Split point for error context.
    label:
        Boundary label for error context.
    """

    if symbolic_shape is None:
        return
    runtime_shape = adapter.shape(value)
    if runtime_shape is None:
        return
    expected = symbolic_shape.as_tuple()
    if len(runtime_shape) != len(expected):
        raise SplitBoundaryError(
            f"Boundary tensor {label!r} has rank {len(runtime_shape)}, expected {len(expected)}.",
            context=SplitErrorContext(
                backend=backend,
                split_point=split_point,
                module_path=None,
                op_type=None,
                layer_label=label,
                reason="rank mismatch",
                traced_shape=expected,
                runtime_shape=runtime_shape,
            ),
        )
    for index, (runtime_dim, expected_dim) in enumerate(zip(runtime_shape, expected)):
        if expected_dim == batch_symbol:
            if dynamic_batch is not None:
                low, high = dynamic_batch
                if not low <= runtime_dim <= high:
                    raise SplitBoundaryError(
                        f"Boundary batch dimension {runtime_dim} is outside {dynamic_batch}.",
                        context=SplitErrorContext(
                            backend=backend,
                            split_point=split_point,
                            module_path=None,
                            op_type=None,
                            layer_label=label,
                            reason="dynamic batch outside allowed range",
                            traced_shape=expected,
                            runtime_shape=runtime_shape,
                        ),
                    )
            continue
        if runtime_dim != expected_dim:
            raise SplitBoundaryError(
                f"Boundary tensor {label!r} dimension {index} is {runtime_dim}, "
                f"expected {expected_dim}.",
                context=SplitErrorContext(
                    backend=backend,
                    split_point=split_point,
                    module_path=None,
                    op_type=None,
                    layer_label=label,
                    reason="shape mismatch",
                    traced_shape=expected,
                    runtime_shape=runtime_shape,
                ),
            )


def rewrite_dynamic_batch_value(
    value: Any,
    *,
    traced_batch_size: int | None,
    runtime_batch_size: int | None,
) -> Any:
    """Rewrite traced batch literals inside common leading-dim shape values.

    Parameters
    ----------
    value:
        Literal value or nested literal value. Flat integer lists/tuples are
        treated as shape descriptors; only their leading dimension is eligible
        for replacement. Other containers are traversed to find nested shape
        descriptors.
    traced_batch_size:
        Batch size captured in the trace.
    runtime_batch_size:
        Batch size being replayed.

    Returns
    -------
    Any
        Rewritten value.
    """

    if traced_batch_size is None or runtime_batch_size is None:
        return value
    if isinstance(value, Integral) and not isinstance(value, bool):
        return value
    if isinstance(value, tuple):
        if _is_flat_shape_literal(value):
            return tuple(
                _rewrite_leading_shape_dim(
                    list(value),
                    traced_batch_size=traced_batch_size,
                    runtime_batch_size=runtime_batch_size,
                )
            )
        return tuple(
            rewrite_dynamic_batch_value(
                item,
                traced_batch_size=traced_batch_size,
                runtime_batch_size=runtime_batch_size,
            )
            for item in value
        )
    if isinstance(value, list):
        if _is_flat_shape_literal(value):
            return _rewrite_leading_shape_dim(
                list(value),
                traced_batch_size=traced_batch_size,
                runtime_batch_size=runtime_batch_size,
            )
        return [
            rewrite_dynamic_batch_value(
                item,
                traced_batch_size=traced_batch_size,
                runtime_batch_size=runtime_batch_size,
            )
            for item in value
        ]
    if isinstance(value, dict):
        return {
            key: rewrite_dynamic_batch_value(
                item,
                traced_batch_size=traced_batch_size,
                runtime_batch_size=runtime_batch_size,
            )
            for key, item in value.items()
        }
    return value


def _is_flat_shape_literal(value: tuple[Any, ...] | list[Any]) -> bool:
    """Return whether ``value`` looks like one shape descriptor."""

    return bool(value) and all(
        (isinstance(item, Integral) and not isinstance(item, bool)) for item in value
    )


def _rewrite_leading_shape_dim(
    value: list[Any],
    *,
    traced_batch_size: int,
    runtime_batch_size: int,
) -> list[Any]:
    """Rewrite only the leading dimension of a flat shape descriptor."""

    if value and int(value[0]) == traced_batch_size:
        value[0] = runtime_batch_size
    return value


_SHAPE_SENSITIVE_OP_TOKENS = frozenset(
    {
        "view",
        "reshape",
        "flatten",
        "expand",
        "broadcast",
        "repeat",
        "tile",
        "zeros",
        "ones",
        "empty",
        "full",
        "fill",
        "new_zeros",
        "new_ones",
        "new_empty",
        "iota",
    }
)


def is_dynamic_batch_shape_sensitive_op(*names: str | None) -> bool:
    """Return whether an op name may carry batch-sized shape literals.

    Parameters
    ----------
    *names:
        Candidate operation/function names.

    Returns
    -------
    bool
        ``True`` when any name contains an audited shape-sensitive token.
    """

    for name in names:
        normalized = "" if name is None else str(name).lower().replace(" ", "")
        if any(token in normalized for token in _SHAPE_SENSITIVE_OP_TOKENS):
            return True
    return False


def maybe_rewrite_dynamic_batch_value(
    value: Any,
    *,
    op_type: str | None,
    func_name: str | None = None,
    traced_batch_size: int | None,
    runtime_batch_size: int | None,
    dynamic_batch: tuple[int, int] | None,
) -> Any:
    """Rewrite batch literals only for audited shape-sensitive operations.

    Parameters
    ----------
    value:
        Literal value or nested literal value.
    op_type:
        Backend operation type.
    func_name:
        Optional callable/function name.
    traced_batch_size:
        Batch size captured in the trace.
    runtime_batch_size:
        Batch size being replayed.
    dynamic_batch:
        Optional inclusive runtime batch range.

    Returns
    -------
    Any
        Rewritten value when the op is shape-sensitive, otherwise ``value``.
    """

    if dynamic_batch is None:
        return value
    if not is_dynamic_batch_shape_sensitive_op(op_type, func_name):
        return value
    return rewrite_dynamic_batch_value(
        value,
        traced_batch_size=traced_batch_size,
        runtime_batch_size=runtime_batch_size,
    )


def require_safe_dynamic_shape_rewrite(
    *,
    op_type: str,
    backend: str,
    split_point: str,
    label: str,
) -> None:
    """Raise for shape-sensitive ops not covered by v1 rewrite support."""

    if is_dynamic_batch_shape_sensitive_op(op_type):
        return
    raise SplitUnsupportedError(
        f"Dynamic batch replay cannot safely rewrite shape literals for {op_type!r}.",
        context=SplitErrorContext(
            backend=backend,
            split_point=split_point,
            module_path=None,
            op_type=op_type,
            layer_label=label,
            reason="unsafe dynamic shape rewrite",
        ),
    )


__all__ = [
    "ShapeEnv",
    "SymbolicDim",
    "SymbolicShape",
    "infer_traced_batch_size",
    "is_dynamic_batch_shape_sensitive_op",
    "maybe_rewrite_dynamic_batch_value",
    "require_safe_dynamic_shape_rewrite",
    "rewrite_dynamic_batch_value",
    "symbolic_shape_from_tensor_ref",
    "validate_tensor_against_symbolic_shape",
]
