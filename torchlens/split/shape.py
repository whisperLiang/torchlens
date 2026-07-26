"""Symbolic shape helpers for split replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import SplitBoundaryError, SplitErrorContext


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
        "slice",
        "dynamic_slice",
        # tinygrad lowers convolution and pooling into explicit PAD/SHRINK
        # shape transforms.  Their shape arguments are just as batch-sensitive
        # as reshape/view arguments, even though they do not contain the word
        # "reshape" in the op name.
        "shrink",
        "pad",
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


__all__ = [
    "SymbolicDim",
    "SymbolicShape",
    "infer_traced_batch_size",
    "is_dynamic_batch_shape_sensitive_op",
    "symbolic_shape_from_tensor_ref",
    "validate_tensor_against_symbolic_shape",
]
