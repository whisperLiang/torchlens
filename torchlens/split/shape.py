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


def symbolic_shape_from_tensor_ref(tensor_ref: Any) -> SymbolicShape | None:
    """Build a concrete shape record from an op, tensor ref, tensor, or shape.

    Batch symbols are NOT introduced here.  A dimension only becomes symbolic
    once the compiled :class:`~torchlens.split.shape_program.ShapeProgram`
    proves it carries batch provenance, which is projected onto graph nodes
    after compilation.

    Parameters
    ----------
    tensor_ref:
        Object exposing ``shape`` or a raw shape tuple/list.

    Returns
    -------
    SymbolicShape | None
        Shape record, or ``None`` when shape metadata is unavailable.
    """

    shape = getattr(tensor_ref, "shape", tensor_ref)
    if shape is None:
        return None
    try:
        dims = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None
    return SymbolicShape(dims)


def validate_tensor_against_symbolic_shape(
    value: Any,
    symbolic_shape: SymbolicShape | None,
    *,
    adapter: Any,
    batch_symbol: str = "B",
    backend: str = "unknown",
    split_point: str = "",
    label: str | None = None,
) -> None:
    """Validate a tensor-like value against a symbolic shape.

    A dimension spelled with ``batch_symbol`` accepts any positive extent:
    the batch dimension is unbounded, so only rank and concrete non-batch
    dimensions are enforced here.

    Parameters
    ----------
    value:
        Runtime tensor-like value.
    symbolic_shape:
        Expected symbolic shape.
    adapter:
        Backend adapter used to query shape.
    batch_symbol:
        Batch symbol expected in ``symbolic_shape``.
    backend:
        Backend name for error context.
    split_point:
        Split point for error context.
    label:
        Boundary label for error context.
    """

    del batch_symbol  # retained for caller compatibility; any non-int dim is symbolic
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
            code=SplitBoundaryError.code,
        )
    for index, (runtime_dim, expected_dim) in enumerate(zip(runtime_shape, expected, strict=True)):
        if not isinstance(expected_dim, int):
            if int(runtime_dim) < 1:
                raise SplitBoundaryError(
                    f"Boundary symbolic dimension {index} is {runtime_dim}; "
                    "a symbolic extent must be positive.",
                    context=SplitErrorContext(
                        backend=backend,
                        split_point=split_point,
                        module_path=None,
                        op_type=None,
                        layer_label=label,
                        reason="non-positive symbolic dimension",
                        traced_shape=expected,
                        runtime_shape=runtime_shape,
                    ),
                    code=SplitBoundaryError.code,
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
                code=SplitBoundaryError.code,
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


def is_batch_shape_sensitive_op(*names: str | None) -> bool:
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


__all__ = (
    "SymbolicDim",
    "SymbolicShape",
    "is_batch_shape_sensitive_op",
    "symbolic_shape_from_tensor_ref",
    "validate_tensor_against_symbolic_shape",
)
