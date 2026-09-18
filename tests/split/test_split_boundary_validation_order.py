"""Boundary guard order and solved-shape behavior survive helper extraction."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from torchlens.split.boundary import ReplayBoundary
from torchlens.split.errors import SplitBoundaryError
from torchlens.split.ir import BoundarySchema
from torchlens.split.shape import SymbolicShape
from torchlens.split.shape_program import DimExpr, ShapeBinding, TensorShapeIR


def _schema() -> BoundarySchema:
    """Return a schema with all identity fields populated."""

    return BoundarySchema(
        value_id="h",
        container_path=("features", 0),
        role="primary",
        shape=SymbolicShape(("B", 3)),
        dtype="torch.float32",
        output_index=0,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"value_id": "other", "role": "skip", "output_index": 1, "container_path": ()},
            "canonical ID",
        ),
        ({"role": "skip", "output_index": 1, "container_path": ()}, "role"),
        ({"output_index": 1, "container_path": ()}, "output index"),
        ({"container_path": ()}, "container path"),
    ],
)
def test_boundary_identity_errors_precede_dtype_and_shape(
    changes: dict[str, Any], message: str
) -> None:
    """The earliest mismatched identity wins even when payload validation fails."""

    expected = _schema()
    boundary = ReplayBoundary(
        backend="torch",
        tensors={"h": torch.ones(2, 7, dtype=torch.float64)},
        spec={"h": replace(expected, **changes)},
        metadata={},
    )
    with pytest.raises(SplitBoundaryError) as exc_info:
        boundary.validate({"h": expected})
    assert str(exc_info.value) == f"Boundary {message} mismatch for 'h'."


@pytest.mark.parametrize("has_shape", [False, True])
@pytest.mark.parametrize("has_batch", [False, True])
def test_solved_shape_requires_both_shape_and_batch_evidence(
    has_shape: bool, has_batch: bool
) -> None:
    """Shape solutions are checked only when both pre-existing inputs are available."""

    expected = _schema()
    boundary = ReplayBoundary(
        backend="torch",
        tensors={"h": torch.ones(2, 7)},
        spec={"h": expected},
        metadata={"runtime_batch_size": 2} if has_batch else {},
    )
    shape = TensorShapeIR("h", (DimExpr.symbol("B"), DimExpr.const(3)))
    program = SimpleNamespace(
        value_shapes={"h": shape} if has_shape else {},
        binding_from_batch=lambda batch: ShapeBinding(symbols={"B": batch}, input_shapes={}),
    )
    if has_shape and has_batch:
        with pytest.raises(SplitBoundaryError, match="expected solved shape"):
            boundary.validate(shape_program=program)
    else:
        boundary.validate(shape_program=program)
