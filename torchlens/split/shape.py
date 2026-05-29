"""Symbolic shape helpers for TorchLens split runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .errors import SplitBoundaryError

Dimension = int | str
SymbolicShape = tuple[Dimension, ...]


@dataclass(frozen=True)
class ShapeEnv:
    """Batch-symbolic shape policy for one split trace."""

    batch_symbol: str = "B"
    traced_batch_size: int | None = None
    dynamic_batch: tuple[int, int] | None = None

    def symbolic_shape(self, shape: tuple[int, ...] | None) -> SymbolicShape | None:
        """Return shape with dim 0 replaced by the batch symbol when applicable."""

        if shape is None:
            return None
        if self.traced_batch_size is None or not shape:
            return shape
        if shape[0] == self.traced_batch_size:
            return (self.batch_symbol, *shape[1:])
        return shape

    def validate_batch(self, batch_size: int) -> None:
        """Validate a concrete batch size against ``dynamic_batch``."""

        if self.dynamic_batch is None:
            return
        low, high = self.dynamic_batch
        if not low <= batch_size <= high:
            raise SplitBoundaryError(
                f"Batch size {batch_size} is outside dynamic_batch range [{low}, {high}]."
            )


def infer_traced_batch_size(inputs: tuple[Any, ...]) -> int | None:
    """Infer trace batch from the first tensor input."""

    for value in _walk_values(inputs):
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    return None


def batch_size_from_value(value: Any) -> int | None:
    """Return dim-0 batch size from the first tensor in ``value``."""

    for item in _walk_values((value,)):
        if isinstance(item, torch.Tensor) and item.ndim > 0:
            return int(item.shape[0])
    return None


def validate_tensor_against_symbolic_shape(
    tensor: torch.Tensor,
    symbolic_shape: SymbolicShape | None,
    *,
    dtype: torch.dtype | None,
    shape_env: ShapeEnv,
    name: str,
) -> int | None:
    """Validate tensor shape and dtype without checking device."""

    if dtype is not None and tensor.dtype != dtype:
        raise SplitBoundaryError(
            f"{name} dtype {tensor.dtype!r} does not match expected dtype {dtype!r}."
        )
    if symbolic_shape is None:
        return batch_size_from_value(tensor)
    actual = tuple(int(dim) for dim in tensor.shape)
    if len(actual) != len(symbolic_shape):
        raise SplitBoundaryError(
            f"{name} rank {len(actual)} does not match expected rank {len(symbolic_shape)}."
        )
    actual_batch: int | None = None
    for index, (actual_dim, expected_dim) in enumerate(zip(actual, symbolic_shape, strict=True)):
        if expected_dim == shape_env.batch_symbol:
            actual_batch = actual_dim
            shape_env.validate_batch(actual_dim)
            continue
        if actual_dim != expected_dim:
            raise SplitBoundaryError(
                f"{name} shape mismatch at dim {index}: got {actual!r}, "
                f"expected symbolic shape {symbolic_shape!r}."
            )
    return actual_batch


def _walk_values(values: Any) -> Any:
    if isinstance(values, dict):
        for item in values.values():
            yield from _walk_values(item)
        return
    if isinstance(values, (tuple, list)):
        for item in values:
            yield from _walk_values(item)
        return
    yield values


__all__ = [
    "Dimension",
    "ShapeEnv",
    "SymbolicShape",
    "batch_size_from_value",
    "infer_traced_batch_size",
    "validate_tensor_against_symbolic_shape",
]
