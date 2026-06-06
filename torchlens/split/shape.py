"""Symbolic shape helpers for TorchLens split runtimes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch

from .errors import SplitBoundaryError

Dimension = int | str
SymbolicShape = tuple[Dimension, ...]
MAX_BATCH_MULTIPLIER = 16
MAX_VALIDATION_BATCH_MULTIPLIER = 512


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
        if shape[0] > 0 and shape[0] % self.traced_batch_size == 0:
            multiplier = shape[0] // self.traced_batch_size
            if 1 < multiplier <= MAX_BATCH_MULTIPLIER:
                return (f"{self.batch_symbol}*{multiplier}", *shape[1:])
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

    for value in iter_tensor_leaves(inputs):
        if value.ndim > 0:
            return int(value.shape[0])
    return None


def batch_size_from_value(value: Any) -> int | None:
    """Return dim-0 batch size from the first tensor in ``value``."""

    for item in iter_tensor_leaves(value):
        if item.ndim > 0:
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
        if (
            shape_env.traced_batch_size is not None
            and expected_dim == shape_env.traced_batch_size
            and index != len(symbolic_shape) - 1
        ):
            actual_batch = actual_dim
            shape_env.validate_batch(actual_dim)
            continue
        folded_multiplier = _concrete_batch_multiplier(expected_dim, shape_env)
        if folded_multiplier is not None and index != len(symbolic_shape) - 1:
            if actual_dim % folded_multiplier != 0:
                raise SplitBoundaryError(
                    f"{name} shape mismatch at dim {index}: got {actual!r}, "
                    f"expected symbolic shape {symbolic_shape!r}."
                )
            actual_batch = actual_dim // folded_multiplier
            shape_env.validate_batch(actual_batch)
            continue
        batch_multiplier = _batch_multiplier(expected_dim, shape_env.batch_symbol)
        if batch_multiplier is not None:
            if actual_dim % batch_multiplier != 0:
                raise SplitBoundaryError(
                    f"{name} shape mismatch at dim {index}: got {actual!r}, "
                    f"expected symbolic shape {symbolic_shape!r}."
                )
            actual_batch = actual_dim // batch_multiplier
            shape_env.validate_batch(actual_batch)
            continue
        if actual_dim != expected_dim:
            raise SplitBoundaryError(
                f"{name} shape mismatch at dim {index}: got {actual!r}, "
                f"expected symbolic shape {symbolic_shape!r}."
            )
    return actual_batch


def _batch_multiplier(expected_dim: Dimension, batch_symbol: str) -> int | None:
    if not isinstance(expected_dim, str):
        return None
    prefix = f"{batch_symbol}*"
    if not expected_dim.startswith(prefix):
        return None
    try:
        multiplier = int(expected_dim[len(prefix) :])
    except ValueError:
        return None
    return multiplier if multiplier > 1 else None


def _concrete_batch_multiplier(expected_dim: Dimension, shape_env: ShapeEnv) -> int | None:
    if not isinstance(expected_dim, int) or shape_env.traced_batch_size is None:
        return None
    if expected_dim <= 0 or expected_dim % shape_env.traced_batch_size != 0:
        return None
    multiplier = expected_dim // shape_env.traced_batch_size
    if multiplier <= 1 or multiplier > MAX_VALIDATION_BATCH_MULTIPLIER:
        return None
    return multiplier


def iter_tensor_leaves(value: Any) -> Iterator[torch.Tensor]:
    """Yield tensor leaves from common Python and model input containers."""

    yield from _iter_tensor_leaves(value, seen=set())


def _iter_tensor_leaves(value: Any, *, seen: set[int]) -> Iterator[torch.Tensor]:
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_tensor_leaves(item, seen=seen)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensor_leaves(item, seen=seen)
        return
    if isinstance(value, torch.nn.Module):
        return
    semantic_leaves: list[torch.Tensor] = []
    for attr_name in ("tensors", "mask"):
        item = getattr(value, attr_name, None)
        if item is not None:
            semantic_leaves.extend(_iter_tensor_leaves(item, seen=seen))
    if semantic_leaves:
        yield from semantic_leaves
    if is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            yield from _iter_tensor_leaves(getattr(value, field.name), seen=seen)
        return
    try:
        attrs = vars(value)
    except TypeError:
        attrs = {}
    if attrs:
        for item in attrs.values():
            yield from _iter_tensor_leaves(item, seen=seen)


__all__ = [
    "Dimension",
    "MAX_BATCH_MULTIPLIER",
    "ShapeEnv",
    "SymbolicShape",
    "batch_size_from_value",
    "infer_traced_batch_size",
    "iter_tensor_leaves",
    "validate_tensor_against_symbolic_shape",
]
