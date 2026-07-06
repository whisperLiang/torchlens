"""Validation helpers for split replay outputs."""

from __future__ import annotations

from typing import Any


def nested_allclose(
    adapter: Any,
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
) -> bool:
    """Recursively compare tensor structures using a split adapter."""

    if adapter.is_tensor(left) or adapter.is_tensor(right):
        return adapter.allclose(left, right, atol=atol, rtol=rtol)
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(
            nested_allclose(adapter, left[key], right[key], atol=atol, rtol=rtol) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, type(left)):
        if len(left) != len(right):
            return False
        return all(
            nested_allclose(adapter, left_item, right_item, atol=atol, rtol=rtol)
            for left_item, right_item in zip(left, right)
        )
    return left == right


__all__ = ["nested_allclose"]
