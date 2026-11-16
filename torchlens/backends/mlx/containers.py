"""Container traversal helpers for the MLX capture backend."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from typing import Any


def iter_arrays_with_paths(
    value: object,
    is_tensor: Callable[[object], bool],
    path: tuple[object, ...] = (),
    *,
    sort_dict: bool = False,
) -> list[tuple[object, tuple[object, ...]]]:
    """Return MLX array leaves paired with their container paths.

    Parameters
    ----------
    value
        Candidate value or builtin container.
    is_tensor
        Backend-owned MLX array predicate.
    path
        Path accumulated by the recursive traversal.
    sort_dict
        Sort dictionary keys when matching input flattening order.
    """

    if is_tensor(value):
        return [(value, path)]
    if isinstance(value, (list, tuple)):
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in iter_arrays_with_paths(item, is_tensor, (*path, index), sort_dict=sort_dict)
        ]
    if isinstance(value, dict):
        items = (
            value.items()
            if not sort_dict
            else sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
        return [
            leaf
            for key, item in items
            for leaf in iter_arrays_with_paths(item, is_tensor, (*path, key), sort_dict=sort_dict)
        ]
    return []


def rebuild_mlx_module(module: Any, template: dict[str, Any]) -> Any:
    """Restore native module types without writing values into the captured model.

    Parameters
    ----------
    module
        Native module retained by the live call sidecar.
    template
        Captured module contents with replay array leaves substituted.
    """

    import mlx.nn as nn

    def rebuild(original: Any, value: Any) -> Any:
        """Restore module classes while retaining the captured container contents."""

        if isinstance(original, nn.Module) and isinstance(value, dict):
            restored = copy(original)
            if hasattr(original, "_no_grad"):
                restored._no_grad = set(original._no_grad)
            restored.clear()
            dict.update(
                restored, {key: rebuild(original.get(key), item) for key, item in value.items()}
            )
            return restored
        if isinstance(value, dict):
            return {
                key: rebuild(original.get(key) if isinstance(original, dict) else None, item)
                for key, item in value.items()
            }
        if isinstance(value, (tuple, list)):
            return type(value)(
                rebuild(
                    original[index]
                    if isinstance(original, (tuple, list)) and index < len(original)
                    else None,
                    item,
                )
                for index, item in enumerate(value)
            )
        return value

    return rebuild(module, template)


__all__ = (
    "iter_arrays_with_paths",
    "rebuild_mlx_module",
)
