"""Stable JAX pytree path encodings for capture and output containers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...ir.container import DictKey, TupleIndex


def _path_to_string(path: Sequence[Any]) -> str:
    """Convert a JAX pytree path to a stable dotted string.

    Parameters
    ----------
    path
        JAX pytree path entries.

    Returns
    -------
    str
        Dotted path string.
    """

    if not path:
        return "root"
    parts: list[str] = []
    for entry in path:
        name = getattr(entry, "name", None)
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if name is not None:
            parts.append(str(name))
        elif key is not None:
            parts.append(str(key))
        elif idx is not None:
            parts.append(str(idx))
        else:
            parts.append(str(entry).strip("[]'"))
    return ".".join(parts)


def _jax_dict_keys(value: Mapping[Any, Any]) -> tuple[Any, ...]:
    """Return dict keys in JAX builtin pytree traversal order.

    Parameters
    ----------
    value
        Builtin dict output.

    Returns
    -------
    tuple[Any, ...]
        Keys sorted when possible, matching JAX's dict pytree order.
    """

    try:
        return tuple(sorted(value.keys()))
    except TypeError:
        return tuple(value.keys())


def _path_to_components(path: Sequence[Any]) -> tuple[object, ...]:
    """Convert a JAX pytree path to TorchLens output-path components.

    Parameters
    ----------
    path
        JAX pytree path entries.

    Returns
    -------
    tuple[object, ...]
        Backend-neutral container path components.
    """

    components: list[object] = []
    for entry in path:
        name = getattr(entry, "name", None)
        key = getattr(entry, "key", None)
        idx = getattr(entry, "idx", None)
        if name is not None:
            components.append(str(name))
        elif key is not None:
            components.append(DictKey(key))
        elif idx is not None:
            components.append(TupleIndex(int(idx)))
        else:
            components.append(str(entry).strip("[]'"))
    return tuple(components)
