"""Pre-call native module and random state for eager MLX replay."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .containers import iter_arrays_with_paths, rebuild_mlx_module


def snapshot_mlx_rng() -> tuple[Any, ...]:
    """Copy and evaluate the native PRNG keys without advancing the generator."""

    import mlx.core as mx

    values = tuple(value.astype(value.dtype) for value in mx.random.state)
    mx.eval(*values)
    return values


@contextmanager
def mlx_rng_scope(values: tuple[Any, ...] | None) -> Iterator[None]:
    """Replay explicit PRNG keys, restoring the ambient generator on every exit."""

    if values is None:
        yield
        return
    previous = snapshot_mlx_rng()
    restore_mlx_rng(values)
    try:
        yield
    finally:
        restore_mlx_rng(previous)


def restore_mlx_rng(values: tuple[Any, ...]) -> None:
    """Restore native keys for both legacy lists and MLX's thread-local state sentinel."""

    import mlx.core as mx

    state = mx.random.state
    if isinstance(state, list):
        state[:] = values
    else:
        for current, value in zip(state, values, strict=True):
            current[:] = value


@dataclass(frozen=True)
class MLXCallState:
    """Native pre-call module values, stable state aliases, and optional PRNG keys."""

    module_ref: Any = None
    source_ids: tuple[tuple[int, int], ...] = ()
    rng_state: tuple[Any, ...] | None = None
    state_writes: tuple[tuple[tuple[object, ...], Any], ...] = ()


def mlx_state_writes(before: Any, after: Any) -> tuple[tuple[tuple[object, ...], Any], ...]:
    """Read array replacements made within one observed native module call.

    Parameters
    ----------
    before
        Pre-call module snapshot, retaining its original array identities.
    after
        Module immediately after the native call, before control returns to
        user code. Only writes made in this interval are attested.

    Returns
    -------
    tuple
        Structural paths and the native post-call array values.
    """

    import mlx.core as mx

    initial = {
        path: value
        for value, path in iter_arrays_with_paths(before, lambda a: isinstance(a, mx.array))
    }
    return tuple(
        (path, value)
        for value, path in iter_arrays_with_paths(after, lambda a: isinstance(a, mx.array))
        if initial.get(path) is not value
    )


def _array_owner(module: Any, path: tuple[object, ...]) -> tuple[int, tuple[object, ...]]:
    """Address an array relative to its actual owning native child module."""

    import mlx.nn as nn

    owner = module
    value = module
    local_path: list[object] = []
    for part in path:
        value = value[part]
        local_path.append(part)
        if isinstance(value, nn.Module):
            owner = value
            local_path.clear()
    return id(owner), tuple(local_path)


def capture_mlx_call_state(
    op_name: str, args: tuple[Any, ...], sources: dict[Any, int]
) -> MLXCallState:
    """Snapshot native module inputs before mutable buffers can be replaced.

    Parameters
    ----------
    op_name
        Eager wrapper's operation name.
    args
        Original positional arguments, including a native module when applicable.
    sources
        Capture-local state addresses, shared across installed eager wrappers.
    """

    import mlx.core as mx
    import mlx.nn as nn

    aliases: dict[int, int] = {}
    module = args[0] if args and isinstance(args[0], nn.Module) else None
    if module is not None:
        for value, path in iter_arrays_with_paths(module, lambda item: isinstance(item, mx.array)):
            source_id = sources.setdefault(_array_owner(module, path), id(value))
            # Different module attributes may point to the same tied parameter.
            source_id = sources.setdefault(("array", id(value)), source_id)
            aliases[id(value)] = source_id
        module = rebuild_mlx_module(module, dict(module))
    return MLXCallState(
        module_ref=module,
        source_ids=tuple(aliases.items()),
        rng_state=snapshot_mlx_rng()
        if op_name == "dropout" or op_name.startswith("random_")
        else None,
    )


__all__ = ("MLXCallState", "capture_mlx_call_state", "mlx_rng_scope", "snapshot_mlx_rng")
