"""Transactional native MLX model and PRNG state for split preparation."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from .. import _state
from ..backends.mlx._call_state import mlx_rng_scope, snapshot_mlx_rng
from ._state_snapshot import PythonStateSnapshot


@contextmanager
def mlx_capture_state(model: Any) -> Iterator[None]:
    """Restore native module bindings, in-place array values, and random state after capture.

    Parameters
    ----------
    model
        Native module, function, or callable with reachable MLX state.
    """

    import mlx.core as mx
    import mlx.nn as nn

    modules: list[tuple[Any, dict[str, Any], dict[str, Any]]] = []
    arrays: list[tuple[Any, Any]] = []

    def visit_native(value: Any, visit: Callable[[Any], None]) -> bool:
        """Snapshot native containers without calling their specialized update methods."""

        if isinstance(value, mx.array):
            saved = mx.stop_gradient(value).astype(value.dtype)
            mx.eval(saved)
            arrays.append((value, saved))
            return True
        if isinstance(value, nn.Module):
            contents, attrs = dict(value), dict(vars(value))
            modules.append((value, contents, attrs))
            for item in (*contents.values(), *attrs.values()):
                visit(item)
            return True
        return False

    python_state = PythonStateSnapshot(visit_native, preserve_tl=True)
    with _state.pause_logging():
        python_state.visit(model)
        rng = snapshot_mlx_rng()
    try:
        with mlx_rng_scope(rng):
            yield
    finally:
        with _state.pause_logging():
            for value, saved in arrays:
                if not bool(mx.array_equal(value, saved, equal_nan=True)):
                    value[tuple(slice(None) for _ in value.shape)] = saved
            for module, contents, attrs in reversed(modules):
                module.clear()
                dict.update(module, contents)
                vars(module).clear()
                vars(module).update(attrs)
            python_state.restore()


__all__ = ["mlx_capture_state"]
