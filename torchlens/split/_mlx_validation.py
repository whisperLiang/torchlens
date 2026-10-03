"""State-preserving native MLX equivalence checks for stochastic split replay."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from .. import _state
from ..backends.mlx._call_state import mlx_rng_scope, snapshot_mlx_rng
from ._mlx_capture import mlx_capture_state
from .validation import nested_allclose


@contextmanager
def mlx_runtime_state(runtime: Any) -> Iterator[None]:
    """Restore the split runtime's effective buffers and lazy state after an internal check."""

    bindings = [runtime.segments.prefix._binding, runtime.segments.suffix._binding]
    coordinator = bindings[0].coordinator
    values, mutable = dict(coordinator.values), set(coordinator.mutable_sources)
    states = [
        (binding.state, dict(binding.state._entries), binding.state.version) for binding in bindings
    ]
    try:
        yield
    finally:
        coordinator.values, coordinator.mutable_sources = values, mutable
        for state, entries, version in states:
            state._entries, state.version = entries, version


def validate_mlx_equivalence(runtime: Any, model: Any, inputs: Any, **options: Any) -> bool:
    """Compare identical native random draws without changing either model's running state.

    Parameters
    ----------
    runtime, model, inputs
        Prepared MLX split, native reference callable, and input argument tuple.
    **options
        Runtime keyword arguments and numeric comparison tolerances.
    """

    import mlx.core as mx

    from ._mlx_training import _map_arrays

    with _state.pause_logging():
        rng = snapshot_mlx_rng()
        with mlx_capture_state(model):
            expected = _map_arrays(
                model(*inputs, **(options["input_kwargs"] or {})),
                lambda value: mx.stop_gradient(value).astype(value.dtype),
            )
            mx.eval(expected)
        with mlx_runtime_state(runtime), mlx_rng_scope(rng):
            actual = runtime.replay(*inputs, input_kwargs=options["input_kwargs"])
            mx.eval(actual)
            return nested_allclose(
                runtime.adapter, expected, actual, atol=options["atol"], rtol=options["rtol"]
            )


__all__ = ["validate_mlx_equivalence"]
