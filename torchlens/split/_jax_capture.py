"""Scoped, batch-stable lowering for JAX split captures."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Any


@contextmanager
def batch_stable_matmul() -> Iterator[None]:
    """Retain equal batch axes in dense matrix products, even at B=1.

    JAX's NumPy matmul squeezes matching singleton batch dimensions before
    dot_general. Split captures need those axes to remain explicit so their
    recipes also describe B>1. Other ranks, broadcasting, dtype promotion and
    weakly typed operands continue through the original implementation.

    Yields
    ------
    None
        A capture scope; both public and operator-dispatch functions restore
        on success or failure.
    """

    import jax
    import jax.numpy as jnp
    from jax._src.numpy import tensor_contractions

    contractions: Any = tensor_contractions
    original = contractions.matmul
    public_original = jnp.matmul

    @wraps(original)
    def matmul(a: Any, b: Any, **kwargs: Any) -> Any:
        """Keep equal dense batch prefixes in dot_general's batch dimensions."""

        a_shape = getattr(a, "shape", ())
        b_shape = getattr(b, "shape", ())
        dtype = getattr(a, "dtype", None)
        if (
            len(a_shape) >= 3
            and len(a_shape) == len(b_shape)
            and a_shape[:-2] == b_shape[:-2]
            and dtype is not None
            and dtype == getattr(b, "dtype", None)
            and jnp.issubdtype(dtype, jnp.inexact)
            and not getattr(a, "weak_type", False)
            and not getattr(b, "weak_type", False)
        ):
            rank = len(a_shape)
            batch_axes = tuple(range(rank - 2))
            return jax.lax.dot_general(
                a,
                b,
                (((rank - 1,), (rank - 2,)), (batch_axes, batch_axes)),
                **kwargs,
            )
        return original(a, b, **kwargs)

    try:
        contractions.matmul = matmul
        jnp.matmul = matmul
        yield
    finally:
        if contractions.matmul is matmul:
            contractions.matmul = original
        if jnp.matmul is matmul:
            jnp.matmul = public_original
