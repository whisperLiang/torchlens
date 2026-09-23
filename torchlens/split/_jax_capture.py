"""Scoped, batch-stable lowering for JAX split captures."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from threading import RLock
from typing import Any

_MATMUL_CAPTURE_LOCK = RLock()


def _operator_dispatch(operators: Any, name: str, matmul: Callable[..., Any]) -> Any:
    """Build the installed JAX dispatch wrapper for a matmul direction.

    Parameters
    ----------
    operators:
        JAX's private array-methods module.
    name:
        ``matmul`` or ``rmatmul``.
    matmul:
        Scoped batch-stable matmul implementation.

    Returns
    -------
    Any
        A callable suitable for an array or abstract-value descriptor.
    """

    defer = getattr(operators, "_defer_to_unrecognized_arg", None)
    if defer is not None:
        return defer("@", matmul, swap=name == "rmatmul")

    # JAX 0.11 removed the private closure factory and exposes the already
    # installed operators instead. They resolve tensor_contractions.matmul at
    # call time, so they naturally use the scoped implementation above. Wrap
    # them to give each capture a fresh descriptor and preserve restoration
    # checks, while retaining JAX's accepted-operand policy.
    native_operator = getattr(operators, f"_operator_{name}")

    @wraps(native_operator)
    def dispatch(a: Any, b: Any) -> Any:
        """Delegate operands to JAX's native operator implementation."""

        return native_operator(a, b)

    return dispatch


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

    Notes
    -----
    JAX installs array operators with closures over its original matmul.
    Replacing the module function alone cannot reach eager arrays or tracers;
    their operator descriptors need independent, temporary replacements too.
    A reentrant lock covers the complete scope so concurrent captures cannot
    save another thread's temporary descriptors and later leak them on restore.
    """

    with _MATMUL_CAPTURE_LOCK, _patched_matmul():
        yield


@contextmanager
def _patched_matmul() -> Iterator[None]:
    """Install batch-stable entry points while the caller owns the capture lock.

    Yields
    ------
    None
        A scope whose original raw descriptors restore on exit.
    """

    import jax
    import jax.numpy as jnp
    from jax._src import core
    from jax._src.array import ArrayImpl
    from jax._src.numpy import array_methods, tensor_contractions

    contractions: Any = tensor_contractions
    operators: Any = array_methods
    original = contractions.matmul
    public_original = jnp.matmul
    operator_patches: list[tuple[Any, str, Any, Any]] = []

    @wraps(original)
    def matmul(a: Any, b: Any, **kwargs: Any) -> Any:
        """Keep equal dense batch prefixes in dot_general's batch dimensions.

        Parameters
        ----------
        a, b:
            Matrix operands, delegated unchanged outside the dense equal-batch case.
        **kwargs:
            Native precision, preferred dtype, and sharding options.
        """

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
        for name in ("matmul", "rmatmul"):
            # Reuse JAX's operand dispatch so reflected operations, __jax_array__,
            # rejected container types and NotImplemented retain native behavior.
            # JAX 0.11 removed the private ``_defer_to_unrecognized_arg`` helper
            # and replaced it with the explicit ``_operator_*`` functions.  The
            # latter are preferable where available: they contain JAX's current
            # accepted-operand and ``__jax_array__`` policy, while their matmul
            # call still resolves through the patched tensor_contractions module.
            operator = _operator_dispatch(operators, name, matmul)
            for owner, attribute, replacement in (
                (ArrayImpl, f"__{name}__", operator),
                (getattr(core, "ShapedArray", None), f"_{name}", staticmethod(operator)),
                (getattr(core, "DShapedArray", None), f"_{name}", staticmethod(operator)),
            ):
                # Dynamic aval classes vary across JAX releases. Only replace
                # descriptors actually owned here; inherited ones follow the base.
                if owner is None or attribute not in vars(owner):
                    continue
                # Keep the raw descriptor, not getattr's bound/unwrapped value.
                prior = vars(owner)[attribute]
                setattr(owner, attribute, replacement)
                operator_patches.append((owner, attribute, prior, replacement))
        yield
    finally:
        for owner, attribute, prior, replacement in reversed(operator_patches):
            if vars(owner).get(attribute) is replacement:
                setattr(owner, attribute, prior)
        if contractions.matmul is matmul:
            contractions.matmul = original
        if jnp.matmul is matmul:
            jnp.matmul = public_original
