"""Audited JIT-call parameter, donation, and sharding predicates."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

JIT_CALL_PARAM_NAMES = frozenset(
    {
        "compiler_options_kvs",
        "ctx_mesh",
        "donated_invars",
        "in_layouts",
        "in_shardings",
        "inline",
        "jaxpr",
        "keep_unused",
        "name",
        "out_layouts",
        "out_shardings",
    }
)


def _jit_call_params_are_known(eqn: Any) -> bool:
    """Return whether a JIT-shaped call uses only audited parameters.

    Parameters
    ----------
    eqn
        Candidate call equation.

    Returns
    -------
    bool
        True when all parameter names are part of the audited JIT-call frame.
    """

    return set(eqn.params) <= JIT_CALL_PARAM_NAMES


def _has_donated_invars(donated_invars: Any) -> bool:
    """Return whether any nested JIT input is donated.

    Parameters
    ----------
    donated_invars
        Donation flags from the JAX call equation.

    Returns
    -------
    bool
        True when any donation flag is truthy.
    """

    if isinstance(donated_invars, bool):
        return donated_invars
    if isinstance(donated_invars, Sequence):
        return any(bool(flag) for flag in donated_invars)
    return bool(donated_invars)


def _jit_shardings_are_unspecified(shardings: Any) -> bool:
    """Return whether a JIT sharding parameter is fully unspecified.

    Parameters
    ----------
    shardings
        JAX ``in_shardings`` or ``out_shardings`` parameter value.

    Returns
    -------
    bool
        True when every entry is JAX's unspecified sharding sentinel.
    """

    if shardings is None:
        return True
    if not isinstance(shardings, Sequence) or isinstance(shardings, str):
        shardings = (shardings,)
    return all(_is_unspecified_jax_sharding(sharding) for sharding in shardings)


def _is_unspecified_jax_sharding(sharding: Any) -> bool:
    """Return whether a value is JAX's unspecified sharding sentinel.

    Parameters
    ----------
    sharding
        Candidate sharding value.

    Returns
    -------
    bool
        True for the JAX ``UnspecifiedValue`` sentinel.
    """

    return type(sharding).__name__ == "UnspecifiedValue"
