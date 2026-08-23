"""Compatibility helpers for behavior-preserving module decomposition."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from functools import lru_cache
from types import FunctionType
from typing import Any, TypeVar, cast

_F = TypeVar("_F", bound=Callable[..., Any])


def rebind_function(function: _F, namespace: dict[str, Any]) -> _F:
    """Recreate a function with a legacy module's global namespace.

    Parameters
    ----------
    function:
        Function whose code and metadata came from a split implementation module.
    namespace:
        Global namespace of the compatibility module that owns the public path.

    Returns
    -------
    Callable[..., Any]
        Equivalent function whose global lookups resolve through ``namespace``.
    """

    rebound = FunctionType(
        function.__code__,
        namespace,
        function.__name__,
        function.__defaults__,
        function.__closure__,
    )
    rebound.__annotations__ = dict(function.__annotations__)
    rebound.__dict__.update(function.__dict__)
    rebound.__doc__ = function.__doc__
    rebound.__kwdefaults__ = function.__kwdefaults__
    rebound.__module__ = namespace["__name__"]
    rebound.__qualname__ = function.__qualname__
    return cast("_F", rebound)


def rebind_contextmanager(function: _F, namespace: dict[str, Any]) -> _F:
    """Recreate a ``contextmanager`` wrapper in a legacy module namespace.

    Parameters
    ----------
    function:
        Decorated context-manager function from a split implementation module.
    namespace:
        Global namespace of the compatibility module that owns the public path.

    Returns
    -------
    Callable[..., Any]
        Equivalent context-manager factory bound to ``namespace``.
    """

    wrapped = cast("_F", getattr(function, "__wrapped__"))
    rebound = contextmanager(rebind_function(wrapped, namespace))
    return cast("_F", rebound)


def rebind_lru_cache(function: _F, namespace: dict[str, Any], *, maxsize: int) -> _F:
    """Recreate an ``lru_cache`` wrapper in a legacy module namespace.

    Parameters
    ----------
    function:
        Cached function from a split implementation module.
    namespace:
        Global namespace of the compatibility module that owns the public path.
    maxsize:
        Maximum cache size declared by the original decorator.

    Returns
    -------
    Callable[..., Any]
        Equivalent cached function bound to ``namespace``.
    """

    wrapped = cast("_F", getattr(function, "__wrapped__"))
    rebound = lru_cache(maxsize=maxsize)(rebind_function(wrapped, namespace))
    return cast("_F", rebound)
