"""repeng bridge helpers."""

from __future__ import annotations

import inspect
from typing import Any

from ._utils import out_at


def control_vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    vector_factory: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a repeng control vector from saved TorchLens outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    positive_site:
        Site containing positive outs.
    negative_site:
        Optional site containing negative outs.
    vector_factory:
        Optional downstream factory or class.
    **kwargs:
        Additional keyword arguments forwarded to the downstream factory.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the downstream control vector.

    Raises
    ------
    ImportError
        If repeng is unavailable.
    RuntimeError
        If no supported factory is exposed.
    """

    try:
        import repeng as repeng_module
    except ImportError as exc:
        raise ImportError(
            "repeng bridge requires the `repeng` extra: install torchlens[repeng]."
        ) from exc

    positive = out_at(log, positive_site)
    negative = None if negative_site is None else out_at(log, negative_site)
    factory, is_default = _resolve_factory(repeng_module, vector_factory)
    result = _call_factory(
        factory, positive=positive, negative=negative, is_default=is_default, **kwargs
    )
    return {
        "schema": "torchlens.repeng.v1",
        "control_vector": result,
        "positive": positive,
        "negative": negative,
    }


def _resolve_factory(module: Any, vector_factory: Any | None) -> tuple[Any, bool]:
    """Return a repeng control-vector factory and whether it is a library default.

    Parameters
    ----------
    module:
        Imported ``repeng`` module.
    vector_factory:
        Optional explicit factory.

    Returns
    -------
    tuple[Any, bool]
        The callable factory and ``True`` when it was resolved as a library
        default (rather than supplied explicitly by the caller).

    Raises
    ------
    RuntimeError
        If no factory is available.
    """

    if vector_factory is not None:
        return vector_factory, False
    control_vector_cls = getattr(module, "ControlVector", None)
    train = getattr(control_vector_cls, "train", None)
    if callable(train):
        return train, True
    if callable(control_vector_cls):
        return control_vector_cls, True
    raise RuntimeError("Installed repeng does not expose ControlVector or ControlVector.train.")


def _accepts_out_pair(
    factory: Any, positive: Any, negative: Any | None, kwargs: dict[str, Any]
) -> bool:
    """Return whether ``factory`` can bind the ``(positive, negative)`` out pair.

    Parameters
    ----------
    factory:
        Candidate factory callable.
    positive:
        Positive outs.
    negative:
        Optional negative outs.
    kwargs:
        Additional keyword arguments to be forwarded.

    Returns
    -------
    bool
        ``True`` when the call binds cleanly (or the signature cannot be
        introspected, in which case the caller attempts the call directly).
    """

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(positive, negative, **kwargs)
    except TypeError:
        return False
    return True


def _call_factory(
    factory: Any, *, positive: Any, negative: Any | None, is_default: bool, **kwargs: Any
) -> Any:
    """Call a repeng vector factory with the normalized out pair.

    Parameters
    ----------
    factory:
        Factory callable.
    positive:
        Positive outs.
    negative:
        Optional negative outs.
    is_default:
        Whether ``factory`` was resolved as a repeng library default.
    **kwargs:
        Additional factory keyword arguments.

    Returns
    -------
    Any
        Downstream vector object.

    Raises
    ------
    RuntimeError
        When the library-default factory cannot accept the ``(positive, negative)``
        out pair. repeng's ``ControlVector.train`` takes ``(model, tokenizer,
        dataset, ...)``, which cannot be driven from saved TorchLens outs; rather
        than silently misbind the activation tensors into those slots, require an
        explicit ``vector_factory``.
    """

    if is_default and not _accepts_out_pair(factory, positive, negative, kwargs):
        raise RuntimeError(
            "The default repeng factory (ControlVector.train) expects "
            "(model, tokenizer, dataset, ...) arguments and cannot be built from "
            "saved TorchLens outs. Pass an explicit vector_factory that accepts "
            "(positive_out, negative_out)."
        )
    return factory(positive, negative, **kwargs)


__all__ = ["control_vector"]
