"""steering-vectors bridge helpers."""

from __future__ import annotations

import inspect
from typing import Any

from ._utils import out_at


def vector(
    log: Any,
    positive_site: Any,
    negative_site: Any | None = None,
    *,
    trainer: Any | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Train or build a steering vector from saved TorchLens outs.

    Parameters
    ----------
    log:
        TorchLens ``Trace``.
    positive_site:
        Site containing positive-class outs.
    negative_site:
        Optional site containing negative-class outs.
    trainer:
        Optional callable trainer. Defaults to the installed package's
        ``train_steering_vector`` function.
    **kwargs:
        Additional keyword arguments forwarded to the trainer.

    Returns
    -------
    dict[str, Any]
        Contract payload containing the downstream steering vector.

    Raises
    ------
    ImportError
        If steering-vectors is unavailable.
    RuntimeError
        If the installed package does not expose a supported trainer.
    """

    try:
        import steering_vectors as steering_module
    except ImportError as exc:
        raise ImportError(
            "steering-vectors bridge requires the `steering` extra: install torchlens[steering]."
        ) from exc

    positive = out_at(log, positive_site)
    negative = None if negative_site is None else out_at(log, negative_site)
    train, is_default = _resolve_trainer(steering_module, trainer)
    result = _call_trainer(
        train, positive=positive, negative=negative, is_default=is_default, **kwargs
    )
    return {
        "schema": "torchlens.steering_vectors.v1",
        "vector": result,
        "positive": positive,
        "negative": negative,
    }


def _resolve_trainer(module: Any, trainer: Any | None) -> tuple[Any, bool]:
    """Return a steering-vector trainer callable and whether it is a library default.

    Parameters
    ----------
    module:
        Imported ``steering_vectors`` module.
    trainer:
        Optional explicit trainer.

    Returns
    -------
    tuple[Any, bool]
        The callable trainer and ``True`` when it was resolved as a library
        default (rather than supplied explicitly by the caller).

    Raises
    ------
    RuntimeError
        If no trainer is available.
    """

    if trainer is not None:
        return trainer, False
    candidate = getattr(module, "train_steering_vector", None)
    if callable(candidate):
        return candidate, True
    vector_cls = getattr(module, "SteeringVector", None)
    class_train = getattr(vector_cls, "train", None)
    if callable(class_train):
        return class_train, True
    raise RuntimeError("Installed steering_vectors does not expose a supported trainer.")


def _accepts_out_pair(
    trainer: Any, positive: Any, negative: Any | None, kwargs: dict[str, Any]
) -> bool:
    """Return whether ``trainer`` can bind the ``(positive, negative)`` out pair.

    Parameters
    ----------
    trainer:
        Candidate trainer callable.
    positive:
        Positive-class outs.
    negative:
        Optional negative-class outs.
    kwargs:
        Additional keyword arguments to be forwarded.

    Returns
    -------
    bool
        ``True`` when the call binds cleanly (or the signature cannot be
        introspected, in which case the caller attempts the call directly).
    """

    try:
        signature = inspect.signature(trainer)
    except (TypeError, ValueError):
        return True
    try:
        signature.bind(positive, negative, **kwargs)
    except TypeError:
        return False
    return True


def _call_trainer(
    trainer: Any, *, positive: Any, negative: Any | None, is_default: bool, **kwargs: Any
) -> Any:
    """Call a steering-vector trainer with the normalized out pair.

    Parameters
    ----------
    trainer:
        Trainer callable.
    positive:
        Positive-class outs.
    negative:
        Optional negative-class outs.
    is_default:
        Whether ``trainer`` was resolved as a steering_vectors library default.
    **kwargs:
        Additional trainer keyword arguments.

    Returns
    -------
    Any
        Downstream steering vector.

    Raises
    ------
    RuntimeError
        When the library-default trainer cannot accept the ``(positive, negative)``
        out pair. ``train_steering_vector`` takes ``(model, tokenizer,
        training_samples, ...)``, which cannot be driven from saved TorchLens outs;
        rather than silently misbind the activation tensors into those slots,
        require an explicit ``trainer``.
    """

    if is_default and not _accepts_out_pair(trainer, positive, negative, kwargs):
        raise RuntimeError(
            "The default steering_vectors trainer (train_steering_vector) expects "
            "(model, tokenizer, training_samples, ...) arguments and cannot be built "
            "from saved TorchLens outs. Pass an explicit trainer that accepts "
            "(positive_out, negative_out)."
        )
    return trainer(positive, negative, **kwargs)


__all__ = ["vector"]
