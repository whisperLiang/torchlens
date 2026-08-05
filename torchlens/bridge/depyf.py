"""depyf bridge helpers."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def dump(model: Any, x: Any, path: str | Path | None = None, **kwargs: Any) -> Any:
    """Dump depyf source/graph context for a model and example input.

    This is a companion bridge, not TorchLens native ``torch.compile`` support.
    TorchLens still reports ``torch.compile`` capture as SCOPE in
    ``tl.compat.report``.

    Parameters
    ----------
    model:
        Model or compiled model to inspect with depyf.
    x:
        Example input passed through to depyf when the installed API accepts it.
    path:
        Optional output directory.
    **kwargs:
        Additional keyword arguments forwarded to depyf's dump function.

    Returns
    -------
    Any
        depyf return value.

    Raises
    ------
    ImportError
        If depyf is unavailable.
    RuntimeError
        If the installed depyf package does not expose a supported dump entrypoint.
    """

    try:
        import depyf as depyf_module
    except ImportError as exc:
        raise ImportError(
            "depyf bridge requires the `depyf` extra: install torchlens[depyf]."
        ) from exc

    output_dir = Path(path) if path is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    for attr_name in ("dump", "decompile", "prepare_debug"):
        candidate = getattr(depyf_module, attr_name, None)
        if callable(candidate):
            if output_dir is None:
                full_args: tuple[Any, ...] = (model, x)
                reduced_args: tuple[Any, ...] = (model,)
            else:
                full_args = (model, x, output_dir)
                reduced_args = (model, output_dir)
            return _call_depyf_entrypoint(candidate, full_args, reduced_args, kwargs)

    raise RuntimeError("Installed depyf does not expose dump, decompile, or prepare_debug.")


def _can_bind(signature: inspect.Signature, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    """Return whether ``args``/``kwargs`` bind cleanly to ``signature``.

    Parameters
    ----------
    signature:
        Callable signature.
    args:
        Positional arguments.
    kwargs:
        Keyword arguments.

    Returns
    -------
    bool
        ``True`` when binding succeeds without a ``TypeError``.
    """

    try:
        signature.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


def _call_depyf_entrypoint(
    candidate: Any,
    full_args: tuple[Any, ...],
    reduced_args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    """Call a depyf entrypoint, choosing arity by signature rather than by exception.

    The example input ``x`` is passed when the entrypoint's signature accepts it.
    A ``TypeError`` raised *inside* the entrypoint body (a genuine depyf failure) is
    never confused with an argument-arity mismatch: we decide the arity up front so
    the entrypoint runs exactly once and its own errors propagate unchanged. This
    prevents the old fallback from silently re-running the entrypoint without the
    user's example input and returning a semantically different result.

    Parameters
    ----------
    candidate:
        Selected depyf entrypoint callable.
    full_args:
        Positional arguments including the example input (and optional output dir).
    reduced_args:
        Positional arguments with the example input dropped.
    kwargs:
        Keyword arguments forwarded to the entrypoint.

    Returns
    -------
    Any
        The entrypoint return value.
    """

    try:
        signature: inspect.Signature | None = inspect.signature(candidate)
    except (TypeError, ValueError):
        signature = None

    if signature is not None:
        if _can_bind(signature, full_args, kwargs):
            return candidate(*full_args, **kwargs)
        if _can_bind(signature, reduced_args, kwargs):
            return candidate(*reduced_args, **kwargs)
        # Neither arity binds cleanly; call with the full arity so the entrypoint's
        # own TypeError (naming the real mismatch) surfaces instead of being masked.
        return candidate(*full_args, **kwargs)

    # No introspectable signature (e.g. some C-level callables): attempt the full
    # arity and only retry the reduced arity when the TypeError was raised binding
    # our arguments (no callee frame), never when it came from inside the callee.
    try:
        return candidate(*full_args, **kwargs)
    except TypeError as exc:
        traceback = exc.__traceback__
        if traceback is not None and traceback.tb_next is not None:
            raise
        return candidate(*reduced_args, **kwargs)


__all__ = ["dump"]
