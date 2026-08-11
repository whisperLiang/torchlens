"""Capability-derived guards for backward-only public accessors.

True-backward accessors must refuse typed on every backend whose registered
capability table declares ``backward_capture=False`` instead of returning
silently-empty results. The guarded surface is wider than the three headline
accessors: ``Op.grads`` / ``Module.grads``, ``Trace.backward_passes``,
``Trace.saved_grad_ops``, ``Trace.last_backward_pass``,
``Trace.num_saved_grad_ops``, ``Trace.saved_grad_module_calls``, and
``Trace.saved_grad_modules`` all route through this guard, so on
``backward_capture=False`` backends they raise where they previously returned
``None``/``0``/empty. The registered table, not a hardcoded backend-name set,
is the authority; the error text redirects to ``trace.derived_grads`` only
when the backend actually declares a derived-gradient surface, and an
UNREGISTERED backend name refuses typed too — fail-closed, never silently
empty.
"""

from __future__ import annotations

from typing import Any

from ..backends.registry import BackendRegistryError, get_backend_spec


def raise_if_no_backward_capture(trace: Any, *, plural_subject: str) -> None:
    """Raise when ``trace``'s backend cannot expose a true-backward accessor.

    Parameters
    ----------
    trace:
        Trace (or object carrying ``backend``) owning the accessor.
    plural_subject:
        Sentence subject describing what is unavailable, e.g.
        ``"op.grads or saved_grad_ops"``.

    Returns
    -------
    None
        Returns when the backend declares ``backward_capture=True``.

    Raises
    ------
    ValueError
        When the backend declares ``backward_capture=False``, or when the
        backend name is not registered at all — capability support cannot be
        checked for an unknown backend, so the guard fails closed instead of
        readmitting the silently-empty accessor behavior it exists to remove.
    """

    backend = str(getattr(trace, "backend", "torch"))
    try:
        capabilities = get_backend_spec(backend).capabilities
    except BackendRegistryError as exc:
        raise ValueError(
            f"trace.backend={backend!r} is not a registered backend, so "
            f"{plural_subject} availability cannot be capability-checked; "
            "refusing instead of returning silently-empty results."
        ) from exc
    if capabilities.backward_capture:
        return
    if capabilities.intermediate_derived_grads:
        raise ValueError(
            f"{backend} traces do not expose {plural_subject} because they do not "
            "capture true backward graphs. Use trace.derived_grads for leaf-level "
            "derived gradients and op.derived_grad for exact op-level derived "
            "gradients when available."
        )
    raise ValueError(
        f"{backend} traces do not expose {plural_subject} because they do not "
        f"capture true backward graphs, and the {backend} preview declares no "
        "derived-gradient surface. Use the PyTorch backend for backward capture."
    )
