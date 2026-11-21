"""Thread-safe registration and lazy resolution of native optimizer adapters."""

from __future__ import annotations

from importlib import import_module
from threading import RLock
from typing import Any

from .base import OptimizerStateAdapter, _refuse

_BUILTINS = {"torch": ("torch", ".torch"), "tf": ("keras", ".tensorflow")}
_CUSTOM: dict[tuple[str, str], OptimizerStateAdapter] = {}
_LOCK = RLock()


def _backend_key(backend: str) -> str:
    """Normalize the TensorFlow alias to the split runtime's backend key."""

    return "tf" if backend == "tensorflow" else backend


def register_optimizer_adapter(backend: str, adapter: OptimizerStateAdapter, /) -> None:
    """Register a uniquely named native optimizer adapter.

    Examples
    --------
    >>> from torchlens.split.optimizers import get_optimizer_adapter, register_optimizer_adapter
    >>> opt = torch.optim.SGD(model.parameters(), lr=0.1)
    >>> callable(register_optimizer_adapter), get_optimizer_adapter('torch', opt).name
    (True, 'torch')
    """

    backend = _backend_key(backend)
    key = (backend, adapter.name)
    with _LOCK:
        if key in _CUSTOM or _BUILTINS.get(backend, (None,))[0] == adapter.name:
            _refuse(
                "optimizer_adapter_conflict",
                f"Optimizer adapter {key!r} is already registered.",
                backend,
            )
        _CUSTOM[key] = adapter


def _adapters(backend: str) -> dict[str, OptimizerStateAdapter]:
    """Load only the requested backend's built-in and registered extensions."""

    with _LOCK:
        adapters = {name: adapter for (key, name), adapter in _CUSTOM.items() if key == backend}
    if backend in _BUILTINS:
        name, module = _BUILTINS[backend]
        adapters[name] = import_module(module, __package__)._ADAPTER
    return adapters


def get_optimizer_adapter(backend: str, optimizer: Any, /) -> OptimizerStateAdapter:
    """Resolve an optimizer to exactly one registered adapter.

    Examples
    --------
    >>> from torchlens.split.optimizers import get_optimizer_adapter
    >>> optimizer = torch.optim.Adam(model.parameters())
    >>> get_optimizer_adapter('torch', optimizer).supports(optimizer)
    True
    """

    backend = _backend_key(backend)
    matches = [adapter for adapter in _adapters(backend).values() if adapter.supports(optimizer)]
    if len(matches) == 1:
        return matches[0]
    reason = "no registered adapter" if not matches else "multiple adapters match"
    _refuse(
        "optimizer_adapter_resolution",
        f"{type(optimizer).__name__}: {reason} for {backend!r}.",
        backend,
    )


def _adapter_by_name(backend: str, name: str) -> OptimizerStateAdapter:
    """Resolve an explicitly recorded adapter name without optimizer guessing."""

    adapter = _adapters(backend).get(name)
    if adapter is None:
        _refuse("optimizer_adapter_resolution", f"Unknown optimizer adapter {name!r}.", backend)
    return adapter


def _has_builtin_factory(backend: str, name: str) -> bool:
    """Return whether TorchLens can reconstruct this registered adapter's native types."""

    builtin = _BUILTINS.get(_backend_key(backend))
    return builtin is not None and builtin[0] == name
