"""Split backend adapter resolution."""

from __future__ import annotations

from importlib import import_module

from ...backends import BackendSpec
from ..errors import SplitErrorContext, SplitUnsupportedError
from .base import SegmentBundle, SplitBackendAdapter

_ADAPTER_CLASSES = {
    "torch": "TorchSplitAdapter",
    "jax": "JaxSplitAdapter",
    "tinygrad": "TinygradSplitAdapter",
    "mlx": "MlxSplitAdapter",
    "paddle": "PaddleSplitAdapter",
    "tf": "TfSplitAdapter",
}


def resolve_split_adapter(backend: BackendSpec | str) -> SplitBackendAdapter:
    """Return the split adapter for a resolved TorchLens backend.

    Parameters
    ----------
    backend:
        Backend spec or backend name.

    Returns
    -------
    SplitBackendAdapter
        Matching split backend adapter.
    """

    name = str(backend.name if isinstance(backend, BackendSpec) else backend)
    if name == "tensorflow":
        name = "tf"
    if name in _ADAPTER_CLASSES:
        return getattr(import_module(f"{__name__}.{name}"), _ADAPTER_CLASSES[name])()
    raise SplitUnsupportedError(
        f"backend={name!r} does not have a split adapter.",
        context=SplitErrorContext(
            backend=name,
            split_point="",
            reason="unknown split backend adapter",
        ),
        code=SplitUnsupportedError.code,
    )


def __getattr__(name: str) -> object:
    """Expose native adapters only when their backend is requested.

    Parameters
    ----------
    name:
        Adapter class name requested by an import consumer.

    Returns
    -------
    object
        The requested native adapter class.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy adapter attribute.
    """

    for backend, class_name in _ADAPTER_CLASSES.items():
        if name == class_name:
            return getattr(import_module(f"{__name__}.{backend}"), class_name)
    raise AttributeError(name)


__all__ = (
    "JaxSplitAdapter",
    "MlxSplitAdapter",
    "PaddleSplitAdapter",
    "SegmentBundle",
    "SplitBackendAdapter",
    "TfSplitAdapter",
    "TinygradSplitAdapter",
    "TorchSplitAdapter",
    "resolve_split_adapter",
)
