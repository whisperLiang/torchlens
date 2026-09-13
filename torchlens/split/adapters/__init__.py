"""Split backend adapter resolution."""

from __future__ import annotations

from ...backends import BackendSpec
from ..errors import SplitErrorContext, SplitUnsupportedError
from .base import SegmentBundle, SplitBackendAdapter
from .jax import JaxSplitAdapter
from .mlx import MlxSplitAdapter
from .paddle import PaddleSplitAdapter
from .tinygrad import TinygradSplitAdapter
from .torch import TorchSplitAdapter


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
    if name == "torch":
        return TorchSplitAdapter()
    if name == "jax":
        return JaxSplitAdapter()
    if name == "tinygrad":
        return TinygradSplitAdapter()
    if name == "mlx":
        return MlxSplitAdapter()
    if name == "paddle":
        return PaddleSplitAdapter()
    if name in {"tf", "tensorflow"}:
        # Importing the TF adapter probes TensorFlow's private op-callback
        # module. Keep that optional native runtime out of tinygrad/JAX/Paddle
        # split subprocesses until a TensorFlow adapter is actually selected.
        from .tf import TfSplitAdapter

        return TfSplitAdapter()
    raise SplitUnsupportedError(
        f"backend={name!r} does not have a split adapter.",
        context=SplitErrorContext(
            backend=name,
            split_point="",
            reason="unknown split backend adapter",
        ),
    )


def __getattr__(name: str) -> object:
    """Lazily expose the TensorFlow adapter without importing TensorFlow.

    Parameters
    ----------
    name:
        Public adapter name requested by an import consumer.

    Returns
    -------
    object
        The requested TensorFlow adapter class.

    Raises
    ------
    AttributeError
        If ``name`` is not a supported lazy adapter attribute.
    """

    if name == "TfSplitAdapter":
        from .tf import TfSplitAdapter

        return TfSplitAdapter
    raise AttributeError(name)


__all__ = [
    "JaxSplitAdapter",
    "MlxSplitAdapter",
    "PaddleSplitAdapter",
    "SegmentBundle",
    "SplitBackendAdapter",
    "TfSplitAdapter",
    "TinygradSplitAdapter",
    "TorchSplitAdapter",
    "resolve_split_adapter",
]
