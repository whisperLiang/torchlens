"""Split backend adapter resolution."""

from __future__ import annotations

from ...backends import BackendSpec
from ..errors import SplitErrorContext, SplitUnsupportedError
from .base import SegmentBundle, SplitBackendAdapter
from .jax import JaxSplitAdapter
from .mlx import MlxSplitAdapter
from .paddle import PaddleSplitAdapter
from .tf import TfSplitAdapter
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
        return TfSplitAdapter()
    raise SplitUnsupportedError(
        f"backend={name!r} does not have a split adapter.",
        context=SplitErrorContext(
            backend=name,
            split_point="",
            module_path=None,
            op_type=None,
            layer_label=None,
            reason="unknown split backend adapter",
        ),
    )


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
