"""Unsupported split adapter shells for non-Torch backends."""

from __future__ import annotations

from typing import Any

from ..errors import SplitErrorContext, SplitUnsupportedError
from .base import SegmentBundle


class UnsupportedSplitAdapter:
    """Capability shell for a backend without v1 generated-eager replay."""

    supports_replay = False
    supports_training = False
    supports_boundary_cache = False
    supports_dynamic_batch = False

    def __init__(self, name: str) -> None:
        """Create an unsupported adapter shell."""

        self.name = name

    def _unsupported(self, capability: str) -> SplitUnsupportedError:
        """Build a structured unsupported-capability error."""

        return SplitUnsupportedError(
            f"backend={self.name!r} does not support split {capability} in v1.",
            context=SplitErrorContext(
                backend=self.name,
                split_point="",
                module_path=None,
                op_type=None,
                layer_label=None,
                reason=f"unsupported split {capability}",
            ),
        )

    def is_tensor(self, value: Any) -> bool:
        """Return ``False`` without importing optional backend runtimes."""

        del value
        return False

    def shape(self, value: Any) -> tuple[int, ...] | None:
        """Return no shape for unsupported shells."""

        del value
        return None

    def dtype_name(self, value: Any) -> str | None:
        """Return no dtype for unsupported shells."""

        del value
        return None

    def requires_grad(self, value: Any) -> bool | None:
        """Return no grad metadata for unsupported shells."""

        del value
        return None

    def detach(self, value: Any) -> Any:
        """Raise for unsupported replay tensor operations."""

        del value
        raise self._unsupported("replay")

    def clone(self, value: Any) -> Any:
        """Raise for unsupported replay tensor operations."""

        del value
        raise self._unsupported("replay")

    def to_device(self, value: Any, device: Any) -> Any:
        """Raise for unsupported replay tensor operations."""

        del value, device
        raise self._unsupported("replay")

    def collate(self, values: list[Any]) -> Any:
        """Raise for unsupported boundary cache/collation."""

        del values
        raise self._unsupported("boundary cache")

    def zeros_like(self, value: Any) -> Any:
        """Raise for unsupported replay tensor operations."""

        del value
        raise self._unsupported("replay")

    def allclose(self, left: Any, right: Any, *, atol: float, rtol: float) -> bool:
        """Raise for unsupported replay tensor operations."""

        del left, right, atol, rtol
        raise self._unsupported("replay")

    def build_segments(self, graph: Any, plan: Any, spec: Any) -> SegmentBundle:
        """Raise for unsupported replay segment construction."""

        del graph, plan, spec
        raise self._unsupported("replay")


__all__ = ["UnsupportedSplitAdapter"]
